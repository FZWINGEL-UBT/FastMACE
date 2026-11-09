"""Lab for interaction 0's symmetric contraction as generated kernels (9 November 2026; the bound is
results/sc0-bound-20261109): forward and backward as straight-line NVRTC code generated from cuEq's own descriptor,
the 1 / (density + 1) division and its backward fused.

The bound found cuEq's ``uniform_1d`` kernels issue bound on the H200 at 1,381 / 2,926 FP32 instructions per
(atom, channel), while the polynomial has only 480 distinct (monomial, output, weight) terms over 410 monomials.
Both generated kernels run one block of 128 threads per atom (``ch = threadIdx.x``) on cuEq's own layouts.

Forward (``sc0_fwd``): x = m * s (s = 1 / (D + 1) when fused); monomials formed once, in consumption order (each
degree-2 prefix, then its degree-3 monomials); every term goes into its (weight a, output C) accumulator with the
merged coefficient as an immediate; y_C = sum_a w[species, a, ch] * t_aC.

Backward (``sc0_bwd``, always fused): q_aC = w_a * dy_C; per monomial the adjoint h = sum coef * q; for a prefix
(i, j) with p = x_i x_j: each degree-3 monomial (i, j, k) adds h p to dx_k and h x_k to the pair adjoint H, the
degree-2 monomial (i, j) adds its h to H, then dx_i += H x_j and dx_j += H x_i (repeated indices come out right);
dm = dx * s in the species GEMM's [atoms, 16, 128] row layout, and dD = -s * sum(dx * x) by a block reduction.

Cases (kernel time by torch.profiler, L2 flushed before each pass): ``cueq`` / ``cueq+div`` / ``gen`` /
``gen-fused`` forwards; ``proxy`` (production's four kernels with autograd: division, cuEq forward, cuEq backward,
division backward) against ``gen-pair`` (fused forward + fused backward). Outputs, dm and dD are compared with cuEq
through autograd (not bitwise: another summation order).

Usage (hybrid environment, FastMACE on PYTHONPATH):
  python tools/sc0_lab.py --checkpoint mace-omat-0-medium.model --output DIR [--profile-once]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_step_time  # noqa: E402
import sc0_bound  # noqa: E402

THREADS = 128


def merged_terms(poly):
    """{(a, monomial (sorted b segments), C): coefficient}, the descriptor's paths with equal keys summed."""
    terms = defaultdict(float)
    for op, stp in poly.operations:
        buffers = op.buffers
        out_buffer = buffers[-1]
        for path in stp.paths:
            a, bs, c = None, [], None
            for buffer, index in zip(buffers, path.indices):
                if buffer == out_buffer and c is None and buffer >= poly.num_inputs:
                    c = index
                elif buffer == 0:
                    a = index
                else:
                    bs.append(index)
            coef = float(path.coefficients.reshape(-1)[0]) if hasattr(path.coefficients, "reshape") \
                else float(path.coefficients)
            terms[(a, tuple(sorted(bs)), c)] += coef
    return {k: v for k, v in terms.items() if v != 0.0}


def _lit(value):
    import numpy as np
    return float(np.float32(value)).hex() + "f"


def render(terms, segments_in, segments_w, segments_out, fused):
    """CUDA source of the generated forward; returns (source, counts)."""
    monos = sorted({m for _, m, _ in terms}, key=lambda m: (len(m), m))
    lines = [f"// generated: {len(terms)} terms, {len(monos)} monomials, fused division: {bool(fused)}",
             'extern "C" __global__ void __launch_bounds__(128) sc0_fwd(',
             "    const float* __restrict__ m, const float* __restrict__ d, const float* __restrict__ w,",
             "    const int* __restrict__ sp, float* __restrict__ y) {",
             "  const int a = blockIdx.x, ch = threadIdx.x;",
             f"  const float* xr = m + (long long)a * {segments_in * THREADS} + ch;",
             f"  const float* wr = w + (long long)sp[a] * {segments_w * THREADS} + ch;"]
    if fused:
        lines.append("  const float s = 1.0f / (d[a] + 1.0f);")
    for i in range(segments_in):
        lines.append(f"  const float x{i} = __ldg(xr + {i * THREADS})" + (" * s;" if fused else ";"))
    # Consumption order: degree 1 and 2 monomials, then each degree-2 prefix (i, j) with its degree-3 monomials, so a
    # pair product dies after its triples; every term goes into its (weight a, output C) accumulator at once.
    by_mono = defaultdict(list)
    for (wa, mono, c), coef in terms.items():
        by_mono[mono].append((wa, c, coef))
    groups = sorted({(wa, c) for wa, _, c in terms})
    acc = {g: f"t{g[0]}_{g[1]}" for g in groups}
    lines.append("  float " + ", ".join(acc[g] for g in groups) + ";")
    started = set()
    fmul = ffma = 0

    def consume(mono, expr):
        nonlocal ffma
        for wa, c, coef in sorted(by_mono.get(mono, ())):
            t = acc[(wa, c)]
            if (wa, c) in started:
                lines.append(f"  {t} = fmaf({_lit(coef)}, {expr}, {t});")
            else:
                lines.append(f"  {t} = {_lit(coef)} * {expr};")
                started.add((wa, c))
            ffma += 1

    for mono in (mm for mm in monos if len(mm) == 1):
        consume(mono, f"x{mono[0]}")
    prefixes = sorted({mm[:2] for mm in monos if len(mm) >= 2})
    for pre in prefixes:
        lines.append(f"  {{ const float p = x{pre[0]} * x{pre[1]};")
        fmul += 1
        if pre in by_mono:
            consume(pre, "p")
        for mono in (mm for mm in monos if len(mm) == 3 and mm[:2] == pre):
            lines.append(f"  {{ const float q = p * x{mono[2]};")
            fmul += 1
            consume(mono, "q")
            lines.append("  }")
        lines.append("  }")
    for c in range(segments_out):
        lines.append(f"  float y{c} = 0.0f;")
    for wa, c in groups:
        lines.append(f"  y{c} = fmaf(__ldg(wr + {wa * THREADS}), {acc[(wa, c)]}, y{c});")
        ffma += 1
    for c in range(segments_out):
        lines.append(f"  y[(long long)a * {segments_out * THREADS} + {c * THREADS} + ch] = y{c};")
    lines.append("}")
    counts = dict(terms=len(terms), monomials=len(monos), groups=len(groups), fmul_monomials=fmul,
                  fp32_per_channel=fmul + ffma + (segments_in if fused else 0))
    return "\n".join(lines) + "\n", counts


def render_bwd(terms, segments_in, segments_w, segments_out):
    """CUDA source of the generated, fused backward (dm, dD from dy); returns (source, counts)."""
    monos = sorted({m for _, m, _ in terms}, key=lambda m: (len(m), m))
    by_mono = defaultdict(list)
    for (wa, mono, c), coef in terms.items():
        by_mono[mono].append((wa, c, coef))
    groups = sorted({(wa, c) for wa, _, c in terms})
    q = {g: f"q{g[0]}_{g[1]}" for g in groups}
    lines = [f"// generated backward: {len(terms)} terms, {len(monos)} monomials, {len(groups)} groups",
             'extern "C" __global__ void __launch_bounds__(128) sc0_bwd(',
             "    const float* __restrict__ m, const float* __restrict__ d, const float* __restrict__ w,",
             "    const int* __restrict__ sp, const float* __restrict__ g, float* __restrict__ dm,",
             "    float* __restrict__ dd) {",
             "  const int a = blockIdx.x, ch = threadIdx.x;",
             f"  const float* xr = m + (long long)a * {segments_in * THREADS} + ch;",
             f"  const float* wr = w + (long long)sp[a] * {segments_w * THREADS} + ch;",
             f"  const float* gr = g + (long long)a * {segments_out * THREADS} + ch;",
             "  const float s = 1.0f / (d[a] + 1.0f);"]
    for i in range(segments_in):
        lines.append(f"  const float x{i} = __ldg(xr + {i * THREADS}) * s;")
    for c in range(segments_out):
        lines.append(f"  const float g{c} = __ldg(gr + {c * THREADS});")
    for wa, c in groups:
        lines.append(f"  const float {q[(wa, c)]} = __ldg(wr + {wa * THREADS}) * g{c};")
    for i in range(segments_in):
        lines.append(f"  float dx{i} = 0.0f;")
    fmul, ffma = len(groups) + segments_in, 0

    def adjoint(mono, target, started):
        """Statements adding sum coef * q over mono's terms to ``target``; returns whether target is now set."""
        nonlocal ffma
        for wa, c, coef in sorted(by_mono.get(mono, ())):
            if started:
                lines.append(f"    {target} = fmaf({_lit(coef)}, {q[(wa, c)]}, {target});")
            else:
                lines.append(f"    {target} = {_lit(coef)} * {q[(wa, c)]};")
                started = True
            ffma += 1
        return started

    for mono in (mm for mm in monos if len(mm) == 1):
        adjoint(mono, f"dx{mono[0]}", True)
    for pre in sorted({mm[:2] for mm in monos if len(mm) >= 2}):
        i, j = pre
        lines.append(f"  {{ const float p = x{i} * x{j}; float H = 0.0f;")
        fmul += 1
        h_set = adjoint(pre, "H", False)
        for mono in (mm for mm in monos if len(mm) == 3 and mm[:2] == pre):
            k = mono[2]
            lines.append("  { float h;")
            adjoint(mono, "h", False)
            lines.append(f"    dx{k} = fmaf(h, p, dx{k});")
            lines.append(f"    H = {'fmaf(h, x%d, H)' % k if h_set else 'h * x%d' % k}; }}")
            h_set = True
            ffma += 2
        lines.append(f"    dx{i} = fmaf(H, x{j}, dx{i}); dx{j} = fmaf(H, x{i}, dx{j}); }}")
        ffma += 2
    lines.append("  float dot = 0.0f;")
    for i in range(segments_in):
        lines.append(f"  dot = fmaf(dx{i}, x{i}, dot);")
        lines.append(f"  dm[(long long)a * {segments_in * THREADS} + {i * THREADS} + ch] = dx{i} * s;")
    ffma += segments_in
    fmul += segments_in
    lines += ["  #pragma unroll",
              "  for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);",
              "  __shared__ float red[4];",
              "  if ((ch & 31) == 0) red[ch >> 5] = dot;",
              "  __syncthreads();",
              "  if (ch == 0) dd[a] = -s * (red[0] + red[1] + red[2] + red[3]);",
              "}"]
    counts = dict(terms=len(terms), monomials=len(monos), groups=len(groups), fp32_per_channel=fmul + ffma)
    return "\n".join(lines) + "\n", counts


class Generated:
    def __init__(self, source, name=b"sc0_fwd"):
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check, compile_cubin

        self.source = source
        cubin = compile_cubin(source, torch_capability())
        import torch
        torch.cuda.current_stream()
        self._module = _check(driver.cuModuleLoadData(cubin))
        self._fn = _check(driver.cuModuleGetFunction(self._module, name))
        attr = driver.CUfunction_attribute
        self.registers = int(_check(driver.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_NUM_REGS, self._fn)))
        self.local_bytes = int(_check(driver.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, self._fn)))

    def __call__(self, *tensors):
        import torch
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check
        stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
        m = tensors[0]
        values = tuple(t.data_ptr() for t in tensors)
        types = (ctypes.c_void_p,) * len(tensors)
        _check(driver.cuLaunchKernel(self._fn, m.shape[0], 1, 1, THREADS, 1, 1, 0, stream, (values, types), 0))


def torch_capability():
    import torch
    return torch.cuda.get_device_capability()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="omat0-medium")
    p.add_argument("--structure", default="results/loup-a100-20261103/SrTiO3.cif@10x10x10")
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile-once", action="store_true")
    args = p.parse_args(argv)
    ab_step_time.prepare_fastmace_hybrid()
    import torch

    if ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    args.output.mkdir(parents=True, exist_ok=True)
    sc, x, weight, indices, meta, loaded = sc0_bound.capture(args)
    del loaded
    torch.cuda.empty_cache()
    natoms = x.shape[0]
    poly = getattr(sc.etp, "polynomial", sc.etp)
    seg_in, seg_w = x.shape[1] // THREADS, weight.shape[1] // THREADS
    seg_out = int(sc.irreps_out.dim) // THREADS
    terms = merged_terms(poly)
    kernels, counts = {}, {}
    for name, fused in (("gen", False), ("gen-fused", True)):
        source, counts[name] = render(terms, seg_in, seg_w, seg_out, fused)
        (args.output / f"{name}.cu").write_text(source)
        kernels[name] = Generated(source)
    source, counts["gen-bwd"] = render_bwd(terms, seg_in, seg_w, seg_out)
    (args.output / "gen-bwd.cu").write_text(source)
    kernels["gen-bwd"] = Generated(source, b"sc0_bwd")
    # The factored invariant circuit (fastmace.kernels.cuda.sc_factored): the same contraction through shared CG
    # pair tensors, the change of basis folded into the weights.
    from fastmace.kernels.cuda import sc_factored

    factored = sc_factored.factor(terms, seg_in, seg_w, seg_out)
    if factored is None:
        raise SystemExit("sc_factored: the candidates do not span the descriptor's basis")
    prog, feats, basis_k = factored
    counts["fac"] = dict(sc_factored.counts(prog, feats), check=sc_factored.check(terms, prog, feats, basis_k, seg_in))
    w_fac = sc_factored.fold_weights(weight, basis_k, seg_w, THREADS)
    source = sc_factored.render_forward(prog, feats, seg_in, seg_w, seg_out, THREADS, name="sc0_fwd")
    (args.output / "fac-fused.cu").write_text(source)
    kernels["fac-fused"] = Generated(source)
    source = sc_factored.render_backward(prog, feats, seg_in, seg_w, seg_out, THREADS, name="sc0_bwd")
    (args.output / "fac-bwd.cu").write_text(source)
    kernels["fac-bwd"] = Generated(source, b"sc0_bwd")
    print("factored", json.dumps(counts["fac"]), flush=True)
    gen = torch.Generator(device="cuda").manual_seed(0)
    density = torch.rand(natoms, 1, device="cuda", generator=gen) * 2.0
    m = (x * (density + 1.0)).contiguous()
    d_flat = density.reshape(natoms).contiguous()
    idx = indices.to(torch.int32).contiguous()
    f = sc.f
    divide = torch.compile(lambda a, dd: a / (dd + 1.0), dynamic=False)
    with torch.no_grad():
        y_ref = f([weight, x], input_indices={0: indices})[0]
        x_div = divide(m, density)
        y_ref_div = f([weight, x_div], input_indices={0: indices})[0]
    y_gen = torch.empty_like(y_ref)
    y_fused = torch.empty_like(y_ref)
    kernels["gen"](x, d_flat, weight, idx, y_gen)
    kernels["gen-fused"](m, d_flat, weight, idx, y_fused)
    y_fac = torch.empty_like(y_ref)
    kernels["fac-fused"](m, d_flat, w_fac, idx, y_fac)
    torch.cuda.synchronize()
    scale = float(y_ref.abs().max())

    def err(a, b):
        return dict(max_abs=float((a - b).abs().max()), max_rel_to_max=float((a - b).abs().max()) / scale,
                    rms_rel=float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()))

    g = (torch.randn(y_ref.shape, device="cuda", generator=gen) * y_ref.std()).contiguous()
    mm = m.detach().requires_grad_(True)
    dd = density.detach().requires_grad_(True)
    y_auto = f([weight, divide(mm, dd)], input_indices={0: indices})[0]
    dm_ref, dd_ref = torch.autograd.grad(y_auto, (mm, dd), g)
    dm_gen = torch.empty_like(m)
    dd_gen = torch.empty(natoms, device="cuda")
    kernels["gen-bwd"](m, d_flat, weight, idx, g, dm_gen, dd_gen)
    dm_fac = torch.empty_like(m)
    dd_fac = torch.empty(natoms, device="cuda")
    kernels["fac-bwd"](m, d_flat, w_fac, idx, g, dm_fac, dd_fac)
    torch.cuda.synchronize()

    def err(a, b):
        a, b = a.reshape(-1), b.reshape(-1)
        return dict(max_abs=float((a - b).abs().max()), max_rel_to_max=float((a - b).abs().max() / b.abs().max()),
                    rms_rel=float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()))

    check = {"gen_vs_cueq": err(y_gen, y_ref), "gen_fused_vs_cueq_div": err(y_fused, y_ref_div),
             "dm_vs_autograd": err(dm_gen, dm_ref), "dD_vs_autograd": err(dd_gen, dd_ref),
             "fac_vs_cueq_div": err(y_fac, y_ref_div), "fac_dm_vs_autograd": err(dm_fac, dm_ref),
             "fac_dD_vs_autograd": err(dd_fac, dd_ref)}
    print("check", json.dumps(check), flush=True)
    if any(v["rms_rel"] > 1e-4 for v in check.values()):
        (args.output / "check.json").write_text(json.dumps(check, indent=1) + "\n")
        raise SystemExit("the generated kernels do not match cuEq's")
    del y_auto, dm_ref, dd_ref, mm, dd
    flush_buf = torch.empty(sc0_bound.FLUSH_BYTES // 4, device="cuda")
    out = torch.empty_like(y_ref)

    def proxy(flush):
        m2 = m.detach().requires_grad_(True)
        d2 = density.detach().requires_grad_(True)
        flush_buf.fill_(flush)
        with torch.enable_grad():
            y = f([weight, divide(m2, d2)], input_indices={0: indices})[0]
            flush_buf.fill_(flush)
            torch.autograd.backward(y, g)

    def gen_pair(flush):
        flush_buf.fill_(flush)
        kernels["gen-fused"](m, d_flat, weight, idx, out)
        flush_buf.fill_(flush)
        kernels["gen-bwd"](m, d_flat, weight, idx, g, dm_gen, dd_gen)

    def fac_pair(flush):
        flush_buf.fill_(flush)
        kernels["fac-fused"](m, d_flat, w_fac, idx, out)
        flush_buf.fill_(flush)
        kernels["fac-bwd"](m, d_flat, w_fac, idx, g, dm_fac, dd_fac)

    def case(fn):
        def run(flush):
            flush_buf.fill_(flush)
            fn()
        return run

    cases = {
        "cueq": case(lambda: f([weight, x], input_indices={0: indices})),
        "cueq+div": case(lambda: f([weight, divide(m, density)], input_indices={0: indices})),
        "gen": case(lambda: kernels["gen"](x, d_flat, weight, idx, out)),
        "gen-fused": case(lambda: kernels["gen-fused"](m, d_flat, weight, idx, out)),
        "proxy": proxy,
        "gen-pair": gen_pair,
        "fac-fused": case(lambda: kernels["fac-fused"](m, d_flat, w_fac, idx, out)),
        "fac-pair": fac_pair,
    }
    with torch.no_grad():
        for fn in cases.values():
            for _ in range(3):
                fn(0.0)
        torch.cuda.synchronize()
        if args.profile_once:
            torch.cuda.profiler.start()
            for key in ("proxy", "gen-pair", "fac-pair"):
                cases[key](0.0)
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            print("profiled once: proxy, gen-pair, fac-pair", flush=True)
            return
        not_fill = lambda k: "FillFunctor" not in k  # noqa: E731
        per_round = {k: [] for k in cases}
        kern = {k: {} for k in cases}
        order = list(cases)
        for r in range(args.rounds):
            for key in (order if r % 2 == 0 else order[::-1]):
                times = sc0_bound.kernel_us(cases[key], args.iterations, float(r), not_fill)
                per_round[key].append(sum(times.values()))
                for name, t in times.items():
                    kern[key].setdefault(name, []).append(t)
    med = {k: statistics.median(v) for k, v in per_round.items()}
    backward = lambda d: sum(t for n, t in d.items() if "bwd" in n or "triton_red" in n)  # noqa: E731
    bwd_med = {k: statistics.median([backward({n: v[r] for n, v in kern[k].items() if r < len(v)})
                                     for r in range(args.rounds)]) for k in ("proxy", "gen-pair", "fac-pair")}
    result = dict(meta, tool="sc0_lab", created_utc=datetime.now(timezone.utc).isoformat(),
                  host=os.uname().nodename, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  fastmace_commit=ab_step_time.fastmace_commit(), rounds=args.rounds, iterations=args.iterations,
                  counts=counts, registers={k: v.registers for k, v in kernels.items()},
                  local_bytes={k: v.local_bytes for k, v in kernels.items()}, check=check,
                  us={k: round(v, 2) for k, v in med.items()},
                  us_spread={k: [round(min(v), 2), round(max(v), 2)] for k, v in per_round.items()},
                  kernels_us={k: {n: round(statistics.median(v), 2) for n, v in d.items()} for k, d in kern.items()},
                  saving_us=round(med["cueq+div"] - med["gen-fused"], 2),
                  gen_over_cueq=round(med["gen"] / med["cueq"], 4),
                  backward_us={k: round(v, 2) for k, v in bwd_med.items()},
                  bucket_saving_us=round(med["proxy"] - med["gen-pair"], 2),
                  fac_over_gen_fused=round(med["fac-fused"] / med["gen-fused"], 4),
                  fac_pair_over_gen_pair=round(med["fac-pair"] / med["gen-pair"], 4))
    (args.output / "sc0_lab.json").write_text(json.dumps(result, indent=1) + "\n")
    for key in cases:
        print(f"{key}: {med[key]:.1f} us  " + "  ".join(f"{n[:50]} {statistics.median(t):.1f}"
                                                    for n, t in kern[key].items()), flush=True)
    print(f"registers {result['registers']}, local bytes {result['local_bytes']}, counts {counts}; "
          f"gen/cueq {result['gen_over_cueq']}, fused saves {result['saving_us']} us of the forward pair; "
          f"backward {result['backward_us']}; bucket saving {result['bucket_saving_us']} us; "
          f"factored forward {result['fac_over_gen_fused']}, pair {result['fac_pair_over_gen_pair']}", flush=True)


if __name__ == "__main__":
    main()
