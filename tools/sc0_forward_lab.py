"""Lab, step 1 of the interaction-0 symmetric contraction as a generated kernel (9 November 2026; the bound is
results/sc0-bound-20261109): the forward only, as straight-line NVRTC code generated from cuEq's own descriptor.

The bound found cuEq's ``uniform_1d`` forward issue bound on the H200 at 1,381 FP32 instructions per (atom, channel),
while the polynomial has only 480 distinct (monomial, output, weight) terms over 410 monomials. The generated kernel
(one block of 128 threads per atom, ``ch = threadIdx.x``, cuEq's own layouts):

- loads the 16 components of x (``x = m * s`` with ``s = 1 / (D + 1)`` when FUSED: the division fused);
- forms every monomial once (degree 2: one FMUL; degree 3: one FMUL on a degree-2 prefix);
- per (weight a, output C) group, sums ``coef * monomial`` with the merged coefficients as immediates
  (the descriptor's paths with the same sorted monomial, a and C summed), then ``y_C += w[species, a, ch] * t``.

Variants: ``gen`` (x given, as cuEq), ``gen-fused`` (m and D given). It is compared with cuEq's forward (``cueq``)
and with the division plus cuEq's forward (``cueq+div``, the production pair), kernel time by torch.profiler,
L2 flushed before each call. The outputs are compared with cuEq's (not bitwise: another summation order).
The adoption bar for going on to the backward: about 30 us for the fused forward on the H200 (cuEq 40 us).

Usage (hybrid environment, FastMACE on PYTHONPATH):
  python tools/sc0_forward_lab.py --checkpoint mace-omat-0-medium.model --output DIR [--profile-once]
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


class Generated:
    def __init__(self, source):
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check, compile_cubin

        self.source = source
        cubin = compile_cubin(source, torch_capability())
        import torch
        torch.cuda.current_stream()
        self._module = _check(driver.cuModuleLoadData(cubin))
        self._fn = _check(driver.cuModuleGetFunction(self._module, b"sc0_fwd"))
        attr = driver.CUfunction_attribute
        self.registers = int(_check(driver.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_NUM_REGS, self._fn)))
        self.local_bytes = int(_check(driver.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, self._fn)))

    def __call__(self, m, d, w, sp, y):
        import torch
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check
        stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
        values = (m.data_ptr(), d.data_ptr(), w.data_ptr(), sp.data_ptr(), y.data_ptr())
        types = (ctypes.c_void_p,) * 5
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
    torch.cuda.synchronize()
    scale = float(y_ref.abs().max())

    def err(a, b):
        return dict(max_abs=float((a - b).abs().max()), max_rel_to_max=float((a - b).abs().max()) / scale,
                    rms_rel=float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()))

    check = {"gen_vs_cueq": err(y_gen, y_ref), "gen_fused_vs_cueq_div": err(y_fused, y_ref_div)}
    print("check", json.dumps(check), flush=True)
    if check["gen_vs_cueq"]["rms_rel"] > 1e-4 or check["gen_fused_vs_cueq_div"]["rms_rel"] > 1e-4:
        (args.output / "check.json").write_text(json.dumps(check, indent=1) + "\n")
        raise SystemExit("the generated forward does not match cuEq's")
    flush_buf = torch.empty(sc0_bound.FLUSH_BYTES // 4, device="cuda")
    out = torch.empty_like(y_ref)

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
    }
    with torch.no_grad():
        for fn in cases.values():
            for _ in range(3):
                fn(0.0)
        torch.cuda.synchronize()
        if args.profile_once:
            torch.cuda.profiler.start()
            for key in ("cueq", "gen", "gen-fused"):
                cases[key](0.0)
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            print("profiled once: cueq, gen, gen-fused", flush=True)
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
    result = dict(meta, tool="sc0_forward_lab", created_utc=datetime.now(timezone.utc).isoformat(),
                  host=os.uname().nodename, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  fastmace_commit=ab_step_time.fastmace_commit(), rounds=args.rounds, iterations=args.iterations,
                  counts=counts, registers={k: v.registers for k, v in kernels.items()},
                  local_bytes={k: v.local_bytes for k, v in kernels.items()}, check=check,
                  us={k: round(v, 2) for k, v in med.items()},
                  us_spread={k: [round(min(v), 2), round(max(v), 2)] for k, v in per_round.items()},
                  kernels_us={k: {n: round(statistics.median(v), 2) for n, v in d.items()} for k, d in kern.items()},
                  saving_us=round(med["cueq+div"] - med["gen-fused"], 2),
                  gen_over_cueq=round(med["gen"] / med["cueq"], 4))
    (args.output / "sc0_forward_lab.json").write_text(json.dumps(result, indent=1) + "\n")
    for key in cases:
        print(f"{key}: {med[key]:.1f} us  " + "  ".join(f"{n[:50]} {statistics.median(t):.1f}"
                                                    for n, t in kern[key].items()), flush=True)
    print(f"registers {result['registers']}, local bytes {result['local_bytes']}, counts {counts}; "
          f"gen/cueq {result['gen_over_cueq']}, fused saves {result['saving_us']} us of the forward pair", flush=True)


if __name__ == "__main__":
    main()
