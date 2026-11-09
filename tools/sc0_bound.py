"""Bound for interaction 0's symmetric contraction as one generated kernel (Codex report v3, proposal 5, 9 November 2026).

The proposal (``FUSED_LAYER0_SC``) replaces interaction 0's cuEquivariance ``uniform_1d`` symmetric contraction of
OMAT-0 medium with a generated channel-diagonal kernel (one block per atom, ``ch = threadIdx.x``) and fuses the
1 / (density + 1) division that feeds it. Per force call it would replace (H200 round-6 breakdown):

- the division ``x = m / (D + 1)`` (``triton_poi_fused_add_div_view_21``, writes x [atoms, 2048]);
- the contraction forward ``segmented_polynomial_algo1_4`` (x, per-species weights [89, 3712] -> [atoms, 512]);
- its backward ``segmented_polynomial_bwd_algo1_4`` (x, dy [atoms, 512] -> dx [atoms, 2048]);
- the division's backward (dm = dx / (D + 1) and dD = -sum(dx * x) / (D + 1)), a Triton reduction.

A fused kernel still has to read m and write y forward, and read m and dy and write dm (and dD) backward. This tool
times, on the GPU it runs on and on interaction 0's real operands (the checkpoint's weights, the structure's
species, x captured from an eager call of the hybrid MACE calculator):

- ``proxy``: the four kernels as production runs them (cuEq's ``SegmentedPolynomial`` of the module itself; the
  division compiled by Inductor, forward and backward), kernel time only (torch.profiler);
- ``floor``: two Triton kernels that do the fused design's reads and writes and nothing else (one program per
  atom; ``floor+w`` also gathers the atom's species weight row, as a per-atom kernel must unless it groups atoms
  by species);
- ``copy``: one [atoms, 2048] copy, the GPU's streaming rate for reference.

L2 is flushed (a 256 MB fill, not counted) before the forward pair and before the backward pair, as in a force call
where other kernels run in between. ``proxy - floor`` is the most a fused kernel could save on traffic alone.

cuEq's ``uniform_1d`` is already channel-diagonal (grid [atoms], block [128] = one thread per channel), so a generated
kernel does the same arithmetic. ``--profile-once`` launches the proxy and the floor once between
cudaProfilerStart/Stop for Nsight Compute (``tools/modal/sc0_bound_job.sh``, NCU=1): the FP32 instructions cuEq's
kernels execute, at the GPU's FP32 rate, are the arithmetic floor a generated kernel keeps. (The descriptor's
``flop`` counts, recorded as ``descriptor_flops``, count every path's multiplies separately and exceed what cuEq's
kernels execute; they are not a floor.)

Usage (hybrid environment, FastMACE on PYTHONPATH):
  python tools/sc0_bound.py --checkpoint mace-omat-0-medium.model --output DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_step_time  # noqa: E402

FP32_TFLOPS = {"H200": 66.9, "H100": 66.9, "A100": 19.5, "5060 Ti": 23.7, "L40S": 91.6}  # dense FP32 (FMA = 2)
FLUSH_BYTES = 256 << 20


class _Captured(Exception):
    pass


def capture(args):
    """(module, x in the kernel's layout, indices, natoms): interaction 0's symmetric contraction and its input."""
    import torch

    from fastmace.model import loading

    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    loaded = loading.load(args.model, checkpoint=str(args.checkpoint), backend="hybrid", dtype="float32",
                          device="cuda:0", sha256=sha)
    calc = loaded.calculator
    model = calc.models[0]
    found = [(n, m) for n, m in model.products[0].named_modules()
             if type(m).__name__ == "SymmetricContraction" and type(m).__module__.startswith("cuequivariance_torch")]
    if len(found) != 1:
        raise SystemExit(f"expected one cuEq SymmetricContraction under products[0], found {[n for n, _ in found]}")
    name, sc = found[0]
    box = {}

    def hook(module, inputs):
        box["x"], box["indices"] = (t.detach().clone() for t in inputs[:2])
        raise _Captured

    handle = sc.register_forward_pre_hook(hook)
    structure, atoms = ab_step_time.read_structure(args.structure, None)
    atoms.rattle(0.05, seed=1)
    atoms.calc = calc
    try:
        atoms.get_potential_energy()
    except _Captured:
        pass
    finally:
        handle.remove()
    if "x" not in box:
        raise SystemExit("the contraction's pre-hook did not run")
    with torch.no_grad():
        x = sc.transpose_in(box["x"]).contiguous()
        weight = sc.weight if sc.projection is None else torch.einsum("zau,ab->zbu", sc.weight, sc.projection)
        weight = weight.flatten(1).detach().contiguous()
    meta = dict(structure=structure, natoms=len(atoms), checkpoint_sha256=sha, module=f"products.0.{name}",
                method=getattr(sc, "method", None), x_shape=list(x.shape), weight_shape=list(weight.shape),
                indices_dtype=str(box["indices"].dtype), x_rms=float(x.pow(2).mean().sqrt()),
                irreps_in=str(sc.irreps_in), irreps_out=str(sc.irreps_out), degree=sc.contraction_degree)
    return sc, x, weight, box["indices"], meta, loaded


def flops(sc, natoms):
    """FLOPs of the forward and of the input-gradient backward from the cuEq descriptor (None where unavailable)."""
    poly, out = sc.etp, {}
    try:
        out["forward"] = int(poly.flop(natoms))
    except Exception as e:  # noqa: BLE001
        out["forward_error"] = repr(e)
    try:
        back = poly.backward([False, True], [True])
        back = back[0] if isinstance(back, tuple) else back
        out["backward"] = int(back.flop(natoms))
    except Exception as e:  # noqa: BLE001
        out["backward_error"] = repr(e)
    out["descriptor"] = str(poly)[:4000]
    return out


def triton_kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def fwd(m_ptr, d_ptr, w_ptr, idx_ptr, y_ptr, IN: tl.constexpr, OUT: tl.constexpr, W: tl.constexpr,
            WBLOCK: tl.constexpr, WITH_W: tl.constexpr):
        a = tl.program_id(0).to(tl.int64)
        s = 1.0 / (tl.load(d_ptr + a) + 1.0)
        cols = tl.arange(0, OUT)
        acc = tl.zeros([OUT], tl.float32)
        for k in tl.static_range(IN // OUT):
            acc += tl.load(m_ptr + a * IN + k * OUT + cols) * s
        if WITH_W:
            sp = tl.load(idx_ptr + a).to(tl.int64)
            wc = tl.arange(0, WBLOCK)
            acc += tl.sum(tl.load(w_ptr + sp * W + wc, mask=wc < W, other=0.0))
        tl.store(y_ptr + a * OUT + cols, acc)

    @triton.jit
    def bwd(m_ptr, d_ptr, g_ptr, w_ptr, idx_ptr, dm_ptr, dd_ptr, IN: tl.constexpr, OUT: tl.constexpr,
            W: tl.constexpr, WBLOCK: tl.constexpr, WITH_W: tl.constexpr):
        a = tl.program_id(0).to(tl.int64)
        s = 1.0 / (tl.load(d_ptr + a) + 1.0)
        cols = tl.arange(0, OUT)
        g = tl.load(g_ptr + a * OUT + cols)
        if WITH_W:
            sp = tl.load(idx_ptr + a).to(tl.int64)
            wc = tl.arange(0, WBLOCK)
            g += tl.sum(tl.load(w_ptr + sp * W + wc, mask=wc < W, other=0.0))
        total = tl.zeros([OUT], tl.float32)
        for k in tl.static_range(IN // OUT):
            x = tl.load(m_ptr + a * IN + k * OUT + cols) * s
            dx = g * x
            total += dx * x
            tl.store(dm_ptr + a * IN + k * OUT + cols, dx * s)
        tl.store(dd_ptr + a, -tl.sum(total) * s)

    return fwd, bwd


def kernel_us(fn, iterations, flush, select):
    """Mean device time per iteration of the kernels whose name ``select`` accepts: {kernel name: us}."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iterations):
            fn(flush)
        torch.cuda.synchronize()
    out = {}
    for e in prof.key_averages():
        t = getattr(e, "self_device_time_total", None)
        if t is None:
            t = e.self_cuda_time_total
        if t and select(e.key):
            out[e.key] = out.get(e.key, 0.0) + t / iterations
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="omat0-medium")
    p.add_argument("--structure", default="results/loup-a100-20261103/SrTiO3.cif@10x10x10")
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--iterations", type=int, default=10, help="calls per profiled round")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile-once", action="store_true",
                   help="no timings: warm up, then one proxy and one floor call between cudaProfilerStart/Stop")
    args = p.parse_args(argv)
    ab_step_time.prepare_fastmace_hybrid()
    import torch

    if ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    args.output.mkdir(parents=True, exist_ok=True)
    sc, x, weight, indices, meta, loaded = capture(args)
    natoms, width = x.shape
    out_width = int(sc.irreps_out.dim)
    del loaded
    torch.cuda.empty_cache()
    gen = torch.Generator(device="cuda").manual_seed(0)
    density = torch.rand(natoms, 1, device="cuda", generator=gen) * 2.0  # D >= 0; values do not change the timing
    m = (x * (density + 1.0)).contiguous()
    poly = sc.f
    with torch.no_grad():
        y_ref = poly([weight, x], input_indices={0: indices})[0]
    g = (torch.randn(y_ref.shape, device="cuda", generator=gen) * y_ref.std()).contiguous()
    assert y_ref.shape == (natoms, out_width), (y_ref.shape, out_width)
    flush_buf = torch.empty(FLUSH_BYTES // 4, device="cuda")

    divide = torch.compile(lambda a, d: a / (d + 1.0), dynamic=False)

    def proxy(flush):
        mm = m.detach().requires_grad_(True)
        dd = density.detach().requires_grad_(True)
        flush_buf.fill_(flush)
        xx = divide(mm, dd)
        y = poly([weight, xx], input_indices={0: indices})[0]
        flush_buf.fill_(flush)
        torch.autograd.backward(y, g)

    fwd_k, bwd_k = triton_kernels()
    w_width = weight.shape[1]
    wblock = 1 << (w_width - 1).bit_length()
    y_out = torch.empty(natoms, out_width, device="cuda")
    dm = torch.empty_like(m)
    dd_out = torch.empty(natoms, device="cuda")
    d_flat = density.reshape(natoms).contiguous()
    idx = indices.contiguous()
    assert width % out_width == 0

    def floor(with_w):
        def run(flush):
            flush_buf.fill_(flush)
            fwd_k[(natoms,)](m, d_flat, weight, idx, y_out, IN=width, OUT=out_width, W=w_width, WBLOCK=wblock,
                             WITH_W=with_w, num_warps=4)
            flush_buf.fill_(flush)
            bwd_k[(natoms,)](m, d_flat, g, weight, idx, dm, dd_out, IN=width, OUT=out_width, W=w_width,
                             WBLOCK=wblock, WITH_W=with_w, num_warps=4)
        return run

    copy_dst = torch.empty_like(m)

    def copy(flush):
        flush_buf.fill_(flush)
        copy_dst.copy_(m)

    cases = {"proxy": proxy, "floor": floor(False), "floor+w": floor(True), "copy": copy}
    not_fill = lambda k: "FillFunctor" not in k and "fill" not in k.split("<")[0].lower()  # noqa: E731
    for fn in cases.values():  # compile, autotune cuEq's run-time choice, load the Triton kernels
        for _ in range(3):
            fn(0.0)
    torch.cuda.synchronize()
    if args.profile_once:  # Nsight Compute (--profile-from-start off) sees: fills, proxy's 4 kernels, fills, floor's 2
        torch.cuda.profiler.start()
        proxy(0.0)
        cases["floor"](0.0)
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        print("profiled once: proxy, floor", flush=True)
        return
    per_round = {k: [] for k in cases}
    kernels = {k: {} for k in cases}
    order = list(cases)
    for r in range(args.rounds):
        for key in (order if r % 2 == 0 else order[::-1]):
            times = kernel_us(cases[key], args.iterations, float(r), not_fill)
            per_round[key].append(sum(times.values()))
            for name, t in times.items():
                kernels[key].setdefault(name, []).append(t)
    med = {k: statistics.median(v) for k, v in per_round.items()}
    kern = {k: {name: round(statistics.median(v), 2) for name, v in d.items()} for k, d in kernels.items()}
    fwd_bytes = (m.numel() + y_out.numel() + natoms) * 4
    bwd_bytes = (m.numel() + g.numel() + dm.numel() + 2 * natoms) * 4
    gpu = torch.cuda.get_device_name()
    peak = next((v for k, v in FP32_TFLOPS.items() if k in gpu), None)
    fl = flops(sc, natoms)
    flop_rate = {k: round(fl[k] / (peak * 1e6), 2) for k in ("forward", "backward") if k in fl and peak}
    result = dict(meta, tool="sc0_bound", created_utc=datetime.now(timezone.utc).isoformat(),
                  host=os.uname().nodename, gpu=gpu, sms=torch.cuda.get_device_properties(0).multi_processor_count,
                  torch=torch.__version__, fastmace_commit=ab_step_time.fastmace_commit(), rounds=args.rounds,
                  iterations=args.iterations, us=({k: round(v, 2) for k, v in med.items()}),
                  us_spread={k: [round(min(v), 2), round(max(v), 2)] for k, v in per_round.items()},
                  kernels_us=kern, floor_bytes=dict(forward=fwd_bytes, backward=bwd_bytes),
                  floor_tb_per_s=round((fwd_bytes + bwd_bytes) / med["floor"] / 1e6, 3),
                  copy_tb_per_s=round(2 * m.numel() * 4 / med["copy"] / 1e6, 3),
                  saving_us=dict(floor=round(med["proxy"] - med["floor"], 2),
                                 floor_w=round(med["proxy"] - med["floor+w"], 2)),
                  descriptor_flops=fl, fp32_tflops=peak, descriptor_flops_at_peak_us=flop_rate)
    (args.output / "sc0_bound.json").write_text(json.dumps(result, indent=1) + "\n")
    for key in cases:
        print(f"{key}: {med[key]:.1f} us  " + "  ".join(f"{n[:60]} {t:.1f}" for n, t in kern[key].items()), flush=True)
    print(f"floor {result['floor_tb_per_s']} TB/s, copy {result['copy_tb_per_s']} TB/s; saving at most "
          f"{result['saving_us']['floor']} us (floor), {result['saving_us']['floor_w']} us (floor+w); "
          f"descriptor flops {fl.get('forward')} / {fl.get('backward')} ({flop_rate} us at peak, not a floor)", flush=True)


if __name__ == "__main__":
    main()
