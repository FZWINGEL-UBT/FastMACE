"""Stage-1 screen: FP32-accurate BF16x6 Triton GEMM against cuBLAS IEEE FP32 (OPT-T01, 23 October 2026).

Kernel and timing follow FastMACE tools/gemm_precision_bench.py (main): the same Triton GEMM with the same
7 fixed tile configs, only ``input_precision`` changed; 5 warm-up calls, then the median of CUDA-event
timings; the fastest config per mode (a screen's upper bound). Row-major float32 inputs, fixed seed.

Errors against a float64 product of the same float32 inputs:
  max_rel   max |c - c64| / max |c64|
  frob_rel  ||c - c64||_F / ||c64||_F
  cancel    max_ij |c - c64| / (sum_k |a_ik b_kj| + 1e-30), sum in float64 as |A| @ |B|
on a random set (randn) and on an alternating-sign cancellation set: A[:, 2k+1] = -A[:, 2k] (1 + 1e-3 u),
B[2k+1, :] = B[2k, :], so every output is ~1e-3 of sum |ab|.

Usage: python probe_bf16x6.py OUTPUT_JSON
"""

import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

import torch
import triton
import triton.language as tl

CONFIGS = ((64, 64, 32, 4, 3), (128, 64, 32, 4, 3), (64, 128, 32, 4, 3), (128, 128, 32, 8, 3),
           (32, 64, 32, 4, 3), (64, 64, 16, 4, 3), (32, 32, 16, 2, 3))
TRITON = {"triton_ieee": "ieee", "triton_tf32x3": "tf32x3", "triton_bf16x3": "bf16x3", "triton_bf16x6": "bf16x6"}
FP32_CLASS = ("cublas_ieee", "triton_ieee", "triton_tf32x3", "triton_bf16x3", "triton_bf16x6")
SHAPES = [(4096, 4096, 4096)] + [(m, 512, 512) for m in (216, 1512, 5832, 13608)]
SEED = 20261023


@triton.jit
def gemm(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PRECISION: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    rm = (pm * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn = (pn * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start in range(0, K, BLOCK_K):
        kk = start + rk
        a = tl.load(a_ptr + rm[:, None] * sam + kk[None, :] * sak,
                    mask=(rm < M)[:, None] & (kk < K)[None, :], other=0.0)
        b = tl.load(b_ptr + kk[:, None] * sbk + rn[None, :] * sbn,
                    mask=(kk < K)[:, None] & (rn < N)[None, :], other=0.0)
        acc += tl.dot(a, b, input_precision=PRECISION)
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc, mask=(rm < M)[:, None] & (rn < N)[None, :])


def run_triton(a, b, out, config, precision):
    bm, bn, bk, warps, stages = config
    m, k = a.shape
    n = b.shape[1]
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    gemm[grid](a, b, out, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1), out.stride(0),
               out.stride(1), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, PRECISION=precision, num_warps=warps,
               num_stages=stages)


def time_us(fn, repeats):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) * 1e3)
    return statistics.median(times)


def errors(got, ref, absprod):
    diff = (got.double() - ref).abs()
    scale = float(ref.abs().max())
    frob = float(ref.norm())
    return dict(max_rel=float(diff.max()) / scale if scale else 0.0,
                frob_rel=float(diff.norm()) / frob if frob else 0.0,
                cancel=float((diff / (absprod + 1e-30)).max()))


def inputs(m, k, n, gen, kind):
    a = torch.randn(m, k, generator=gen, device="cuda")
    b = torch.randn(k, n, generator=gen, device="cuda")
    if kind == "cancel":
        u = torch.rand(k // 2, generator=gen, device="cuda")
        a[:, 1::2] = -(a[:, 0::2] * (1 + 1e-3 * u))
        b[1::2, :] = b[0::2, :]
    return a.contiguous(), b.contiguous()


def error_set(a, b):
    """{mode: errors} for every mode on one input set (eager outputs, best-in-list not needed for errors:
    all configs of a mode use the same dot, so the first accepted config is used)."""
    ref = a.double() @ b.double()
    absprod = a.double().abs() @ b.double().abs()
    out = {}
    for mode in ("cublas_ieee", "cublas_tf32"):
        torch.backends.cuda.matmul.fp32_precision = "ieee" if mode == "cublas_ieee" else "tf32"
        out[mode] = errors(torch.mm(a, b), ref, absprod)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    c = torch.empty(a.shape[0], b.shape[1], device="cuda")
    for mode, prec in TRITON.items():
        per_config = []
        for config in CONFIGS:
            try:
                run_triton(a, b, c, config, prec)
                torch.cuda.synchronize()
            except Exception:
                continue
            per_config.append(errors(c, ref, absprod))
        if per_config:  # worst over configs (block K changes the summation order)
            out[mode] = {key: max(e[key] for e in per_config) for key in per_config[0]}
    del ref, absprod
    return out


def main():
    path = sys.argv[1]
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    summary = dict(tool="probe_bf16x6", created_utc=datetime.now(timezone.utc).isoformat(),
                   gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                   torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
                   python=sys.version.split()[0], NVIDIA_TF32_OVERRIDE=os.environ.get("NVIDIA_TF32_OVERRIDE"),
                   configs=[list(c) for c in CONFIGS], seed=SEED, shapes={}, refused={})
    for (m, k, n) in SHAPES:
        key = f"{m}x{k}x{n}"
        reps = 20 if m == 4096 else 50
        flop = 2 * m * k * n
        a, b = inputs(m, k, n, gen, "random")
        c = torch.empty(m, n, device="cuda")
        res = {}
        for mode in ("cublas_ieee", "cublas_tf32"):
            torch.backends.cuda.matmul.fp32_precision = "ieee" if mode == "cublas_ieee" else "tf32"
            res[mode] = dict(us=time_us(lambda: torch.mm(a, b, out=c), reps))
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        ab, bb = a.bfloat16(), b.bfloat16()
        cb = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        res["cublas_bf16_dense"] = dict(us=time_us(lambda: torch.mm(ab, bb, out=cb), reps))
        del ab, bb, cb
        for mode, prec in TRITON.items():
            best = None
            for config in CONFIGS:
                try:
                    run_triton(a, b, c, config, prec)
                    torch.cuda.synchronize()
                except Exception as exc:
                    summary["refused"].setdefault(mode, {}).setdefault(str(config), f"{type(exc).__name__}: {str(exc)[:300]}")
                    continue
                us = time_us(lambda cf=config: run_triton(a, b, c, cf, prec), reps)
                if best is None or us < best[0]:
                    best = (us, config)
            if best is not None:
                res[mode] = dict(us=best[0], config=list(best[1]))
        for r in res.values():
            r["tflops"] = flop / (r["us"] * 1e-6) / 1e12
        err = {"random": error_set(a, b)}
        del a, b, c
        a, b = inputs(m, k, n, gen, "cancel")
        err["cancel"] = error_set(a, b)
        del a, b
        torch.cuda.empty_cache()
        summary["shapes"][key] = dict(m=m, k=k, n=n, timing=res, errors=err)
        print(key, {k_: round(v["tflops"], 2) for k_, v in res.items()}, flush=True)
    peak = summary["shapes"]["4096x4096x4096"]["timing"]
    summary["ratio_bf16dense_over_6x_fp32_4096"] = peak["cublas_bf16_dense"]["tflops"] / (6 * peak["cublas_ieee"]["tflops"])
    wins = {}
    for key, s in summary["shapes"].items():
        if s["k"] == 512 and "triton_bf16x6" in s["timing"]:
            wins[key] = s["timing"]["triton_bf16x6"]["us"] < s["timing"]["cublas_ieee"]["us"]
    summary["bf16x6_beats_cublas_ieee_k512"] = wins
    summary["worst_errors"] = {
        kind: {mode: {metric: max(s["errors"][kind][mode][metric] for s in summary["shapes"].values()
                                  if mode in s["errors"][kind])
                      for metric in ("max_rel", "frob_rel", "cancel")}
               for mode in FP32_CLASS + ("cublas_tf32",)
               if any(mode in s["errors"][kind] for s in summary["shapes"].values())}
        for kind in ("random", "cancel")}
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print("ratio", summary["ratio_bf16dense_over_6x_fp32_4096"], "wins", wins)
    print(json.dumps(summary["worst_errors"], indent=1))
    print("refused", json.dumps(summary["refused"])[:2000])


if __name__ == "__main__":
    t0 = time.time()
    main()
    print("elapsed_s", round(time.time() - t0, 1))
