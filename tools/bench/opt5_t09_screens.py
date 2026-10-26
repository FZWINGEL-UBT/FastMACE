#!/usr/bin/env python3
"""OPT5-T09 screens: concurrent split, FP32 baseline options, tensor-only
FP16x3 configurations, and producer-side A splitting / epilogue fusion.

Experimental microbenchmarks only. No production dispatch is changed.
The representative epilogue is bias + sigmoid gate + residual; it estimates
fusion value and is not a MACE layer implementation.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import opt5_t08_hybrid_gemm as base
import torch
import triton
import triton.language as tl


@triton.jit
def presplit_kernel(Ah, Al, Bh, Bl, C, Asc, Bsc, M, N, K,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = (pm * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn = (pn * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    rk = tl.arange(0, BLOCK_K).to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + rk
        kmask = (k < K)[None, :]
        a_h = tl.load(Ah + rm[:, None] * K + k[None, :], mask=(rm < M)[:, None] & kmask, other=0.0)
        a_l = tl.load(Al + rm[:, None] * K + k[None, :], mask=(rm < M)[:, None] & kmask, other=0.0)
        b_h = tl.load(Bh + k[:, None] * N + rn[None, :], mask=(k < K)[:, None] & (rn < N)[None, :], other=0.0)
        b_l = tl.load(Bl + k[:, None] * N + rn[None, :], mask=(k < K)[:, None] & (rn < N)[None, :], other=0.0)
        acc += tl.dot(a_h, b_h, out_dtype=tl.float32)
        acc += (tl.dot(a_h, b_l, out_dtype=tl.float32) + tl.dot(a_l, b_h, out_dtype=tl.float32)) * (1.0 / 2048.0)
    asc = tl.load(Asc + rm, mask=rm < M, other=1.0)
    bsc = tl.load(Bsc + rn, mask=rn < N, other=1.0)
    out = acc * asc[:, None] * bsc[None, :]
    tl.store(C + rm[:, None] * N + rn[None, :], out, mask=(rm < M)[:, None] & (rn < N)[None, :])


@triton.jit
def presplit_epilogue_kernel(Ah, Al, Bh, Bl, C, Asc, Bsc, Bias, Gate, Res, M, N, K,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = (pm * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn = (pn * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    rk = tl.arange(0, BLOCK_K).to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + rk
        kmask = (k < K)[None, :]
        a_h = tl.load(Ah + rm[:, None] * K + k[None, :], mask=(rm < M)[:, None] & kmask, other=0.0)
        a_l = tl.load(Al + rm[:, None] * K + k[None, :], mask=(rm < M)[:, None] & kmask, other=0.0)
        b_h = tl.load(Bh + k[:, None] * N + rn[None, :], mask=(k < K)[:, None] & (rn < N)[None, :], other=0.0)
        b_l = tl.load(Bl + k[:, None] * N + rn[None, :], mask=(k < K)[:, None] & (rn < N)[None, :], other=0.0)
        acc += tl.dot(a_h, b_h, out_dtype=tl.float32)
        acc += (tl.dot(a_h, b_l, out_dtype=tl.float32) + tl.dot(a_l, b_h, out_dtype=tl.float32)) * (1.0 / 2048.0)
    asc = tl.load(Asc + rm, mask=rm < M, other=1.0)
    bsc = tl.load(Bsc + rn, mask=rn < N, other=1.0)
    out = acc * asc[:, None] * bsc[None, :]
    bias = tl.load(Bias + rn, mask=rn < N, other=0.0)
    gate = tl.load(Gate + rn, mask=rn < N, other=0.0)
    res = tl.load(Res + rm[:, None] * N + rn[None, :], mask=(rm < M)[:, None] & (rn < N)[None, :], other=0.0)
    out = (out + bias[None, :]) * tl.sigmoid(gate[None, :]) + res
    tl.store(C + rm[:, None] * N + rn[None, :], out, mask=(rm < M)[:, None] & (rn < N)[None, :])


def launch_presplit(parts, out, cfg, epilogue=None):
    ah, al, asc, bh, bl, bsc = parts
    m, k = ah.shape
    n = bh.shape[1]
    bm, bn, bk, nw, ns = cfg
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    if epilogue is None:
        presplit_kernel[grid](ah, al, bh, bl, out, asc, bsc, m, n, k,
                              BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw, num_stages=ns)
    else:
        bias, gate, res = epilogue
        presplit_epilogue_kernel[grid](ah, al, bh, bl, out, asc, bsc, bias, gate, res, m, n, k,
                                       BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw, num_stages=ns)


def split_global(x, dim):
    maxabs = x.abs().amax(dim=dim, keepdim=True)
    exponent = torch.ceil(torch.log2((maxabs / 32767.0).clamp_min(1e-30)))
    scale = torch.exp2(exponent).where(maxabs > 0, torch.ones_like(maxabs))
    xn = x / scale
    hi = xn.half()
    lo = ((xn - hi.float()) * 2048.0).half()
    return hi.contiguous(), lo.contiguous(), scale.squeeze(dim).contiguous()


def event_time_us(fn, warmup=5, reps=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    return round(statistics.median(times), 2)


def concurrent_time_us(fn_a, fn_b, reps=30):
    s_a, s_b = torch.cuda.Stream(), torch.cuda.Stream()
    for _ in range(5):
        fn_a()
        fn_b()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        with torch.cuda.stream(s_a):
            fn_a()
        with torch.cuda.stream(s_b):
            fn_b()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    return round(statistics.median(times), 2)


def sequential_time_us(fn_a, fn_b, reps=30):
    for _ in range(5):
        fn_a()
        fn_b()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn_a()
        fn_b()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    return round(statistics.median(times), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", default="13608x512x512")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    m, k, n = (int(x) for x in args.shape.split("x"))
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda")
    b = torch.randn(k, n, device="cuda")
    out = torch.empty(m, n, device="cuda")
    ref = a.double() @ b.double()
    norm = a.double().abs() @ b.double().abs()
    result = {"metadata": {
        "utc": datetime.now(timezone.utc).isoformat(), "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()), "torch": torch.__version__,
        "triton": triton.__version__, "shape_mkn": [m, k, n],
        "commit": subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[2]),
                                           "rev-parse", "HEAD"], text=True).strip(),
        "precision": "FP32 IEEE for cuBLAS; normalized FP16x3 for Triton tensor paths",
    }}

    # 1. Two-kernel concurrent column split.
    full_cublas = base.graph_time_us(lambda: torch.mm(a, b, out=out))
    splits = []
    for ns in (256, 384, 448):
        nt = n - ns
        out_s = torch.empty(m, ns, device="cuda")
        out_t = torch.empty(m, nt, device="cuda")
        def fn_s(ns=ns, out_s=out_s):
            return torch.mm(a, b[:, :ns], out=out_s)

        def fn_t(ns=ns, out_t=out_t):
            return base.launch_triton_fp16x3_rowcol(a, b[:, ns:], out_t)

        fn_s()
        fn_t()
        torch.cuda.synchronize()
        combined = torch.cat([out_s, out_t], dim=1)
        splits.append({"simt_cols": ns, "mma_cols": nt,
                       "simt_alone_us": base.graph_time_us(fn_s),
                       "mma_alone_us": base.graph_time_us(fn_t),
                       "sequential_us": sequential_time_us(fn_s, fn_t),
                       "concurrent_us": concurrent_time_us(fn_s, fn_t),
                       "errors": base.calc_errors(combined, ref, norm)})
        print("split", splits[-1], flush=True)
    result["two_kernel_split"] = {"full_cublas_us": full_cublas, "splits": splits}

    # 3. FP32 baseline options (CUTLASS Python is checked separately; not installed here).
    fp32 = {"torch_mm_us": full_cublas}
    bt = b.t().contiguous()
    fp32["torch_mm_bt_view_us"] = base.graph_time_us(lambda: torch.mm(a, bt.t(), out=out))
    try:
        torch.backends.cuda.preferred_blas_library("cublaslt")
        fp32["torch_mm_cublaslt_us"] = base.graph_time_us(lambda: torch.mm(a, b, out=out))
    except Exception as exc:  # recorded, not hidden
        fp32["cublaslt_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            torch.backends.cuda.preferred_blas_library("cublas")
        except Exception:
            pass
    result["fp32_baseline"] = fp32
    print("fp32", fp32, flush=True)

    # 4. Tensor-only FP16x3 configuration sweep.
    configs = [(64, 64, 32, 4, 3), (128, 64, 32, 4, 3), (64, 128, 32, 4, 3),
               (128, 128, 32, 8, 3), (128, 64, 64, 4, 2), (256, 128, 32, 8, 3)]
    tensor = []
    for cfg in configs:
        def fn(cfg=cfg):
            return base.launch_triton_fp16x3_rowcol(a, b, out, *cfg)

        try:
            fn()
            torch.cuda.synchronize()
            tensor.append({"kind": "rowcol_on_the_fly", "cfg_bm_bn_bk_warps_stages": cfg,
                           "us": base.graph_time_us(fn), "errors": base.calc_errors(out, ref, norm)})
        except Exception as exc:
            tensor.append({"kind": "rowcol_on_the_fly", "cfg_bm_bn_bk_warps_stages": cfg,
                           "error": f"{type(exc).__name__}: {exc}"})
        print("tensor", tensor[-1], flush=True)
    result["tensor_only"] = tensor

    # 5. Producer-side splitting and epilogue fusion.
    ah, al, asc = split_global(a, dim=1)
    bh, bl, bsc = split_global(b, dim=0)
    parts = (ah, al, asc, bh, bl, bsc)
    presplit = []
    for cfg in [(64, 64, 32, 4, 3), (128, 128, 32, 8, 3), (128, 128, 64, 8, 2)]:
        def fn(cfg=cfg):
            return launch_presplit(parts, out, cfg)

        fn()
        torch.cuda.synchronize()
        presplit.append({"cfg_bm_bn_bk_warps_stages": cfg, "kernel_us": base.graph_time_us(fn),
                         "errors": base.calc_errors(out, ref, norm)})
        print("presplit", presplit[-1], flush=True)
    a_prep_us = event_time_us(lambda: split_global(a, dim=1))
    best_presplit = min(presplit, key=lambda x: x["kernel_us"])
    bias = torch.randn(n, device="cuda")
    gate = torch.randn(n, device="cuda")
    residual = torch.randn(m, n, device="cuda")
    y = torch.empty(m, n, device="cuda")
    cfg = best_presplit["cfg_bm_bn_bk_warps_stages"]
    fused_us = base.graph_time_us(lambda: launch_presplit(parts, y, cfg, epilogue=(bias, gate, residual)))

    def separate_epilogue():
        launch_presplit(parts, out, cfg)
        return (out + bias) * torch.sigmoid(gate) + residual

    separate_total_us = event_time_us(separate_epilogue)
    result["producer_and_epilogue"] = {
        "a_presplit_recurring_us": a_prep_us,
        "presplit_configs": presplit,
        "best_presplit_kernel_plus_a_prep_us": best_presplit["kernel_us"] + a_prep_us,
        "representative_epilogue": "bias + sigmoid(gate) + residual",
        "gemm_plus_separate_epilogue_us": separate_total_us,
        "gemm_with_fused_epilogue_us": fused_us,
    }
    print("producer", result["producer_and_epilogue"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
