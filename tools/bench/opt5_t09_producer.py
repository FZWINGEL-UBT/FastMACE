#!/usr/bin/env python3
"""OPT5-T09 producer screen: fused Triton A-split kernel + pre-split GEMM.

Measures (a) a single fused kernel that splits FP32 A into FP16 high/low
parts with per-row power-of-two scales, (b) a plain copy baseline for the
producer-emission marginal cost, and (c) the pre-split GEMM consuming the
Triton-produced parts. Experimental only; no production code is changed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import opt5_t09_screens as screens
import torch
import triton
import triton.language as tl

base = screens.base


@triton.jit
def split_a_kernel(A, Ah, Al, Asc, M, K, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pm = tl.program_id(0)
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    maxabs = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + rk
        tile = tl.load(A + rm[:, None] * K + k[None, :],
                       mask=(rm < M)[:, None] & (k < K)[None, :], other=0.0)
        maxabs = tl.maximum(maxabs, tl.max(tl.abs(tile), axis=1))
    scale = tl.exp2(tl.ceil(tl.log2(maxabs / 32767.0)))
    scale = tl.where(maxabs > 0, scale, 1.0)
    tl.store(Asc + rm, scale, mask=rm < M)
    inv = (1.0 / scale)[:, None]
    for k0 in range(0, K, BLOCK_K):
        k = k0 + rk
        mask = (rm < M)[:, None] & (k < K)[None, :]
        tile = tl.load(A + rm[:, None] * K + k[None, :], mask=mask, other=0.0)
        xn = tile * inv
        hi = xn.to(tl.float16)
        lo = ((xn - hi.to(tl.float32)) * 2048.0).to(tl.float16)
        tl.store(Ah + rm[:, None] * K + k[None, :], hi, mask=mask)
        tl.store(Al + rm[:, None] * K + k[None, :], lo, mask=mask)


def launch_split_a(a, ah, al, asc, bm=64, bk=128):
    m, k = a.shape
    split_a_kernel[(triton.cdiv(m, bm),)](a, ah, al, asc, m, k, BLOCK_M=bm, BLOCK_K=bk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", default="13608x512x512")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    m, k, n = (int(x) for x in args.shape.split("x"))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda")
    b = torch.randn(k, n, device="cuda")
    out = torch.empty(m, n, device="cuda")
    ref = a.double() @ b.double()
    norm = a.double().abs() @ b.double().abs()

    ah = torch.empty(m, k, device="cuda", dtype=torch.float16)
    al = torch.empty(m, k, device="cuda", dtype=torch.float16)
    asc = torch.empty(m, device="cuda")
    launch_split_a(a, ah, al, asc)
    torch.cuda.synchronize()
    bh, bl, bsc = screens.split_global(b, dim=0)
    parts = (ah, al, asc, bh, bl, bsc)
    cfg = (64, 64, 32, 4, 3)
    screens.launch_presplit(parts, out, cfg)
    torch.cuda.synchronize()

    result = {"metadata": {
        "utc": datetime.now(timezone.utc).isoformat(), "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()), "torch": torch.__version__,
        "triton": triton.__version__, "shape_mkn": [m, k, n],
        "commit": subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[2]),
                                           "rev-parse", "HEAD"], text=True).strip()},
        "cublas_fp32_us": base.graph_time_us(lambda: torch.mm(a, b, out=out)),
        "fused_triton_split_a_us": base.graph_time_us(lambda: launch_split_a(a, ah, al, asc)),
        "torch_unfused_split_a_us": screens.event_time_us(lambda: screens.split_global(a, dim=1)),
        "copy_a_us": base.graph_time_us(lambda: a.clone()),
        "presplit_gemm_us": base.graph_time_us(lambda: screens.launch_presplit(parts, out, cfg)),
        "presplit_gemm_errors": base.calc_errors(out, ref, norm),
    }
    result["split_plus_gemm_us"] = result["fused_triton_split_a_us"] + result["presplit_gemm_us"]
    print(json.dumps(result, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
