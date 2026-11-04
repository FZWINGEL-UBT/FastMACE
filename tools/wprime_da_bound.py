"""The a'-path bound of the wprime backward (4 November 2026, Benchmark_loup speed goal; owner: "Try it").

The wprime backward's slope is ``dr = sum_c dw_c w'_c(r)`` with ``w' = a'(r) [W, b]^T`` read per edge from the
derivative table ``G'`` (a [terms, Q] slab per edge). The alternative ``dr = a'(r) . (dw W)`` needs the per-edge
product ``da = dw W`` ([E, Q] x [Q, J]) instead. This times that product alone as one large GEMM, the most efficient
form it can take (inside the streaming kernel it can only be slower), against the slope's measured cost (A100-SXM4:
wprime 4.70 ms, without slope 2.89 ms; ``results/omat0-kernel-lab-20261103/a100-sxm4-wprime-deg5``).

Products of Benchmark_loup's shape (373,760 edges, Q = 1,280, J = 64 and 72): CUTLASS 3xTF32 (``cutlass.tf32x3``,
the A100's product), cuBLAS FP32 and cuBLAS TF32 (one pass, for scale only); CUDA events, median of ``--rounds``.

Usage: python tools/wprime_da_bound.py --output DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

# isort: split
import torch

EDGES, Q = 373760, 1280


def timed(fn, rounds):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(rounds):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return sorted(times)[len(times) // 2]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=20)
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    from fastmace.kernels.cutlass import tf32x3
    g = torch.Generator(device="cuda").manual_seed(0)
    dw = torch.randn(EDGES, Q, device="cuda", generator=g)
    result = dict(gpu=torch.cuda.get_device_name(), edges=EDGES, q=Q, products={})
    for j in (64, 72):
        w = torch.randn(Q, j, device="cuda", generator=g) / Q ** 0.5
        out = torch.empty(EDGES, j, device="cuda")
        row = {}
        if tf32x3.available(dw.device) and tf32x3.eligible(dw, w, out):
            row["cutlass_tf32x3_us"] = timed(lambda: tf32x3.mm_into(dw, w, out), args.rounds)
            exact = dw.double() @ w.double()
            row["cutlass_tf32x3_max_rel"] = float((out.double() - exact).abs().max() / exact.abs().max())
        torch.backends.cuda.matmul.allow_tf32 = False
        row["cublas_fp32_us"] = timed(lambda: torch.mm(dw, w, out=out), args.rounds)
        torch.backends.cuda.matmul.allow_tf32 = True
        row["cublas_tf32_us"] = timed(lambda: torch.mm(dw, w, out=out), args.rounds)
        torch.backends.cuda.matmul.allow_tf32 = False
        row["gflop"] = 2.0 * EDGES * Q * j / 1e9
        result["products"][j] = row
        print(f"J {j}: " + "  ".join(f"{k} {v:.4g}" for k, v in row.items()), flush=True)
    (args.output / "wprime_da_bound.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
