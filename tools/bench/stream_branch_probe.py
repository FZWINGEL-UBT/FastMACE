#!/usr/bin/env python3
"""Two-stream independent-branch probe (megakernel Architecture A falsifier).

Times two genuinely different workloads -- an atom-style FP32 GEMM (linear_up
shapes) and an edge-style radial elementwise/MLP workload -- sequentially on one
stream versus concurrently on two streams, in CUDA-graph replays. If running
them concurrently does not beat running them back-to-back, SM-level scheduling
of independent branches has no headroom to recover on this GPU.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

# isort: split
import torch  # noqa: E402


def graph_time_us(fn, warmup=10, reps=50, blocks=7):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(blocks):
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6 / reps)
    return round(statistics.median(times), 2)


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    out = {"gpu": torch.cuda.get_device_name(), "cases": {}}
    # (label, GEMM M, edge rows E)
    cases = [("NaCl_like", 1512, 8192), ("NbCl5_like", 13608, 56320), ("NaNbO3_like", 9720, 82688)]
    s2 = torch.cuda.Stream()
    for label, m, e in cases:
        a = torch.randn(m, 512, device="cuda")
        w = torch.randn(512, 512, device="cuda")
        gemm_out = torch.empty(m, 512, device="cuda")
        h = torch.randn(e, 64, device="cuda")
        w1 = torch.randn(64, 64, device="cuda")
        w2 = torch.randn(64, 64, device="cuda")
        edge_out = torch.empty(e, 64, device="cuda")

        def gemm():
            torch.mm(a, w, out=gemm_out)

        def edge():
            x = torch.nn.functional.silu(torch.nn.functional.layer_norm(h, (64,)))
            x = x @ w1
            x = torch.nn.functional.silu(torch.nn.functional.layer_norm(x, (64,)))
            edge_out.copy_(x @ w2)

        def sequential():
            gemm()
            edge()

        def concurrent():
            s0 = torch.cuda.current_stream()
            s2.wait_stream(s0)
            gemm()
            with torch.cuda.stream(s2):
                edge()
            s0.wait_stream(s2)

        t_seq = graph_time_us(sequential)
        t_con = graph_time_us(concurrent)
        t_gemm = graph_time_us(gemm)
        t_edge = graph_time_us(edge)
        out["cases"][label] = dict(M=m, E=e, sequential_us=t_seq, concurrent_us=t_con,
                                   gemm_only_us=t_gemm, edge_only_us=t_edge,
                                   concurrent_over_sequential=round(t_con / t_seq, 3))
        print(label, out["cases"][label], flush=True)
    Path("/tmp/stream_branch_probe.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
