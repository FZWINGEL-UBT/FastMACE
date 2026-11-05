"""Idea 14's other half (ideas session, third revision, 6 November 2026): the dense product that would compute w once
per mirror pair, ``w = a @ W.T`` for one edge of every pair, ``[pairs, K] @ [K, Q]`` with its output written to HBM.
Timed under CUDA graphs with the CUTLASS 3xTF32 kernel (``tf32x3.mm_into``, the forward's precision) and, for
reference, torch's FP32 matmul with TF32 off. Pair the result with the kernel lab's ``F-tp-stored-w`` (the forward
reading w per edge instead of computing it): build the pair-shared forward only if the two together stay well below
the plan's forward.

    python tools/bench/pair_w_bench.py [--pairs 186880] [--k 64] [--q 1280] [--output FILE]
"""

import argparse
import json
import sys
from pathlib import Path

import fastmace  # noqa: F401  (soft file locks before torch)


def graph_us(fn, reps=20, rounds=7):
    import torch
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    times = []
    for _ in range(rounds):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            graph.replay()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) * 1000 / reps)
    return sorted(times)[len(times) // 2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=186880)  # 373,760 edges of Benchmark_loup / 2
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--q", type=int, default=1280)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    import torch

    from fastmace.kernels.cutlass import tf32x3

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    a = torch.randn(args.pairs, args.k, device="cuda")
    weight = torch.randn(args.q, args.k, device="cuda")
    out = torch.empty(args.pairs, args.q, device="cuda")
    copy = torch.empty_like(out)
    result = {"gpu": torch.cuda.get_device_name(), "pairs": args.pairs, "k": args.k, "q": args.q,
              "w_bytes": out.numel() * 4}
    if tf32x3.takes(args.pairs, args.k, args.q):
        b = weight.t()  # [K, Q], column-major view
        try:
            result["tf32x3_us"] = graph_us(lambda: tf32x3.mm_into(a, b, out))
            reference = a.double() @ weight.double().t()
            result["tf32x3_rel_err"] = float(((out.double() - reference).abs().max() / reference.abs().max()))
        except RuntimeError as error:  # no CUTLASS cubin for this GPU
            result["tf32x3_error"] = str(error)
    result["fp32_us"] = graph_us(lambda: torch.matmul(a, weight.t(), out=out))
    result["copy_w_us"] = graph_us(lambda: copy.copy_(out))  # read + write of w: HBM bandwidth at this size
    result["hbm_gbs"] = round(2 * out.numel() * 4 / result["copy_w_us"] / 1e3, 1)
    print(json.dumps(result), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
