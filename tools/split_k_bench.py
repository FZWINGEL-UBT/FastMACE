"""Micro-benchmark of the split-K product (plan task OPT2-T05; docs/design/opt2-T05-split-k.md).

For each shape ``[m, k] @ [k, n]`` (float32, CUDA), the time of one cuBLAS product (``a @ b``) against
``fastmace.rewrites.linear.modules.split_k_mm`` with 2, 4 and 8 parts and with the shape rule's choice
(``tile_rule.split_k_parts``), each captured in a CUDA graph of ``--reps`` products and replayed (so launch
overhead counts as in the engine), median of ``--repeats`` replays. The shapes default to NaCl's l = 0
products and an l = 1 product the rule leaves alone. A screen, not the A/B: the gain in the step decides.

    python tools/split_k_bench.py --output results/<run>/bench.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

SHAPES = ((216, 512, 512), (216, 2048, 512), (216, 512, 2048), (648, 512, 512))  # (m, k, n)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reps", type=int, default=50, help="products per graph replay")
    p.add_argument("--repeats", type=int, default=30, help="timed replays per variant")
    p.add_argument("--output", type=Path, help="JSON file to write")
    return p.parse_args(argv)


def time_graph(function, reps, repeats):
    import torch

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            function()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            function()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / reps)  # µs per product
    return statistics.median(times)


def main(argv=None):
    import torch

    from fastmace.kernels.triton import tile_rule
    from fastmace.rewrites.linear.modules import split_k_mm

    args = parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    sms = tile_rule.device_sms("cuda")
    generator = torch.Generator(device="cuda").manual_seed(1)
    rows = []
    for m, k, n in SHAPES:
        a = torch.randn((m, k), device="cuda", generator=generator)
        b = torch.randn((k, n), device="cuda", generator=generator)
        rule = tile_rule.split_k_parts(m, n, k, sms)
        row = {"m": m, "k": k, "n": n, "rule_parts": rule, "cublas_us": time_graph(lambda: a @ b, args.reps,
                                                                                  args.repeats)}
        reference = (a.double() @ b.double())
        for parts in (2, 4, 8):
            if k % parts or k // parts < 1:
                continue
            row[f"split{parts}_us"] = time_graph(lambda p=parts: split_k_mm(a, b, p), args.reps, args.repeats)
            row[f"split{parts}_max_rel_err"] = float(((split_k_mm(a, b, parts).double() - reference).abs().max()
                                                      / reference.abs().max()))
        row["cublas_max_rel_err"] = float(((a @ b).double() - reference).abs().max() / reference.abs().max())
        rows.append(row)
        print(json.dumps(row))
    out = {"gpu": torch.cuda.get_device_name(), "sms": sms, "torch": torch.__version__, "reps": args.reps,
           "repeats": args.repeats, "rows": rows}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=1) + "\n")
    return out


if __name__ == "__main__":
    main()
