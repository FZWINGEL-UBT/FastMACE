"""Falsifier of the H200 ideas plan's idea 3 (piece-sorted radial-table evaluation): ``radial_table_forward`` on
edges in receiver order (unrelated (pair, piece) rows within a block) against the same edges sorted by (pair, piece),
plus the cost of the sort and of scattering the rows back to receiver order. Synthetic table of OMAT-0's shape
(9 pairs, 624 pieces, 6 terms, 66 columns: 8.9 MB) and 373,760 edges (Benchmark_loup); r uniform in [1.0, 6.0] Å in
random order, as within a receiver's ~75 edges. Keep the idea if (sorted kernel + sort + scatter) / base <= 0.7.

    python tools/bench/radial_sort_bench.py [--output FILE]
"""

import argparse
import json
import sys
from pathlib import Path

import fastmace  # noqa: F401  (soft file locks before torch)


def timed(fn, reps=50):
    import torch
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) * 1000)
    times.sort()
    return times[len(times) // 2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", type=int, default=373760)
    parser.add_argument("--pairs", type=int, default=9)
    parser.add_argument("--pieces", type=int, nargs="*", default=[624, 896])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    import torch

    from fastmace.kernels.triton.radial_table import radial_table_forward

    torch.manual_seed(0)
    rows = []
    for pieces in args.pieces:
        terms, width, r_lo, r_max = 6, 66, 0.5, 6.0
        h = (r_max - r_lo) / pieces
        coef = torch.randn(args.pairs, pieces, terms, width, device="cuda")
        starts = r_lo + h * torch.arange(pieces, device="cuda", dtype=torch.float32)
        scales = torch.full((pieces,), 2.0 / h, device="cuda")
        bins = torch.arange(pieces, device="cuda", dtype=torch.int32)
        bin_scale = 1.0 / h
        r = 1.0 + 5.0 * torch.rand(args.edges, device="cuda")
        pair = torch.randint(0, args.pairs, (args.edges,), device="cuda", dtype=torch.int32)
        call = (coef, starts, scales, bins, r_lo, bin_scale, r_max)

        base = timed(lambda: radial_table_forward(r, pair, *call))
        piece = ((r - r_lo) * bin_scale).to(torch.int32)
        key = pair * 4096 + piece
        perm = torch.sort(key).indices
        rs, ps = r[perm].contiguous(), pair[perm].contiguous()
        sorted_us = timed(lambda: radial_table_forward(rs, ps, *call))
        sort_us = timed(lambda: torch.sort(key))
        gather_us = timed(lambda: (r[perm], pair[perm]))
        g, _ = radial_table_forward(rs, ps, *call)
        out = torch.empty_like(g)
        scatter_us = timed(lambda: out.index_copy_(0, perm, g))
        ref, _ = radial_table_forward(r, pair, *call)
        out.index_copy_(0, perm, g)
        row = dict(pieces=pieces, table_mb=round(coef.numel() * 4 / 1e6, 2), base_us=round(base, 1),
                   sorted_us=round(sorted_us, 1), sort_us=round(sort_us, 1), gather_us=round(gather_us, 1),
                   scatter_us=round(scatter_us, 1), bitwise=bool(torch.equal(out, ref)),
                   kernel_ratio=round(sorted_us / base, 3),
                   total_ratio=round((sorted_us + sort_us + gather_us + scatter_us) / base, 3))
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = dict(gpu=torch.cuda.get_device_name(), edges=args.edges, rows=rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
