"""Round 4 (5 November 2026): launch shapes of ``radial_table_forward`` on real edges.

Interaction 2's table forward feeds the fused forward's ``a`` (OMAT-0 medium, Benchmark_loup: 204 us on the A100,
J = 66 columns in a 128-wide register tile). This times the op on the real edge lengths and pair order of SrTiO3
10x10x10 with a random table of the production shape ([pairs, pieces, 6 terms, 66 columns]) for edge blocks, column
blocks and warp counts (module constants), each against the defaults in the same process; outputs compared bitwise.

Usage: python tools/radial_table_lab.py --output DIR
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)
from tabled_sum_lab import edges_of


def main(argv=None):
    import numpy as np
    import torch

    from fastmace.kernels.triton import radial_table as rt

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", default="results/loup-a100-20261103/SrTiO3.cif@10x10x10")
    p.add_argument("--cutoff", type=float, default=6.0)
    p.add_argument("--jitter", type=float, default=0.07)
    p.add_argument("--pieces", type=int, default=622)
    p.add_argument("--rounds", type=int, default=41)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    key, pair, dist, slots, pairs = edges_of(args.structure, args.cutoff, args.jitter)
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(3)
    r_lo, r_max, n_bins = 0.5, args.cutoff, 4 * args.pieces
    width = (r_max - r_lo) / args.pieces
    coef = torch.randn((pairs, args.pieces, 6, 66), generator=g).to(dev)
    starts = (r_lo + width * torch.arange(args.pieces, dtype=torch.float64)).float().to(dev)
    scales = torch.full((args.pieces,), 2.0 / width).to(dev)
    bins = (torch.arange(n_bins) * args.pieces // n_bins).to(torch.int32).to(dev)
    bin_scale = n_bins / (r_max - r_lo)
    r = torch.from_numpy(dist.astype(np.float32)).to(dev)
    pair_t = torch.from_numpy(pair.astype(np.int32)).to(dev)

    def call():
        return rt.radial_table_forward(r, pair_t, coef, starts, scales, bins, r_lo, bin_scale, r_max)

    def timed():
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        times = []
        for _ in range(args.rounds):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            call()
            stop.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(stop) * 1e3)
        return float(np.median(times))

    defaults = dict(BLOCK_E=rt.BLOCK_E, COLUMN_BLOCK=rt.COLUMN_BLOCK, WARPS=rt.WARPS)
    ref = call()
    base = timed()
    rows = [dict(variant="default", **defaults, us=base)]
    print(f"default {defaults} {base:.1f} us  ({len(r)} edges)", flush=True)
    for block_e in (8, 16, 32, 64):
        for column_block in (None, 64, 32, 16):
            for warps in (None, 1, 2, 8):
                rt.BLOCK_E, rt.COLUMN_BLOCK, rt.WARPS = block_e, column_block, warps
                try:
                    us = timed()
                    out = call()
                    same = all(torch.equal(a, b) for a, b in zip(out, ref))
                except Exception as error:  # a shape Triton refuses
                    print(f"e{block_e}-c{column_block}-w{warps}: {str(error)[:120]}", flush=True)
                    continue
                finally:
                    rt.BLOCK_E, rt.COLUMN_BLOCK, rt.WARPS = (defaults[k] for k in ("BLOCK_E", "COLUMN_BLOCK",
                                                                                     "WARPS"))
                rows.append(dict(variant=f"e{block_e}-c{column_block}-w{warps}", BLOCK_E=block_e,
                                 COLUMN_BLOCK=column_block, WARPS=warps, us=us, ratio=us / base, bitwise=same))
                print(f"e{block_e:<3d} c{str(column_block):5s} w{str(warps):5s} {us:8.1f} us  {us / base:.3f}  "
                      f"bitwise {same}", flush=True)
    meta = dict(tool="radial_table_lab", created_utc=datetime.now(timezone.utc).isoformat(),
                gpu=torch.cuda.get_device_name(), torch=torch.__version__, edges=len(r), pairs=pairs,
                pieces=args.pieces, structure=args.structure, rounds=args.rounds)
    (args.output / "radial_table_lab.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=1))


if __name__ == "__main__":
    main()
