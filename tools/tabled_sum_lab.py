"""Round 2 follow-up (5 November 2026): launch shapes of ``tabled_outer_sum`` (FUSED_FIRST_BACKWARD, TABLED_FORWARD).

The A100 profile at ``2e9a3878`` (Benchmark_loup, OMAT-0 medium) has the tabled forward at 391 us (one warp per
32-edge program, 128 registers, 25 % occupancy) and the fused backward at 459 us (one warp per 16-edge program). This
times both custom ops on the real key order and edge lengths (SrTiO3 10x10x10, a jittered lattice, edges sorted by
receiver and then sender species) with a random table of the production shape ([pairs, pieces, 6 terms, 66 columns],
64 table columns used, D = 16) for chunk sizes and warp counts set through the module constants, each against the
module's defaults in the same process. Accuracy: every variant against the defaults' outputs (max |d| / max |ref|).

Usage: python tools/tabled_sum_lab.py --output DIR
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)


def edges_of(structure, cutoff, jitter, seed=0):
    import numpy as np
    from ase.io import read
    from ase.neighborlist import neighbor_list

    path, _, repeat = structure.partition("@")
    atoms = read(path)
    if repeat:
        atoms = atoms.repeat(tuple(int(v) for v in repeat.split("x")))
    rng = np.random.default_rng(seed)
    atoms.positions += rng.normal(0.0, jitter, atoms.positions.shape)
    i, j, d = neighbor_list("ijd", atoms, cutoff)
    species = np.unique(atoms.numbers, return_inverse=True)[1]
    slots = int(species.max()) + 1
    order = np.lexsort((species[i], j))  # receiver, then sender species
    i, j, d = i[order], j[order], d[order]
    key = j * slots + species[i]
    pair = species[j] * slots + species[i]
    return key, pair, d, len(atoms) * slots, slots * slots


def main(argv=None):
    import numpy as np
    import torch

    from fastmace.kernels.triton import tabled_outer_sum as tos

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", default="results/loup-a100-20261103/SrTiO3.cif@10x10x10")
    p.add_argument("--cutoff", type=float, default=6.0)
    p.add_argument("--jitter", type=float, default=0.07)
    p.add_argument("--pieces", type=int, default=896)
    p.add_argument("--rounds", type=int, default=31)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    key, pair, dist, slots, pairs = edges_of(args.structure, args.cutoff, args.jitter)
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(3)
    r_lo, r_max, n_bins = 0.5, args.cutoff, 4 * args.pieces
    width = (r_max - r_lo) / args.pieces
    coef = torch.randn((pairs, args.pieces, 6, 66), generator=g).to(dev)
    dcoef = torch.randn((pairs, args.pieces, 5, 66), generator=g).to(dev)
    starts = (r_lo + width * torch.arange(args.pieces, dtype=torch.float64)).float().to(dev)
    scales = torch.full((args.pieces,), 2.0 / width).to(dev)
    bins = (torch.arange(n_bins) * args.pieces // n_bins).to(torch.int32).to(dev)
    bin_scale = n_bins / (r_max - r_lo)
    edges, d, k = len(key), 16, 64
    r = torch.from_numpy(dist.astype(np.float32)).reshape(-1, 1).to(dev)
    pair_t = torch.from_numpy(pair.astype(np.int32)).to(dev)
    key_t = torch.from_numpy(key.astype(np.int64)).to(dev)
    y = torch.randn((edges, d), generator=g).to(dev)
    grad = torch.randn((slots, d * k), generator=g).to(dev)
    grad_density = torch.randn((edges, 1), generator=g).to(dev)
    table = (r, pair_t, coef, starts, scales, bins, r_lo, bin_scale, r_max)

    def forward():
        return tos.tabled_outer_sum_forward(*table, k, y, key_t, slots)

    def backward():  # TABLED_FORWARD's backward: a evaluated again
        return tos.tabled_outer_sum_backward(grad, y.new_empty((0, 0)), y, key_t, r, pair_t, dcoef, starts, scales,
                                             bins, grad_density, r_lo, bin_scale, r_max, 1, coef, k)

    def timed(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(args.rounds):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            stop.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(stop) * 1e3)
        return float(np.median(times))

    def err(new, ref):
        return max(float((a - b).abs().max() / b.abs().max().clamp_min(1e-30)) for a, b in zip(new, ref))

    defaults = {"FORWARD_CHUNK": tos.FORWARD_CHUNK, "FORWARD_WARPS": tos.FORWARD_WARPS, "CHUNK": tos.CHUNK,
                "WARPS": tos.WARPS}
    ref_f, ref_b = forward(), backward()
    base_f, base_b = timed(forward), timed(backward)
    rows = [{"variant": "default", **defaults, "forward_us": base_f, "backward_us": base_b}]
    print(f"default {defaults} forward {base_f:.1f} us backward {base_b:.1f} us", flush=True)
    for chunk in (8, 16, 32, 64):
        for warps in (1, 2, 4):
            tos.FORWARD_CHUNK, tos.FORWARD_WARPS = chunk, warps
            f = timed(forward)
            ef = err(forward(), ref_f)
            tos.FORWARD_CHUNK, tos.FORWARD_WARPS = defaults["FORWARD_CHUNK"], defaults["FORWARD_WARPS"]
            tos.CHUNK, tos.WARPS = chunk, warps
            b = timed(backward)
            eb = err(backward(), ref_b)
            tos.CHUNK, tos.WARPS = defaults["CHUNK"], defaults["WARPS"]
            rows.append({"variant": f"c{chunk}-w{warps}", "chunk": chunk, "warps": warps, "forward_us": f,
                         "forward_ratio": f / base_f, "forward_err": ef, "backward_us": b,
                         "backward_ratio": b / base_b, "backward_err": eb})
            print(f"c{chunk}-w{warps}  forward {f:7.1f} us ({f / base_f:.3f}, err {ef:.1e})  "
                  f"backward {b:7.1f} us ({b / base_b:.3f}, err {eb:.1e})", flush=True)
    meta = {"tool": "tabled_sum_lab", "created_utc": datetime.now(timezone.utc).isoformat(),
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "edges": edges, "slots": slots,
            "pairs": pairs, "pieces": args.pieces, "structure": args.structure, "rounds": args.rounds}
    (args.output / "tabled_sum_lab.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
