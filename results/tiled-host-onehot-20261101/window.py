"""The gx window of host-state aliasing (the neighbour-list session's proposal), measured from the real tile plan.

Sweep 2 visits the tiles in plan order. Tile t touches its receivers and its halo (the senders of its edges, from
the per-tile neighbour lists, as the tiled call builds them). An atom's dE/dx is open from its first touching tile
to its last one; the window is the number of atoms open after tile t, and its maximum over t is the buffer gx
would need. Records the maximum, as a fraction of N, and the tiles' receiver and halo sizes. No model.

Run as ``run.sh NAME results/tiled-host-onehot-20261101/window.py --repeat 35x35x105``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "bench"))
import tiled_capacity  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--structure", default="NbCl5")
    p.add_argument("--repeat", required=True)
    p.add_argument("--receivers", type=int, default=4096)
    p.add_argument("--cutoff", type=float, default=6.0)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    import torch

    from fastmace.model.tiled import plan_tiles
    from fastmace.neighbors.tile_list import TileNeighbours

    repeat = tuple(int(x) for x in args.repeat.lower().split("x"))
    atoms = tiled_capacity.supercell(args.structure, repeat, 0.0, 0.0)
    n = len(atoms)
    device = torch.device("cuda")
    positions = torch.tensor(atoms.positions, dtype=torch.float64, device=device)
    cell = torch.tensor(atoms.cell.array, dtype=torch.float64, device=device)
    tiles, grid = plan_tiles(positions, cell, torch.zeros(n + 1, dtype=torch.int64, device=device), args.receivers,
                             2**62)
    bins = TileNeighbours(args.cutoff, atoms.cell.array, torch.float32, device=device)
    bins.bin(positions)
    first = torch.full((n,), len(tiles), dtype=torch.int64, device=device)
    last = torch.full((n,), -1, dtype=torch.int64, device=device)
    halos = []
    for t, receivers in enumerate(tiles):
        senders = bins.edges(receivers)[0]
        nodes = torch.unique(torch.cat([receivers, senders]))
        halos.append(int(nodes.numel() - receivers.numel()))
        first[nodes] = torch.minimum(first[nodes], torch.tensor(t, device=device))
        last[nodes] = t
    opened = torch.bincount(first, minlength=len(tiles) + 1)[:len(tiles)]
    closed = torch.bincount(last, minlength=len(tiles))
    window = torch.cumsum(opened - closed, 0)  # open after tile t: first <= t < last
    peak = int(window.max())
    record = dict(structure=args.structure, atoms=n, receivers=args.receivers, cutoff=args.cutoff,
                  tiles=len(tiles), grid=grid, host=os.uname().nodename,
                  source_commit=os.environ.get("FASTMACE_SOURCE_COMMIT"), max_window=peak,
                  max_window_fraction=peak / n, mean_halo=sum(halos) / len(halos), max_halo=max(halos),
                  window_quartiles=[int(torch.quantile(window.double(), q)) for q in (0.25, 0.5, 0.75)])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"window-{args.structure}-{n}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record))


if __name__ == "__main__":
    main()
