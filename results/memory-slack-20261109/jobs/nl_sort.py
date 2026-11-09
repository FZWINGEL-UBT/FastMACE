"""build_gpu with SORT_KEYS_ONLY on and off on one GPU: the same buffers (bitwise), each build's peak above the list's
own buffers, and its time. SrTiO3 by default (a = 3.905 A), cutoff 6 A (OMAT-0 medium), receivers sorted by species
group as the engine sorts them.

    python nl_sort.py --repeat 36x36x36 [--repeats 3] [--length-sorted]
"""
from __future__ import annotations

import argparse
import json
import time

import fastmace  # noqa: F401  (soft file locks before torch)

# isort: split
import numpy as np
import torch
from ase import Atoms

from fastmace.config import NeighbourConfig
from fastmace.neighbors import skin


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeat", default="36x36x36")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cutoff", type=float, default=6.0)
    p.add_argument("--length-sorted", action="store_true")
    p.add_argument("--output")
    args = p.parse_args()
    cell = Atoms("SrTiO3", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
                 cell=[3.905] * 3, pbc=True)
    atoms = cell.repeat(tuple(int(n) for n in args.repeat.split("x")))
    rng = np.random.default_rng(1)
    atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
    groups = np.unique(atoms.numbers, return_inverse=True)[1]
    skin.SORT_BY_LENGTH = bool(args.length_sorted)
    device = torch.device("cuda")
    positions = torch.tensor(atoms.positions, dtype=torch.float64, device=device)
    out = dict(gpu=torch.cuda.get_device_name(), natoms=len(atoms), repeat=args.repeat, torch=torch.__version__,
               length_sorted=args.length_sorted, runs={})
    lists = {}
    for keys_only in (False, True, False, True):  # each side twice, alternating
        skin.SORT_KEYS_ONLY = keys_only
        nl = skin.SkinNeighbourList(atoms.cell.array, args.cutoff, device, torch.float32, NeighbourConfig(),
                                    sort_receivers=True, gpu_build=True, sender_groups=groups)
        nl.build_gpu(positions)  # allocates the buffers and the screen's
        torch.cuda.synchronize()
        times, peaks = [], []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            start = time.perf_counter()
            nl.build_gpu(positions)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            peaks.append(torch.cuda.max_memory_allocated() - base)
        nl.compact(positions)
        torch.cuda.synchronize()
        key = "keys_only" if keys_only else "key_value"
        run = out["runs"].setdefault(key, dict(peak_above_buffers_MiB=[], build_ms=[]))
        run["peak_above_buffers_MiB"] += [round(x / 2**20, 1) for x in peaks]
        run["build_ms"] += [round(1e3 * x, 2) for x in times]
        run["candidates"] = nl.num_candidates
        run["candidates_per_atom"] = round(nl.num_candidates / len(atoms), 2)
        lists[key] = {name: t.clone() for name, t in nl.buffers().items() if torch.is_tensor(t)}
        del nl
        torch.cuda.empty_cache()
    a, b = lists["key_value"], lists["keys_only"]
    out["bitwise"] = {name: bool(torch.equal(a[name], b[name])) for name in a}
    print(json.dumps(out, indent=1))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
