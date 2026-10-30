"""Same-process A/B of ``SkinNeighbourList.compact``: main's against this tree's chunked one.

``--baseline DIR`` is a FastMACE source tree of main before the change; its ``fastmace/neighbors/skin.py`` is loaded
under another module name and its ``compact`` runs on this tree's list (the method reads only the list's buffers).
For each ``--repeat`` supercell of ``--structure`` (rattled by 0.05 Å, built with the GPU list at cutoff 6 Å and
skin 0.3 Å, sorted with sender groups as MH-1's plan does), the two compactions alternate ABBA for ``--rounds``
rounds (CUDA events, medians), at positions moved by up to 0.1 Å. Records the time of each, their peak above the
memory allocated before (one call each, after the rounds) and whether the edge index, unit shifts, shifts and
count are bitwise equal. Writes ``<output>/compact-ab.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiled_capacity  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--structure", default="NbCl5")
    p.add_argument("--repeat", nargs="+", default=["4x4x8", "6x6x20"])
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)
    import numpy as np
    import torch

    from fastmace.config import NeighbourConfig
    from fastmace.neighbors.skin import SkinNeighbourList

    spec = importlib.util.spec_from_file_location("_baseline_skin", args.baseline / "fastmace/neighbors/skin.py",
                                                  submodule_search_locations=None)
    old = importlib.util.module_from_spec(spec)
    old.__package__ = "fastmace.neighbors"  # its relative imports resolve to this tree's unchanged modules
    spec.loader.exec_module(old)
    record = dict(baseline=str(args.baseline), structure=args.structure, device=torch.cuda.get_device_name(),
                  cells=[])
    for text in args.repeat:
        repeat = tuple(int(x) for x in text.split("x"))
        atoms = tiled_capacity.supercell(args.structure, repeat, 0.05, 0.0)
        groups = torch.as_tensor(atoms.numbers % 7, dtype=torch.int64)
        nl = SkinNeighbourList(atoms.cell.array, 6.0, "cuda", torch.float32,
                               NeighbourConfig(skin=0.3, sort_receivers="on", gpu_neighbour_list="on"),
                               sender_groups=groups.cuda())
        x0 = torch.tensor(atoms.get_positions(), dtype=torch.float64, device="cuda")
        nl.build(x0)
        moved = x0 + torch.tensor(np.random.default_rng(3).uniform(-0.1, 0.1, atoms.positions.shape),
                                  dtype=torch.float64, device="cuda")
        sides = {"old": lambda nl=nl: old.SkinNeighbourList.compact(nl, moved), "new": lambda nl=nl: nl.compact(moved)}
        results = {}
        for name, run in sides.items():
            run()
            results[name] = (nl.edge_index.clone(), nl.unit_shifts.clone(), nl.shifts.clone(), int(nl.count))
        same = all(torch.equal(a, b) for a, b in zip(results["old"][:3], results["new"][:3]))
        same = same and results["old"][3] == results["new"][3]
        del results
        times = {"old": [], "new": []}
        for _ in range(args.rounds):
            for name in ("old", "new", "new", "old"):
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                sides[name]()
                stop.record()
                torch.cuda.synchronize()
                times[name].append(start.elapsed_time(stop))
        peaks = {}
        for name, run in sides.items():
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            run()
            torch.cuda.synchronize()
            peaks[name] = int(torch.cuda.max_memory_allocated() - before)
        entry = dict(repeat=text, atoms=len(atoms), candidates=int(nl.num_candidates),
                     capacity=int(nl.candidate_index.shape[1]), edges=int(nl.count), bitwise_equal=bool(same),
                     old_ms=statistics.median(times["old"]), new_ms=statistics.median(times["new"]),
                     old_peak_bytes=peaks["old"], new_peak_bytes=peaks["new"])
        entry["new_over_old"] = entry["new_ms"] / entry["old_ms"]
        record["cells"].append(entry)
        print(json.dumps(entry))
        del nl
        torch.cuda.empty_cache()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "compact-ab.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
