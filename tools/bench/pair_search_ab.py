"""Same-process A/B of the GPU pair screen: main's ``pair_search`` against this tree's (pair-search grid fix).

``--baseline DIR`` is a FastMACE source tree of main before the fix; its ``fastmace/neighbors/pair_search.py`` is
loaded under another module name, so both kernels run in this process. For each ``--repeat`` NbCl5 supercell
(wrapped positions, the 27 nearest image vectors, cutoff + skin), the two ``screen_codes`` alternate ABAB for
``--rounds`` rounds after a warm-up (CUDA events, median per side), and their codes are compared as sorted sets.
A full ``SkinNeighbourList`` rebuild with this tree is timed too. Writes ``<output>/pair-search-ab.json``.
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
    p.add_argument("--repeat", nargs="+", default=["4x4x8", "6x6x20"])
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--radius", type=float, default=6.3)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)
    import numpy as np
    import torch

    from fastmace.config import NeighbourConfig
    from fastmace.neighbors import pair_search as new
    from fastmace.neighbors.skin import SkinNeighbourList

    spec = importlib.util.spec_from_file_location("_baseline_pair_search",
                                                  args.baseline / "fastmace/neighbors/pair_search.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    record = dict(baseline=str(args.baseline), device=torch.cuda.get_device_name(), cells=[])
    for text in args.repeat:
        repeat = tuple(int(x) for x in text.split("x"))
        atoms = tiled_capacity.supercell("NbCl5", repeat, 0.05, 0.0)
        cell = torch.tensor(atoms.cell.array, dtype=torch.float64, device="cuda")
        images = torch.tensor(np.stack(np.meshgrid(*[np.arange(-1, 2)] * 3, indexing="ij"), -1).reshape(-1, 3),
                              dtype=torch.float64, device="cuda")
        vectors = images @ cell
        wrapped = torch.tensor(atoms.get_positions(wrap=True), dtype=torch.float64, device="cuda")
        states = {"old": {}, "new": {}}
        sides = {"old": lambda: old.screen_codes(wrapped, vectors, args.radius, states["old"]),
                 "new": lambda: new.screen_codes(wrapped, vectors, args.radius, states["new"])}
        codes = {name: torch.sort(run().clone()).values for name, run in sides.items()}
        times = {"old": [], "new": []}
        for _ in range(args.rounds):
            for name in ("old", "new", "new", "old"):
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                sides[name]()
                stop.record()
                torch.cuda.synchronize()
                times[name].append(start.elapsed_time(stop))
        nl = SkinNeighbourList(atoms.cell.array, 6.0, "cuda", torch.float32,
                               NeighbourConfig(skin=0.3, sort_receivers="on", gpu_neighbour_list="on"))
        positions = torch.tensor(atoms.get_positions(), dtype=torch.float64, device="cuda")
        nl.build(positions)
        builds = []
        for _ in range(3):
            torch.cuda.synchronize()
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            nl.build(positions)
            stop.record()
            torch.cuda.synchronize()
            builds.append(start.elapsed_time(stop))
        entry = dict(repeat=text, atoms=len(atoms), codes=int(codes["new"].numel()),
                     same_codes=bool(torch.equal(codes["old"], codes["new"])),
                     old_ms=statistics.median(times["old"]), new_ms=statistics.median(times["new"]),
                     build_ms_new=statistics.median(builds))
        entry["new_over_old"] = entry["new_ms"] / entry["old_ms"]
        record["cells"].append(entry)
        print(json.dumps(entry))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "pair-search-ab.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
