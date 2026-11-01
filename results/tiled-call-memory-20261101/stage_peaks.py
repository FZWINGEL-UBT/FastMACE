"""The device memory of a tiled call's whole-system stages: each stage's peak above the memory allocated when it
starts, and what it leaves allocated, over ``--calls`` calls. The stages: the tile list's binning
(``TileNeighbours.bin``), the tile plan (``plan_tiles``), the call's global buffers (``_globals``), the edge source
(``_source``, with the binning), and the whole call. The force field as tiled_capacity.py builds it (tile lists,
the state on the host and aliased, the species index); eager tiles.

Run as ``run.sh NAME results/tiled-call-memory-20261101/stage_peaks.py --repeat 47x47x141``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import functools
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
    p.add_argument("--calls", type=int, default=2)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    from fastmace.model.compile import configure_allocator

    configure_allocator()
    import torch

    import fastmace as fm
    from fastmace.model import tiled as tiled_module
    from fastmace.model.tiled import TileConfig, TiledForceField
    from fastmace.neighbors.tile_list import TileNeighbours
    from fastmace.planner import minimal

    repeat = tuple(int(x) for x in args.repeat.lower().split("x"))
    atoms = tiled_capacity.supercell(args.structure, repeat, 0.0, 0.0)
    n = len(atoms)
    records = []
    depth = [0]
    running = [0]  # the call's peak across the stages' resets of the peak counter

    def measured(owner, name, label):
        original = getattr(owner, name)

        @functools.wraps(original)
        def wrapped(*a, **k):
            if depth[0]:  # nested stages are measured by the outer one only
                return original(*a, **k)
            torch.cuda.synchronize()
            start = torch.cuda.memory_allocated()
            running[0] = max(running[0], torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            depth[0] += 1
            try:
                return original(*a, **k)
            finally:
                depth[0] -= 1
                torch.cuda.synchronize()
                records.append(dict(stage=label, call=len([r for r in records if r["stage"] == "call"]) + 1,
                                    start=start, peak_above=torch.cuda.max_memory_allocated() - start,
                                    left=torch.cuda.memory_allocated() - start))

        setattr(owner, name, wrapped)

    with contextlib.ExitStack() as stack:
        stack.enter_context(tiled_capacity.no_global_list(True))
        user = fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float32")
        planner = minimal.default_planner(user)
        plan = planner.plan_evaluation(user, atoms)
        eager = dataclasses.replace(plan, engine=dataclasses.replace(plan.engine, compile_mode="none",
                                                                       dynamic_edges=False))
        field = planner.build_evaluation(eager, user, atoms, call_rows=4096, release_node_attrs=True)
        tiled = TiledForceField(field, TileConfig(4096, mode="eager", neighbours="tiles", state="host",
                                                  alias_state=True, species="index"))
        positions = field._prepare_one_off(atoms)
        for owner, name, label in ((TileNeighbours, "bin", "tile list: binning"),
                                   (tiled_module, "plan_tiles", "tile plan"),
                                   (TiledForceField, "_globals", "global buffers"),
                                   (TiledForceField, "_source", "edge source (with the binning)")):
            measured(owner, name, label)
        out = None
        for _ in range(args.calls):
            torch.cuda.synchronize()
            start = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            running[0] = 0
            out = tiled(positions)  # the previous call's outputs stay alive meanwhile, as in a caller's loop
            torch.cuda.synchronize()
            peak = max(running[0], torch.cuda.max_memory_allocated())
            records.append(dict(stage="call", call=len([r for r in records if r["stage"] == "call"]) + 1,
                                start=start, peak_above=peak - start,
                                left=torch.cuda.memory_allocated() - start))
        del out
    record = dict(structure=args.structure, atoms=n, host=os.uname().nodename,
                  source_commit=os.environ.get("FASTMACE_SOURCE_COMMIT"), device=torch.cuda.get_device_name(),
                  stages=records)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"stage-peaks-{args.structure}-{n}.json").write_text(json.dumps(record, indent=1))
    for r in records:
        print(f"call {r['call']} {r['stage']:34s} start {r['start'] / 2**30:6.2f} GiB  peak above "
              f"{r['peak_above'] / 2**30:6.2f} GiB ({r['peak_above'] / n:6.1f} B/atom)  left "
              f"{r['left'] / 2**30:6.2f} GiB ({r['left'] / n:5.1f} B/atom)")


if __name__ == "__main__":
    main()
