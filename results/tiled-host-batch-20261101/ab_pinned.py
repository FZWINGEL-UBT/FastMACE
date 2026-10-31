"""The tiled call with its host state in pageable memory (the default now) against the same call with the state in
pinned memory, in one process: alternating blocks of eager tiled calls (R = 4,096, tile lists, species index),
the order flipped every block.

Run as ``run.sh NAME results/tiled-host-batch-20261101/ab_pinned.py --repeat 6x6x20``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import sys
import time
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "bench"))
import tiled_capacity  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--structure", default="NbCl5")
    p.add_argument("--repeat", required=True)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--calls", type=int, default=3)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    import torch

    import fastmace as fm
    from fastmace.model.tiled import TileConfig, TiledForceField
    from fastmace.planner import minimal

    repeat = tuple(int(x) for x in args.repeat.lower().split("x"))
    atoms = tiled_capacity.supercell(args.structure, repeat, 0.0, 0.0)
    user = fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float32")
    planner = minimal.default_planner(user)
    plan = planner.plan_evaluation(user, atoms)
    eager = dataclasses.replace(plan, engine=dataclasses.replace(plan.engine, compile_mode="none",
                                                                   dynamic_edges=False))
    tiled = {}
    for name in ("pageable", "pinned"):
        field = planner.build_evaluation(eager, user, atoms, call_rows=4096)
        tiled[name] = TiledForceField(field, TileConfig(4096, neighbours="tiles", state="host", species="index"))
    positions = tiled["pageable"].field._prepare_one_off(atoms)
    outputs = {}

    def call(name):
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs[name] = tiled[name](positions)
        torch.cuda.synchronize()
        return time.perf_counter() - started

    for name in tiled:
        call(name)
    state = tiled["pinned"]._g["state"]
    state.tensors = {key: tensor.pin_memory() for key, tensor in state.tensors.items()}
    assert all(t.is_pinned() for t in state.tensors.values())
    assert not any(t.is_pinned() for t in tiled["pageable"]._g["state"].tensors.values())
    for name in tiled:
        call(name)
    times = {name: [] for name in tiled}
    for block in range(args.blocks):
        order = list(tiled) if block % 2 == 0 else list(tiled)[::-1]
        for name in order:
            times[name].append(sum(call(name) for _ in range(args.calls)) / args.calls)
    a, b = statistics.median(times["pinned"]), statistics.median(times["pageable"])
    record = dict(structure=args.structure, atoms=len(atoms), device=torch.cuda.get_device_name(),
                  host=os.uname().nodename, source_commit=os.environ.get("FASTMACE_SOURCE_COMMIT"),
                  blocks=times, pinned_median=a, pageable_median=b, ratio=b / a,
                  max_force_diff=float((outputs["pageable"].forces.double()
                                        - outputs["pinned"].forces.double()).abs().max()))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"ab-pinned-{args.structure}-{len(atoms)}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k != "blocks"}))


if __name__ == "__main__":
    main()
