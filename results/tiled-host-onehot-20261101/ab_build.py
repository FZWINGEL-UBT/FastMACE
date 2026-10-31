"""The force field built with its one-hot on the host (``release_node_attrs=True``) against the current build
(one-hot on the device, released by the tiled path), in one process: each build timed and its device peak
measured, then alternating blocks of eager tiled calls (R = 4,096, tile lists, the state on the host, the species
index), the order flipped every block.

Run as ``run.sh NAME results/tiled-host-onehot-20261101/ab_build.py --repeat 6x6x20``.
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
    tiled, builds = {}, {}
    for name, release in (("device_build", False), ("host_build", True)):
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        field = planner.build_evaluation(eager, user, atoms, call_rows=4096, release_node_attrs=release)
        torch.cuda.synchronize()
        builds[name] = dict(seconds=time.perf_counter() - started,
                            device_peak_above=int(torch.cuda.max_memory_allocated() - before))
        tiled[name] = TiledForceField(field, TileConfig(4096, neighbours="tiles", state="host", species="index"))
    positions = tiled["device_build"].field._prepare_one_off(atoms)
    outputs = {}

    def call(name):
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs[name] = tiled[name](positions)
        torch.cuda.synchronize()
        return time.perf_counter() - started

    for name in tiled:
        call(name)
        call(name)
    times = {name: [] for name in tiled}
    for block in range(args.blocks):
        order = list(tiled) if block % 2 == 0 else list(tiled)[::-1]
        for name in order:
            times[name].append(sum(call(name) for _ in range(args.calls)) / args.calls)
    a, b = statistics.median(times["device_build"]), statistics.median(times["host_build"])
    record = dict(structure=args.structure, atoms=len(atoms), device=torch.cuda.get_device_name(),
                  host=os.uname().nodename, source_commit=os.environ.get("FASTMACE_SOURCE_COMMIT"), builds=builds,
                  blocks=times, device_build_median=a, host_build_median=b, ratio=b / a,
                  energy_diff=float(outputs["host_build"].energy.double().sum()
                                    - outputs["device_build"].energy.double().sum()),
                  max_force_diff=float((outputs["host_build"].forces.double()
                                        - outputs["device_build"].forces.double()).abs().max()))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"ab-build-{args.structure}-{len(atoms)}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k != "blocks"}))


if __name__ == "__main__":
    main()
