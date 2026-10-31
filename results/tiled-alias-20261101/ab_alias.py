"""The tiled call with its host state aliased (TileConfig.alias_state) against the plain host state, in one
process: two tiled force fields on one field (eager tiles, R = 4,096, tile lists, the species index), alternating
blocks of calls, the order flipped every block. Records the ratio, the state bytes of both and whether energy,
forces and stress are bitwise equal (with their differences).

Run as ``run.sh NAME results/tiled-alias-20261101/ab_alias.py --repeat 6x6x20``.
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
    field = planner.build_evaluation(eager, user, atoms, call_rows=4096, release_node_attrs=True)
    tiled = {name: TiledForceField(field, TileConfig(4096, neighbours="tiles", state="host", species="index",
                                                     alias_state=alias))
             for name, alias in (("plain", False), ("aliased", True))}
    positions = field._prepare_one_off(atoms)
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
    a, b = statistics.median(times["plain"]), statistics.median(times["aliased"])
    p_, q = outputs["plain"], outputs["aliased"]
    record = dict(structure=args.structure, atoms=len(atoms), device=torch.cuda.get_device_name(),
                  host=os.uname().nodename, source_commit=os.environ.get("FASTMACE_SOURCE_COMMIT"),
                  blocks=times, plain_median=a, aliased_median=b, ratio=b / a,
                  state_bytes={name: t.last["global_bytes"] for name, t in tiled.items()},
                  energy_equal=bool(torch.equal(p_.energy, q.energy)),
                  forces_equal=bool(torch.equal(p_.forces, q.forces)),
                  stress_equal=bool(torch.equal(p_.stress, q.stress)) if p_.stress is not None else None,
                  # The device's force scatter is atomic (any two calls differ in rounding); the bitwise gate is
                  # the CPU test.
                  max_force_diff=float((p_.forces.double() - q.forces.double()).abs().max()),
                  energy_diff=float(p_.energy.double().sum() - q.energy.double().sum()))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"ab-alias-{args.structure}-{len(atoms)}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k != "blocks"}))


if __name__ == "__main__":
    main()
