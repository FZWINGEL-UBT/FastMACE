"""The compiled tiled call against the eager one at scale, in one process (the conductor's item (a)).

Two force fields of the same structure: the default compiled evaluation plan (its traceable model) and the eager
plan, both with their one-hot on the host. Their tiled calls: TileConfig(mode="compiled" or "eager", tile lists,
the state on the host, the species index). The compiled call's first call (compilation of every slot's sweeps)
and a steady call are timed; then ``--blocks`` alternating blocks of ``--calls`` calls, the order flipped every
block. The forces gate: energy, forces and stress of the compiled call against the eager call.

Run as ``run.sh NAME results/tiled-compiled-scale-20261101/ab_compiled.py --repeat 35x35x105``.
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
    p.add_argument("--receivers", type=int, default=4096)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--calls", type=int, default=1)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    from fastmace.model.compile import configure_allocator

    configure_allocator()
    import torch

    import fastmace as fm
    from fastmace.model.tiled import TileConfig, TiledForceField
    from fastmace.planner import minimal

    repeat = tuple(int(x) for x in args.repeat.lower().split("x"))
    atoms = tiled_capacity.supercell(args.structure, repeat, 0.0, 0.0)
    n = len(atoms)
    user = fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float32")
    planner = minimal.default_planner(user)
    plan = planner.plan_evaluation(user, atoms)
    eager_plan = dataclasses.replace(plan, engine=dataclasses.replace(plan.engine, compile_mode="none",
                                                                        dynamic_edges=False))
    tiled, record = {}, dict(structure=args.structure, atoms=n, receivers=args.receivers,
                             device=torch.cuda.get_device_name(), host=os.uname().nodename,
                             source_commit=os.environ.get("FASTMACE_SOURCE_COMMIT"), builds={})
    for name, built in (("eager", eager_plan), ("compiled", plan)):
        started = time.perf_counter()
        field = planner.build_evaluation(built, user, atoms, call_rows=args.receivers, release_node_attrs=True)
        record["builds"][name] = time.perf_counter() - started
        tiled[name] = TiledForceField(field, TileConfig(args.receivers, mode=name, neighbours="tiles",
                                                        state="host", species="index"))
    positions = tiled["eager"].field._prepare_one_off(atoms)
    outputs = {}

    def call(name):
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs[name] = tiled[name](positions)
        torch.cuda.synchronize()
        return time.perf_counter() - started

    record["first_call"] = {name: call(name) for name in ("compiled", "eager")}
    record["second_call"] = {name: call(name) for name in ("compiled", "eager")}
    record["compiled_slots"] = {str(k): list(v) for k, v in tiled["compiled"].last.get("slots", {}).items()}
    record["compiled_slot_changes"] = tiled["compiled"].slot_changes
    times = {name: [] for name in tiled}
    for block in range(args.blocks):
        order = list(tiled) if block % 2 == 0 else list(tiled)[::-1]
        for name in order:
            times[name].append(sum(call(name) for _ in range(args.calls)) / args.calls)
    a, b = statistics.median(times["eager"]), statistics.median(times["compiled"])
    e, c = outputs["eager"], outputs["compiled"]
    record.update(blocks=times, eager_median=a, compiled_median=b, ratio=b / a,
                  energy_diff_meV_per_atom=float(1e3 * (c.energy.double().sum() - e.energy.double().sum()) / n),
                  max_force_diff=float((c.forces.double() - e.forces.double()).abs().max()),
                  max_force=float(e.forces.double().abs().max()),
                  max_stress_diff=float((c.stress.double() - e.stress.double()).abs().max())
                  if c.stress is not None and e.stress is not None else None)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"ab-compiled-{args.structure}-{n}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k not in ("blocks", "compiled_slots")}))


if __name__ == "__main__":
    main()
