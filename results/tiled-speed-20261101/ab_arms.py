"""Tiled call arms at scale, in one process (the speed phase).

Arms (``--arms``, two or more): ``eager`` or ``compiled``, with suffixes ``+budget`` (budget_receivers (8192,
16384); without it, none, whatever the default), ``+nocache`` (list_cache off), ``+noreuse`` (list_reuse off;
on by default, repeated calls on the same positions reuse the lists after the first; ``+reuse`` names the default),
``+nooverlap`` (overlap off) and
``+old`` (the tiled call of an older tree, ``--old-tree``: its ``fastmace/model/tiled.py`` and
``fastmace/neighbors/tile_list.py`` loaded as modules of their own, everything else this tree's; only for trees that
differ from this one in those two files). The compiled
arms run on a force field of the default compiled evaluation plan (its traceable model), the eager arms on one of
the eager plan; both with their one-hot on the host, the force field's global list left empty (the tiles build
their own, as ``tiled_capacity.py --tile-list``). Every arm: TileConfig(tile lists, the state on the host, the
species index, the state aliased).

Each arm's first call (compilation, or the budget's measurement) and its following calls until steady are timed;
then ``--blocks`` alternating blocks of ``--calls`` calls, the order rotated every block. The forces gate: energy,
forces and stress of every arm against the first. Dynamo's counters (graphs, graph breaks, recompiles) are
recorded; run with ``TORCH_LOGS=graph_breaks,recompiles`` for their reasons in the log.

Run as ``run.sh NAME results/tiled-speed-20261101/ab_arms.py --repeat 35x35x105 --arms compiled,compiled+nocache``.
"""

from __future__ import annotations

import argparse
import contextlib
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


def old_tiled(tree):
    """An older tree's tiled module, its tile list module beside it (both under names of their own)."""
    import importlib.util

    if "tiled_old" in sys.modules:
        return sys.modules["tiled_old"]
    for name, path, substitutions in (
            ("tile_list_old", tree / "fastmace" / "neighbors" / "tile_list.py",
             (("from .skin import", "from fastmace.neighbors.skin import"),)),
            ("tiled_old", tree / "fastmace" / "model" / "tiled.py",
             (("from fastmace.neighbors.tile_list import", "from tile_list_old import"),))):
        source = path.read_text(encoding="utf-8")
        for a, b in substitutions:
            assert a in source, (path, a)
            source = source.replace(a, b)
        spec = importlib.util.spec_from_loader(name, loader=None, origin=str(path))
        module = importlib.util.module_from_spec(spec)
        module.__file__ = str(path)
        sys.modules[name] = module
        exec(compile(source, str(path), "exec"), module.__dict__)
    return sys.modules["tiled_old"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--structure", default="NbCl5")
    p.add_argument("--repeat", required=True)
    p.add_argument("--receivers", type=int, default=4096)
    p.add_argument("--arms", default="eager,compiled")
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--calls", type=int, default=1)
    p.add_argument("--old-tree", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    from fastmace.model.compile import configure_allocator

    configure_allocator()
    import torch

    import fastmace as fm
    from fastmace.model.tiled import TileConfig, TiledForceField
    from fastmace.planner import minimal

    arms = args.arms.split(",")
    repeat = tuple(int(x) for x in args.repeat.lower().split("x"))
    atoms = tiled_capacity.supercell(args.structure, repeat, 0.0, 0.0)
    n = len(atoms)
    record = dict(structure=args.structure, atoms=n, receivers=args.receivers, arms=arms,
                  device=torch.cuda.get_device_name(), host=os.uname().nodename,
                  source_commit=os.environ.get("FASTMACE_SOURCE_COMMIT"), builds={}, warm={})
    with contextlib.ExitStack() as stack:
        stack.enter_context(tiled_capacity.no_global_list(True))
        user = fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float32")
        planner = minimal.default_planner(user)
        plan = planner.plan_evaluation(user, atoms)
        plans = dict(compiled=plan, eager=dataclasses.replace(plan, engine=dataclasses.replace(
            plan.engine, compile_mode="none", dynamic_edges=False)))
        fields, tiled = {}, {}
        for arm in arms:
            mode = arm.split("+")[0]
            if mode not in fields:
                started = time.perf_counter()
                fields[mode] = planner.build_evaluation(plans[mode], user, atoms, call_rows=args.receivers,
                                                        release_node_attrs=True)
                record["builds"][mode] = time.perf_counter() - started
            budget = (8192, 16384) if "+budget" in arm else ()
            options = dict(mode=mode, neighbours="tiles", state="host", alias_state=True, species="index",
                           budget_receivers=budget, list_cache="+nocache" not in arm,
                           list_reuse="+noreuse" not in arm)
            if "+old" in arm:
                old = old_tiled(args.old_tree)
                tiled[arm] = old.TiledForceField(fields[mode], old.TileConfig(args.receivers, **options))
            else:
                tiled[arm] = TiledForceField(fields[mode], TileConfig(args.receivers, overlap="+nooverlap" not in arm,
                                                                      **options))
        positions = fields[arms[0].split("+")[0]]._prepare_one_off(atoms)
        outputs = {}

        def call(arm):
            torch.cuda.synchronize()
            started = time.perf_counter()
            outputs[arm] = tiled[arm](positions)
            torch.cuda.synchronize()
            return time.perf_counter() - started

        for arm in arms:  # until steady: compilation, the budget's switch (and its recompilation)
            record["warm"][arm] = [call(arm) for _ in range(3 if "+budget" in arm else 2)]
            record.setdefault("list_cache_bytes", {})[arm] = tiled[arm].last.get("list_cache_bytes")
            record.setdefault("receivers_chosen", {})[arm] = tiled[arm].receivers
            record.setdefault("slot_changes", {})[arm] = tiled[arm].slot_changes
        times = {arm: [] for arm in arms}
        for block in range(args.blocks):
            order = arms[block % len(arms):] + arms[:block % len(arms)]
            for arm in order:
                times[arm].append(sum(call(arm) for _ in range(args.calls)) / args.calls)
    medians = {arm: statistics.median(t) for arm, t in times.items()}
    first = outputs[arms[0]]
    gate = {}
    for arm in arms[1:]:
        o = outputs[arm]
        gate[arm] = dict(energy_diff_meV_per_atom=float(1e3 * (o.energy.double().sum() - first.energy.double().sum())
                                                        / n),
                         max_force_diff=float((o.forces.double() - first.forces.double()).abs().max()),
                         max_stress_diff=float((o.stress.double() - first.stress.double()).abs().max())
                         if o.stress is not None and first.stress is not None else None)
    from torch._dynamo.utils import counters

    record.update(blocks=times, medians=medians, ratios={arm: medians[arm] / medians[arms[0]] for arm in arms},
                  gate=gate, max_force=float(first.forces.double().abs().max()),
                  dynamo_counters={group: dict(values) for group, values in counters.items()})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"ab-{'-'.join(arms).replace('+', '_')}-{args.structure}-{n}.json").write_text(
        json.dumps(record, indent=1, default=str))
    print(json.dumps({k: v for k, v in record.items() if k not in ("blocks", "dynamo_counters")}, default=str))
    print(json.dumps({"dynamo_counters": record["dynamo_counters"]}, default=str)[:3000])


if __name__ == "__main__":
    main()
