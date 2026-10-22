"""Same-process A/B of the neighbour-list skin (plan task 3-T05, rule 11).

Side "a" is phase 1a's fixed skin and headroom (0.3 Å, 1.15); side "b" is rule 11's skin and headroom for the
run (``fastmace.neighbors.adaptive.rule_11`` from the temperature, the lightest mass, the timestep and the atom
count; ``--target-steps`` overrides n*). Both are passed to the minimal planner explicitly
(``NeighbourConfig(skin=..., headroom="fixed", capacity_headroom=...)``), so the MD-only plan uses them. Both
sides are MH-1 hybrid float32 NVT engines of the same structure at the same temperature and timestep; they warm
up outside the compile-counter window, then their timed blocks alternate (``fastmace.bench.ab.run_ab``, ABAB or
ABBA). Besides the step times, each side records its neighbour rebuilds, early (pipelined) rebuilds, edge sizes,
captured graphs and pool resets over the timed blocks (``GraphEngine.info``) and its pair capacity. A run is
invalid when anything compiled while timed or another process used the GPU. Writes ``<output>.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_step_time  # noqa: E402

COUNTED = ("neighbour_rebuilds", "pipelined_early_rebuilds", "pipelined_rebuild_redos", "pipelined_redos",
           "pool_resets", "rebuilds_beside")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--structures", nargs="+", default=["NaCl", "NbCl5"])
    p.add_argument("--temperature-K", type=float, default=1500.0)
    p.add_argument("--timestep-fs", type=float, default=1.0)
    p.add_argument("--target-steps", type=int, default=None, help="n* of rule 11 (default: the library's)")
    p.add_argument("--warmup", type=int, default=600)
    p.add_argument("--block", type=int, default=300)
    p.add_argument("--blocks", type=int, default=12)
    p.add_argument("--pattern", choices=("abab", "abba"), default="abab")
    p.add_argument("--order", choices=("ab", "ba"), default="ab", help="Build order")
    return p.parse_args(argv)


def side_neighbours(name, base, args, cutoff):
    """(NeighbourConfig, record) of side ``name`` for structure ``base``."""
    from fastmace.config import DEFAULT_SKIN_A, NeighbourConfig
    from fastmace.neighbors import adaptive

    if name == "a":
        return (NeighbourConfig(skin=DEFAULT_SKIN_A, headroom="fixed"),
                dict(rule="fixed", skin_A=DEFAULT_SKIN_A, capacity_headroom=NeighbourConfig().capacity_headroom))
    target = args.target_steps or adaptive.TARGET_REBUILD_STEPS
    sized = adaptive.rule_11(max_temperature_K=args.temperature_K, lightest_mass_u=float(min(base.get_masses())),
                             timestep_fs=args.timestep_fs, natoms=len(base), cutoff_A=cutoff,
                             edge_bucket_fraction=NeighbourConfig().edge_bucket_fraction,
                             target_rebuild_steps=target)
    record = dataclasses.asdict(sized)
    record.update(rule="rule 11", target_rebuild_steps=target, text=sized.text())
    return NeighbourConfig(skin=sized.skin_A, headroom="fixed", capacity_headroom=sized.capacity_headroom), record


def counted(engine):
    info = engine.info()
    out = {key: info.get(key, 0) for key in COUNTED}
    out.update(edge_sizes_used=list(info["edge_sizes_used"]), graphs=len(info["graphs"]),
               capacity=int(engine.force_field.neighbours.capacity),
               candidates=int(engine.force_field.neighbours.num_candidates))
    return out


def difference(before, after):
    out = {key: after[key] - before[key] for key in COUNTED}
    out.update(new_edge_sizes=sorted(set(after["edge_sizes_used"]) - set(before["edge_sizes_used"])),
               new_graphs=after["graphs"] - before["graphs"], capacity_before=before["capacity"],
               capacity_after=after["capacity"], candidates_after=after["candidates"])
    return out


def main(argv=None):
    args = parse_args(argv)
    ab_step_time.prepare_fastmace_hybrid()
    import torch
    from torch._dynamo.utils import counters

    from fastmace.bench import ab, step_time
    from fastmace.config import EngineConfig, IntegratorConfig
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.planner.workload import Workload

    if ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    protocol = step_time.TimingProtocol(warmup=args.warmup, block=args.block, blocks=args.blocks)
    hardware = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid")).hardware()
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    loaded = loading.load_pristine(args.model, checkpoint=str(args.checkpoint), sha256=sha)
    order = list(args.order)
    meta = dict(tool="ab_skin", created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
                gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                fastmace_commit=ab_step_time.fastmace_commit(), model=args.model, checkpoint_sha256=sha,
                temperature_K=args.temperature_K, timestep_fs=args.timestep_fs, protocol=protocol.as_dict(),
                pattern=args.pattern, order=order, load=list(os.getloadavg()), structures={})
    for spec in args.structures:
        name, base = ab_step_time.read_structure(spec, None)
        sides, setup, records, plans = {}, {}, {}, {}
        for side in order:
            started = time.perf_counter()
            field = ForceField(loaded, base, precision="float32")
            neighbours, records[side] = side_neighbours(side, base, args, float(field.traits.r_max))
            planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"), neighbours=neighbours)
            integrator = IntegratorConfig(temperature_K=args.temperature_K, timestep_fs=args.timestep_fs)
            plan = planner.plan(field, base, integrator, Workload(max_temperature_K=args.temperature_K), hardware)
            engine = planner.build(plan, field, base.copy(), temperature_K=args.temperature_K)
            sides[side] = ab_step_time.FastMACESide(engine, torch)
            setup[side] = time.perf_counter() - started
            plans[side] = dict(skin=plan.neighbours.skin, capacity_headroom=plan.neighbours.capacity_headroom,
                               recapture_budget=plan.engine.recapture_budget)
        for side in order:  # warm up outside the compile-counter window (new edge sizes compile here)
            step_time.warm_up(sides[side], protocol.warmup)
        before = {side: counted(sides[side].engine) for side in sides}
        compiles = ab_step_time.compile_counters(counters)
        result = ab.run_ab(sides, dataclasses.replace(protocol, warmup=0), pattern=args.pattern)
        changes, recompiled = ab_step_time.compile_changes(compiles, ab_step_time.compile_counters(counters))
        after = {side: counted(sides[side].engine) for side in sides}
        timed_steps = protocol.block * protocol.blocks
        medians = {side: result.statistics[side].median_ms for side in sides}
        blocks = {side: result.statistics[side].block_ms for side in sides}
        during = {side: difference(before[side], after[side]) for side in sides}
        entry = dict(natoms=len(base), sides=records, plans=plans, ab=result.as_dict(),
                     ratio_b_over_a=medians["b"] / medians["a"],
                     pair_ratios=[b / a for a, b in zip(blocks["a"], blocks["b"])], setup_s=setup,
                     timed=during, steps_per_rebuild={side: timed_steps / max(1, during[side]["neighbour_rebuilds"])
                                                      for side in sides},
                     compile_during_timing=dict(changes=changes, recompiled=recompiled),
                     observables={side: sides[side].observables() for side in sides}, load_after=list(os.getloadavg()),
                     other_gpu_processes=[pid for pid in ab_step_time.gpu_processes() if pid != os.getpid()])
        entry["valid"] = not recompiled and not entry["other_gpu_processes"]
        meta["structures"][name] = entry
        print(name, " ".join(f"{s} {medians[s]:.4f} ms skin {records[s]['skin_A']:g} rebuilds "
                             f"{during[s]['neighbour_rebuilds']}" for s in sides),
              f"b/a {entry['ratio_b_over_a']:.4f}", "valid" if entry["valid"] else "INVALID", flush=True)
        args.output.with_name(args.output.name + ".json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        sides = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
