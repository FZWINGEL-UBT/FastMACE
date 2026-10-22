"""Rule 9 measured: the graph tier against the compiled tier with frequent host callbacks (plan task 3-T06).

SPEC §9.3 rule 9 plans the compiled tier when host callbacks come more often than every
``planner.minimal.RULE_9_HOST_CALLBACKS`` steps, because they break graph replays up. This tool checks
that on one model and structure, in one process: it builds ``fm.Simulation(mode="graph")`` and
``fm.Simulation(mode="compiled")`` from the same seed, warms both up, and then, for each callback
interval k, times blocks round-robin (graph, compiled, graph, ...). A block is ``--block-steps`` steps
as ``run(steps=k)`` followed by a callback, ``committed()``, which syncs and copies the state to the
host as an observer would. Times are ms per step; the ratio is compiled over graph (> 1: graph faster).

    PYTHONPATH=$PWD python tools/ab_host_callbacks.py --label mh1-r2scan --structure NaCl \\
        --intervals 1 5 10 --output results/3-T06-rule9/NaCl.json

Structures are benchmark names under ``FASTMACE_STRUCTURES`` (the baseline export) or files with
``PATH@NXxNYxNZ`` (tools/force_stress_gate.py's ``structure_spec``). Use an otherwise idle GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

MODES = ("graph", "compiled")


def structure_spec(spec, repeat):
    """tools/force_stress_gate.py's ``structure_spec`` (benchmark names under ``FASTMACE_STRUCTURES``)."""
    location = Path(__file__).with_name("force_stress_gate.py")
    module_spec = importlib.util.spec_from_file_location("_force_stress_gate", location)
    gate = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(gate)
    return gate.structure_spec(spec, repeat, os.environ.get("FASTMACE_STRUCTURES", "."))


def build(label, atoms, mode, seed, temperature_K):
    from ase import units

    import fastmace as fm

    integrator = fm.integrators.Langevin(timestep_fs=1.0, temperature_K=temperature_K,
                                         friction_per_fs=0.02 * units.fs)
    return fm.Simulation(atoms.copy(), fm.load(label), integrator=integrator, seed=seed, mode=mode)


def block(sim, steps, every):
    """ms per step of ``steps`` steps with a callback (``committed()``) every ``every`` steps."""
    import torch

    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(steps // every):
        sim.run(steps=every)
        sim.committed()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / (steps // every * every)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--label", default="mh1-r2scan")
    p.add_argument("--structure", default="NaCl")
    p.add_argument("--repeat", type=int, nargs=3, default=(3, 3, 3))
    p.add_argument("--intervals", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--block-steps", type=int, default=100)
    p.add_argument("--blocks", type=int, default=8, help="blocks per mode and interval")
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=20261022)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    import torch
    from ase.io import read

    name, path, repeat = structure_spec(args.structure, args.repeat)
    atoms = read(path).repeat(repeat)
    sims, setup_s = {}, {}
    for mode in MODES:
        started = time.perf_counter()
        sims[mode] = build(args.label, atoms, mode, args.seed, args.temperature_K)
        sims[mode].run(steps=args.warmup_steps)
        sims[mode].committed()
        setup_s[mode] = time.perf_counter() - started
    rows = []
    for every in args.intervals:
        steps = max(every, args.block_steps // every * every)
        for mode in MODES:  # warm the cadence up (Python paths, allocator) before timing it
            block(sims[mode], steps, every)
        times = {mode: [] for mode in MODES}
        for _ in range(args.blocks):
            for mode in MODES:
                times[mode].append(block(sims[mode], steps, every))
        median = {mode: statistics.median(values) for mode, values in times.items()}
        spread = {mode: (max(values) - min(values)) / median[mode] for mode, values in times.items()}
        rows.append({"every": every, "steps_per_block": steps, "ms_per_step": median, "relative_spread": spread,
                     "compiled_over_graph": median["compiled"] / median["graph"], "blocks": times})
        print(f"{name} every {every:>3}: graph {median['graph']:.3f} ms/step, compiled {median['compiled']:.3f} "
              f"ms/step, compiled/graph {median['compiled'] / median['graph']:.3f}", flush=True)
    plans = {mode: sim.plan for mode, sim in sims.items()}
    record = {
        "task": "3-T06", "tool": "tools/ab_host_callbacks.py", "date": datetime.now(timezone.utc).isoformat(),
        "label": args.label, "structure": name, "repeat": list(repeat), "atoms": len(atoms),
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "fastmace": fastmace.__version__,
        "backend": plans["graph"].engine.backend, "dtype": plans["graph"].precision.dtype,
        "tiers": {mode: plan.tier for mode, plan in plans.items()}, "setup_s": setup_s,
        "timestep_fs": 1.0, "temperature_K": args.temperature_K, "rows": rows,
    }
    for sim in sims.values():
        sim.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
