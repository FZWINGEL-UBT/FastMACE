"""Where the ``fm.md``/``Simulation`` loop loses time against the engine loop (fp32 profile, O1; 28 October 2026).

One process, one ``fm.md`` setup, then alternating blocks of ``--block`` steps (``--repeats`` rounds) of:

- ``engine``: ``sim._engine.step()`` alone (the step tool's loop), then a flush;
- ``no-output``: ``sim.run`` with thermo rows every 10^9 steps and no trajectory or restart;
- ``thermo``: thermo rows every 20 steps, no trajectory;
- ``default``: the production cadences (thermo and trajectory every 20 steps).

Per block: wall ms per step, and host seconds inside ``engine.snapshot`` (its flush), ``_queue.drain`` (the wait for
the copy and the writing) and the writer's ``write``, wrapped on the instances only. Measurement only; hold the
local GPU reservation. Writes overhead.json into --output.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ab_step_time  # noqa: E402

HOST = defaultdict(float)


def wrap(owner, name, label):
    original = getattr(owner, name)

    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            HOST[label] += time.perf_counter() - started

    setattr(owner, name, wrapper)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--structure", default="NaCl")
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--temperature-K", type=float, default=1000.0)
    p.add_argument("--timestep-fs", type=float, default=1.0)
    p.add_argument("--block", type=int, default=600)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--variants", nargs="+", default=["engine", "no-output", "thermo", "default"])
    args = p.parse_args(argv)
    import numpy as np
    import torch
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    import fastmace as fm

    if ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    args.output.mkdir(parents=True, exist_ok=False)
    structure, atoms = ab_step_time.read_structure(args.structure, None)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_K, rng=np.random.default_rng(20261007))
    Stationary(atoms)
    result = fm.md(atoms, model=args.model, temperature_K=args.temperature_K, timestep_fs=args.timestep_fs,
                   steps=300, output=fm.Output(str(args.output / "run")), seed=1)
    sim = result.simulation
    engine = sim._engine
    base = sim.output
    outputs = {"no-output": dataclasses.replace(base, thermo_every=10**9, trajectory_every=None, restart_every=None),
               "thermo": dataclasses.replace(base, trajectory_every=None, restart_every=None),
               "default": dataclasses.replace(base, restart_every=None)}
    wrap(engine, "snapshot", "engine.snapshot (flush + copies)")
    wrap(sim._queue, "drain", "queue.drain (wait + write)")
    if sim._writer is not None:
        wrap(sim._writer, "write", "writer.write")
    blocks = defaultdict(list)
    host = defaultdict(lambda: defaultdict(list))
    sim.run(steps=args.block)  # warm-up with the default output
    for _ in range(args.repeats):
        for variant in args.variants:
            HOST.clear()
            torch.cuda.synchronize()
            started = time.perf_counter()
            if variant == "engine":
                for _ in range(args.block):
                    engine.step()
                engine.flush()
            else:
                sim.output = outputs[variant]
                sim.run(steps=args.block)
                sim.flush_output()
            torch.cuda.synchronize()
            blocks[variant].append((time.perf_counter() - started) * 1e3 / args.block)
            for label, seconds in HOST.items():
                host[variant][label].append(seconds * 1e3 / args.block)
    sim.output = base
    summary = dict(
        tool="sim_loop_overhead", created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
        fastmace_commit=ab_step_time.fastmace_commit(), gpu=torch.cuda.get_device_name(), structure=structure,
        natoms=len(atoms), block=args.block, repeats=args.repeats,
        ms_per_step={v: dict(median=statistics.median(x), blocks=x) for v, x in blocks.items()},
        host_ms_per_step={v: {k: statistics.median(x) for k, x in d.items()} for v, d in host.items()})
    (args.output / "overhead.json").write_text(json.dumps(summary, indent=2) + "\n")
    engine_ms = summary["ms_per_step"].get("engine", {}).get("median")
    for v, d in summary["ms_per_step"].items():
        extra = f" (+{d['median'] - engine_ms:.3f} vs engine)" if engine_ms else ""
        print(f"{structure} {v:10} {d['median']:.3f} ms/step{extra}  host: {summary['host_ms_per_step'].get(v, {})}")


if __name__ == "__main__":
    main()
