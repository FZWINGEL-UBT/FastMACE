"""Step time of the fused integrator stage with each fixcm projection on NaCl (plan task 3-T22b).

    FASTMACE_MODELS=~/models python tools/fused_projection_timing.py --out results/3-T22b/timing.json

Two ``fm.Simulation`` engines in one process, NaCl 3×3×3 (216 atoms), MH-1 (``mh1-r2scan``), Langevin
1 fs at 600 K, with the planner's default plan (pipelined, the fused stage on). They differ only in
``fixcm_projection`` (``"ase"``, ``"mass_weighted"``). After a warm-up, alternating blocks of ``block``
steps time each (``torch.cuda.synchronize`` around a block), and the median per-step time per projection
is reported. The fused stage alone is also timed with CUDA events, ``stage(pipelined=True)`` called
``repeats`` times on each engine, again alternating.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import fastmace as fm  # before torch (SPEC §14)

# isort: split
import numpy as np
import torch
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

PROJECTIONS = ("ase", "mass_weighted")


def simulation(model, projection):
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 3))
    MaxwellBoltzmannDistribution(atoms, temperature_K=600.0, rng=np.random.default_rng(5))
    integrator = fm.integrators.Langevin(timestep_fs=1.0, temperature_K=600.0, friction_per_fs=0.02,
                                         fixcm_projection=projection)
    sim = fm.Simulation(atoms, fm.ForceField(model, atoms), integrator=integrator, seed=7,
                        workload=fm.Workload(max_temperature_K=600.0))
    engine = sim._engine
    assert engine._fused is not None, engine.fused_integrator_reason
    assert engine.integrator.config.fixcm_projection == projection
    return sim


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--block", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=2000)
    args = parser.parse_args(argv)
    model = fm.load("mh1-r2scan")
    sims = {projection: simulation(model, projection) for projection in PROJECTIONS}
    for sim in sims.values():
        sim.run(steps=args.block)  # warm-up: captures, buckets
    steps = {projection: [] for projection in PROJECTIONS}
    for index in range(args.blocks):
        order = PROJECTIONS if index % 2 == 0 else PROJECTIONS[::-1]
        for projection in order:
            torch.cuda.synchronize()
            started = time.perf_counter()
            sims[projection].run(steps=args.block)
            sims[projection]._engine.flush()
            torch.cuda.synchronize()
            steps[projection].append((time.perf_counter() - started) / args.block * 1e6)
    stage = {projection: [] for projection in PROJECTIONS}
    for index in range(10):
        order = PROJECTIONS if index % 2 == 0 else PROJECTIONS[::-1]
        for projection in order:
            fused = sims[projection]._engine._fused
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.repeats // 10):
                fused.stage(pipelined=True)
            end.record()
            torch.cuda.synchronize()
            stage[projection].append(start.elapsed_time(end) * 1e3 / (args.repeats // 10))
    record = {
        "structure": "NaCl 3x3x3 (216 atoms)", "model": "mh1-r2scan", "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "plan": {name: sims[name].plan.engine.backend for name in PROJECTIONS},
        "fastmace": fm.__version__, "block_steps": args.block,
        "step_us": {name: {"median": statistics.median(values), "blocks": values} for name, values in steps.items()},
        "fused_stage_us": {name: {"median": statistics.median(values), "rounds": values}
                           for name, values in stage.items()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    for name in PROJECTIONS:
        print(f"{name}: step {record['step_us'][name]['median']:.1f} us, fused stage "
              f"{record['fused_stage_us'][name]['median']:.2f} us")


if __name__ == "__main__":
    main()
