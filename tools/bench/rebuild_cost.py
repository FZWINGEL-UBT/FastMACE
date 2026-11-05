"""What a neighbour-list rebuild really costs per MD step (round 5, 5 November 2026).

The step breakdown times a rebuild at 11.35 ms on the A100 (OMAT-0 medium, SrTiO3 10x10x10), but the rebuilds run
beside the force call (``EngineConfig.overlap_rebuild``), so that is elapsed time while the force graph runs, not added
time. This tool runs the benchmark's simulation (as ``loup_md.py``) in short blocks, records each block's wall time
and the engine's rebuild count, and fits block time = steps x t_step + rebuilds x t_rebuild by least squares: t_rebuild
is the added time per rebuild, t_rebuild x rebuilds / steps its share of a step.

Usage: python tools/bench/rebuild_cost.py --structure PATH@10x10x10 --checkpoint CK --output out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", required=True, help="PATH@NXxNYxNZ")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--ensemble", choices=("nve", "langevin"), default="langevin")
    p.add_argument("--skin", type=float, default=0.5)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--block-steps", type=int, default=4)
    p.add_argument("--blocks", type=int, default=200)
    p.add_argument("--sha256", default="d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    import numpy as np
    import torch
    from ase.io import read

    import fastmace as fm
    from fastmace.config import NeighbourConfig

    path, repeat = args.structure.split("@")
    atoms = read(path).repeat(tuple(int(n) for n in repeat.split("x")))
    loaded = fm.load(str(args.checkpoint), sha256=args.sha256)
    field = fm.ForceField(loaded, atoms, precision="float32", rewrites="auto")
    integrator = (fm.integrators.NVE(timestep_fs=1.0) if args.ensemble == "nve" else
                  fm.integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.002))
    sim = fm.Simulation(atoms, field, integrator=integrator, workload=fm.Workload(max_temperature_K=300.0),
                        output=None, seed=1, neighbours=NeighbourConfig(skin=args.skin))
    sim.run(steps=args.warmup)
    torch.cuda.synchronize()
    engine = sim._engine
    rows = []
    for _ in range(args.blocks):
        before = engine.rebuilds
        torch.cuda.synchronize()
        t = time.perf_counter()
        sim.run(steps=args.block_steps)
        torch.cuda.synchronize()
        rows.append((time.perf_counter() - t, engine.rebuilds - before))
    sim.close()
    times = np.array([r[0] for r in rows]) * 1e3
    rebuilds = np.array([r[1] for r in rows], dtype=float)
    design = np.stack([np.full_like(rebuilds, args.block_steps), rebuilds], axis=1)
    (t_step, t_rebuild), *_ = np.linalg.lstsq(design, times, rcond=None)
    by = {int(k): statistics.median(times[rebuilds == k].tolist()) for k in sorted(set(rebuilds.tolist()))}
    total_steps = args.blocks * args.block_steps
    record = dict(tool="rebuild_cost", structure=args.structure, atoms=len(atoms), ensemble=args.ensemble,
                  skin=args.skin, block_steps=args.block_steps, blocks=args.blocks, gpu=torch.cuda.get_device_name(0),
                  fastmace=fm.__version__, rebuilds=int(rebuilds.sum()), steps=total_steps,
                  ms_per_step_fit=float(t_step), ms_per_rebuild_fit=float(t_rebuild),
                  rebuild_share_ms_per_step=float(t_rebuild * rebuilds.sum() / total_steps),
                  median_block_ms_by_rebuilds=by, block_ms=times.tolist(), block_rebuilds=rebuilds.tolist())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1))
    keys = ("gpu", "skin", "rebuilds", "steps", "ms_per_step_fit", "ms_per_rebuild_fit", "rebuild_share_ms_per_step",
            "median_block_ms_by_rebuilds")
    print(json.dumps({k: record[k] for k in keys}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
