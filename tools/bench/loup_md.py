"""MD-step time per atom under the Benchmark_loup protocol (MargrafGroup, A100, 3 November 2026), one fresh process.

``fm.Simulation`` on the graph engine from rest (no initial velocities), skin --skin, 1 fs, NVE or Langevin 300 K
(friction 0.002 per fs), seed 1, no output: --warmup untimed steps, then --blocks timed blocks of --steps steps between
CUDA synchronisations. ``--rewrites FIELD=VALUE ...`` sets RewriteConfig fields, the others stay "auto" (empty: the
plan's own "auto"). Prints and writes one JSON record with the plan's enabled rewrites, the GPU and the times.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--structure", required=True, help="PATH@NXxNYxNZ")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--ensemble", choices=("nve", "langevin"), default="nve")
    p.add_argument("--skin", type=float, default=0.5)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--blocks", type=int, default=1)
    p.add_argument("--rewrites", nargs="*", default=[], metavar="FIELD=VALUE")
    p.add_argument("--sha256", default="d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a",
                   help="The checkpoint's SHA-256 (default: MACE-OMAT-0 medium's)")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    import torch
    from ase.io import read

    import fastmace as fm
    from fastmace.config import NeighbourConfig, RewriteConfig

    path, repeat = args.structure.split("@")
    atoms = read(path).repeat(tuple(int(n) for n in repeat.split("x")))
    started = time.perf_counter()
    loaded = fm.load(str(args.checkpoint), sha256=args.sha256)
    fields = dict(item.split("=", 1) for item in args.rewrites)
    field = fm.ForceField(loaded, atoms, precision="float32", rewrites=RewriteConfig(**fields) if fields else "auto")
    integrator = (fm.integrators.NVE(timestep_fs=1.0) if args.ensemble == "nve" else
                  fm.integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.002))
    sim = fm.Simulation(atoms, field, integrator=integrator, workload=fm.Workload(max_temperature_K=300.0),
                        output=None, seed=1, neighbours=NeighbourConfig(skin=args.skin))
    setup_s = time.perf_counter() - started
    sim.run(steps=args.warmup)
    torch.cuda.synchronize()
    blocks = []
    for _ in range(args.blocks):
        torch.cuda.synchronize()
        t = time.perf_counter()
        sim.run(steps=args.steps)
        torch.cuda.synchronize()
        blocks.append(time.perf_counter() - t)
    n = len(atoms)
    plan = getattr(sim, "plan", None)
    rewrites = getattr(plan, "rewrites", None)
    enabled = sorted(k for k, v in vars(rewrites).items() if v == "on") if rewrites is not None else None
    record = dict(tool="loup_md", structure=args.structure, atoms=n, ensemble=args.ensemble, skin=args.skin,
                  warmup=args.warmup, steps=args.steps, blocks=args.blocks, requested=fields,
                  gpu=torch.cuda.get_device_name(0), fastmace=fm.__version__, setup_s=setup_s, block_s=blocks,
                  us_per_atom_step=[b / (args.steps * n) * 1e6 for b in blocks],
                  median_us_per_atom_step=statistics.median(blocks) / (args.steps * n) * 1e6,
                  ms_per_step=statistics.median(blocks) / args.steps * 1e3, rewrites_on=enabled,
                  max_memory_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
    sim.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1))
    print(json.dumps({k: record[k] for k in ("gpu", "atoms", "ensemble", "requested", "median_us_per_atom_step",
                                             "ms_per_step", "rewrites_on")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
