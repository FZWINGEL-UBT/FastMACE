"""Whether a production build uses SPLINE_W: builds Benchmark_loup's simulation (tools/bench/loup_md.py: SrTiO3,
Langevin 300 K, 1 fs, skin 0.5, from rest) with ``--constant`` set, runs a few steps, and prints every interaction's
``wprime_backward`` (its spline: intervals, certificate, whether its kernels are compiled and in use). Exits 1 when no
interaction uses the spline, so a packed job can stop before timing a build that does not exercise it."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--structure", default="results/loup-a100-20261103/SrTiO3.cif@4x4x4")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--constant", nargs="*", default=["fastmace.kernels.oeq_streaming.spline_forward:SPLINE_W=True"])
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)

    import fastmace as fm
    from fastmace.config import NeighbourConfig

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ab_rewrites import constant_overrides
    from ase.io import read

    for (module, name), value in constant_overrides(args.constant).items():
        setattr(sys.modules[module], name, value)
    from fastmace.kernels.oeq_streaming.wprime_backward import WPrime

    path, repeat = args.structure.split("@")
    atoms = read(path).repeat(tuple(int(n) for n in repeat.split("x")))
    loaded = fm.load(str(args.checkpoint), sha256=None)
    field = fm.ForceField(loaded, atoms, precision="float32", rewrites="auto")
    sim = fm.Simulation(atoms, field, integrator=fm.integrators.Langevin(timestep_fs=1.0, temperature_K=300.0,
                                                                         friction_per_fs=0.002),
                        workload=fm.Workload(max_temperature_K=300.0), output=None, seed=1,
                        neighbours=NeighbourConfig(skin=0.5))
    sim.run(steps=args.steps)
    rows = []
    for obj in gc.get_objects():
        if isinstance(obj, WPrime):
            spline = obj.spline or {}
            rows.append(dict(repr=obj.extra_repr(), attached=obj.spline is not None,
                             in_use=obj.spline_block is not None,
                             intervals=sum(g[2] for g in spline.get("grid", [])) or None,
                             certificate=(spline.get("certificate") or {}).get("worst"),
                             grid=spline.get("grid"),
                             forward_registers=obj.spline_block[1].registers if obj.spline_block else None))
    used = any(r["in_use"] for r in rows)
    record = dict(constants=args.constant, structure=args.structure, wprime=rows, spline_in_use=used)
    print(json.dumps(record, indent=1, default=str))
    if args.output:
        args.output.write_text(json.dumps(record, indent=1, default=str) + "\n")
    return 0 if used else 1


if __name__ == "__main__":
    raise SystemExit(main())
