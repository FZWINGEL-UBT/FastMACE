"""NVE energy drift in the default plan (plan phase 4, choice 7; 4-T13, recorded, not acceptance).

``fm.Simulation`` with ``fm.integrators.NVE`` (``fixcm`` and ``dof="auto"``, decision 45) and the planner's
default plan (MH-1: hybrid float32 forces, float64 energy sums) on one structure, from Maxwell-Boltzmann
momenta at ``--temperature-K`` (``tools/ab_frontends.py``'s ``thermalize``), ``--ps`` picoseconds at
``--timestep-fs``. The conserved quantity (E_kin + E_pot, the integrator's ``conserved`` output) per atom is
read every ``--every`` steps. Recorded: the linear drift (meV/atom/ps, a least-squares fit over the whole run),
the RMS about that line and the range, as 4-T03 recorded NHC's and MTK's extended energies. Output:
``<output>/nve-drift-<structure>.json``. Run on an otherwise idle GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

TOOLS = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", default="NaCl",
                   help="Benchmark name (under FASTMACE_STRUCTURES) or a file with PATH@NXxNYxNZ")
    p.add_argument("--model", default="mh1-r2scan", help="Label (its file under FASTMACE_MODELS)")
    p.add_argument("--output", required=True, type=Path, help="Directory (created if missing)")
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--timestep-fs", type=float, default=1.0)
    p.add_argument("--ps", type=float, default=20.0)
    p.add_argument("--every", type=int, default=10, help="steps between samples")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--rewrites", nargs="*", default=[], metavar="FIELD=VALUE",
                   help="RewriteConfig overrides on the default plan, e.g. tf32x3_gemms=on (OPT4-T03)")
    p.add_argument("--precision", choices=("float32", "float64"), default="float32",
                   help="The ForceField's force precision (f64-T01: float64 plans)")
    p.add_argument("--engine", nargs="*", default=[], metavar="FIELD=VALUE",
                   help="EngineConfig overrides, e.g. fp64_emulation=off (f64-T01)")
    return p.parse_args(argv)


def drift(times_ps, values):
    """Least-squares slope (per ps), the RMS about the fitted line and the range of ``values``."""
    import numpy as np

    times_ps, values = np.asarray(times_ps, dtype=float), np.asarray(values, dtype=float)
    slope, intercept = np.polyfit(times_ps, values, 1)
    residual = values - (slope * times_ps + intercept)
    return float(slope), float(np.sqrt(np.mean(residual ** 2))), float(np.ptp(values))


def main(argv=None):
    args = parse_args(argv)
    import torch

    import fastmace as fm
    from fastmace.config import EngineConfig, RewriteConfig
    from fastmace.integrators.base import CONSERVED_KEY
    from fastmace.quick import nve_max_temperature_K

    sys.path.insert(0, str(TOOLS))
    import ab_step_time
    import force_stress_gate

    rewrites = force_stress_gate.rewrite_overrides(args.rewrites)

    spec = importlib.util.spec_from_file_location("_ab_frontends", TOOLS / "ab_frontends.py")
    frontends = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frontends)
    if ab_step_time.gpu_processes():
        raise RuntimeError("GPU must be idle")
    structure, atoms = ab_step_time.read_structure(args.structure, None)
    frontends.thermalize(atoms, args.temperature_K, args.seed)
    n = len(atoms)
    started = time.perf_counter()
    engine_options = dict(item.split("=", 1) for item in args.engine)
    field = fm.ForceField(fm.load(args.model), atoms, precision=args.precision, rewrites=RewriteConfig(**rewrites))
    sim = fm.Simulation(atoms, field, integrator=fm.integrators.NVE(timestep_fs=args.timestep_fs), seed=args.seed,
                        workload=fm.Workload(max_temperature_K=nve_max_temperature_K(atoms)),
                        engine=EngineConfig(**engine_options) if engine_options else None)
    setup_s = time.perf_counter() - started
    engine = sim._engine

    def sample(step):
        values = engine.integrator.output_values(engine.state)
        return [step, float(values[CONSERVED_KEY][0]) / n, float(values["kinetic"][0]) / n]

    steps = int(round(1000 * args.ps / args.timestep_fs))
    rows = [sample(0)]
    started = time.perf_counter()
    for step in range(args.every, steps + 1, args.every):
        sim.run(steps=args.every)
        rows.append(sample(step))
    seconds = time.perf_counter() - started
    times_ps = [row[0] * args.timestep_fs * 1e-3 for row in rows]
    slope, rms, spread = drift(times_ps, [row[1] for row in rows])
    record = dict(tool="nve_drift", created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
                  python=sys.executable, fastmace_commit=ab_step_time.fastmace_commit(), model=args.model,
                  structure=structure, natoms=n, temperature_K=args.temperature_K, timestep_fs=args.timestep_fs,
                  ps=args.ps, every=args.every, rewrites=rewrites, precision=args.precision,
                  engine_options=engine_options, plan=sim.plan.describe(), setup_s=setup_s,
                  seconds=seconds,
                  drift_meV_per_atom_per_ps=1e3 * slope, rms_meV_per_atom=1e3 * rms, range_meV_per_atom=1e3 * spread,
                  rows_step_conserved_kinetic_eV_per_atom=rows, gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__)
    sim.close()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"nve-drift-{Path(str(structure)).stem}.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"NVE {structure}: drift {1e3 * slope:.3g} meV/atom/ps, rms {1e3 * rms:.3g} meV/atom ({path})")


if __name__ == "__main__":
    main()
