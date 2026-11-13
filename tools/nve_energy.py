"""Equilibrate, save, and run NVE from a saved state with an energy log (paper experiment 1).

  equilibrate  Langevin (``fm.integrators.Langevin``) at ``--temperature-K`` for ``--ps`` from ``--start`` (a saved
               state) or from the structure with Maxwell-Boltzmann momenta; writes ``--state`` (npz: numbers,
               positions, momenta, masses, cell) and ``<state>.json`` (its manifest and SHA-256)
  nve          ``fm.integrators.NVE`` from ``--start`` for ``--ps`` at ``--timestep-fs``; the conserved energy
               (E_kin + E_pot, the integrator's ``conserved`` output) and the kinetic energy every ``--every`` steps

Both build the configuration of ``tools/run_manifest.py`` (``--config a|b|c``) in this process, before anything is
built. The NVE record: the least-squares drift (meV/atom/ps) with its standard error (ordinary, and inflated by the
residuals' lag-1 autocorrelation), the RMS about the line, the range, the mean temperature (3N - 3 degrees of
freedom with fixcm), the temperature over the first and last tenth, the neighbour rebuilds (the engine's counter),
wall time and the manifest. Output: ``<output>/nve-<tag>.json`` and ``nve-<tag>.csv`` (step, conserved, kinetic,
potential, eV).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import run_manifest  # noqa: E402

KB_EV = 8.617333262e-5


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("equilibrate", "nve"))
    p.add_argument("--config", choices=sorted(run_manifest.CONFIGS), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="omat0-medium")
    p.add_argument("--cif", type=Path, help="Unit cell when there is no --start")
    p.add_argument("--repeat", default="10x10x10")
    p.add_argument("--start", type=Path, help="Saved state (npz) to start from")
    p.add_argument("--state", type=Path, help="equilibrate: where to save the final state")
    p.add_argument("--output", type=Path, help="nve: record directory")
    p.add_argument("--tag", default=None)
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--friction-per-fs", type=float, default=0.01)
    p.add_argument("--timestep-fs", type=float, default=1.0)
    p.add_argument("--ps", type=float, required=True)
    p.add_argument("--every", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args(argv)


def load_state(path):
    import numpy as np
    from ase import Atoms

    data = np.load(path)
    atoms = Atoms(numbers=data["numbers"], positions=data["positions"], cell=data["cell"], pbc=True)
    atoms.set_masses(data["masses"])
    atoms.set_momenta(data["momenta"])
    return atoms


def save_state(path, committed, atoms):
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        np.savez(handle, numbers=atoms.numbers, positions=np.asarray(committed.positions, dtype=np.float64),
                 momenta=np.asarray(committed.momenta, dtype=np.float64), masses=atoms.get_masses(),
                 cell=np.asarray(committed.cell, dtype=np.float64))


def drift_fit(times_ps, values):
    """Least-squares slope with its ordinary standard error and one inflated by the residuals' lag-1
    autocorrelation rho (effective samples n (1 - rho) / (1 + rho))."""
    import numpy as np

    t, y = np.asarray(times_ps, float), np.asarray(values, float)
    n = len(t)
    slope, intercept = np.polyfit(t, y, 1)
    res = y - (slope * t + intercept)
    sxx = np.sum((t - t.mean()) ** 2)
    se = float(np.sqrt(np.sum(res ** 2) / max(n - 2, 1) / sxx))
    rho = float(np.sum(res[1:] * res[:-1]) / np.sum(res ** 2)) if n > 2 else 0.0
    factor = np.sqrt((1 + rho) / (1 - rho)) if -1 < rho < 1 else float("inf")
    return dict(slope=float(slope), se=se, se_ar1=float(se * factor), rho1=rho,
                rms=float(np.sqrt(np.mean(res ** 2))), range=float(np.ptp(y)), std=float(np.std(y)))


def main(argv=None):
    args = parse_args(argv)
    rewrites = run_manifest.apply_config(args.config)
    import numpy as np
    import torch

    import fastmace as fm
    from fastmace.config import RewriteConfig
    from fastmace.integrators.base import CONSERVED_KEY
    from fastmace.quick import nve_max_temperature_K

    if args.start is not None:
        atoms = load_state(args.start)
        structure = f"state {args.start} sha256 {run_manifest.sha256(args.start)}"
    else:
        from ase.io import read
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

        atoms = read(args.cif).repeat(tuple(int(n) for n in args.repeat.split("x")))
        atoms.set_pbc(True)
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_K, rng=np.random.default_rng(args.seed))
        Stationary(atoms)
        structure = f"{args.cif}@{args.repeat} Maxwell-Boltzmann {args.temperature_K} K seed {args.seed}"
    n = len(atoms)
    model = fm.load(args.model, checkpoint=args.checkpoint, sha256=run_manifest.sha256(args.checkpoint))
    started = time.perf_counter()
    field = fm.ForceField(model, atoms, precision="float32", rewrites=RewriteConfig(**rewrites) if rewrites else "auto")
    if args.mode == "equilibrate":
        integrator = fm.integrators.Langevin(timestep_fs=args.timestep_fs, temperature_K=args.temperature_K,
                                             friction_per_fs=args.friction_per_fs)
        workload = fm.Workload(max_temperature_K=max(args.temperature_K, nve_max_temperature_K(atoms)))
    else:
        integrator = fm.integrators.NVE(timestep_fs=args.timestep_fs)
        workload = fm.Workload(max_temperature_K=max(args.temperature_K, nve_max_temperature_K(atoms)))
    sim = fm.Simulation(atoms, field, integrator=integrator, seed=args.seed, workload=workload)
    setup_s = time.perf_counter() - started
    engine = sim._engine
    dof = 3 * n - 3

    def sample(step):
        values = engine.integrator.output_values(engine.state)
        kinetic = float(values["kinetic"][0])
        total = float(values["energy"][0]) + kinetic  # E_pot (float64 sum) + E_kin
        # NVE: the integrator's own conserved output (equal to the total; Langevin has none)
        conserved = float(values[CONSERVED_KEY][0]) if CONSERVED_KEY in values else total
        return [step, conserved, kinetic]

    steps = int(round(1000 * args.ps / args.timestep_fs))
    rows = [sample(0)]
    started = time.perf_counter()
    for step in range(args.every, steps + 1, args.every):
        sim.run(steps=args.every)
        rows.append(sample(step))
        if not np.isfinite(rows[-1][1]):
            break
    seconds = time.perf_counter() - started
    info = dict(engine.info())
    temps = [2 * r[2] / (dof * KB_EV) for r in rows]
    tenth = max(1, len(rows) // 10)
    md = dict(mode=args.mode, steps=steps, timestep_fs=args.timestep_fs, ps=args.ps, every=args.every,
              temperature_target_K=args.temperature_K, setup_s=setup_s, seconds=seconds,
              ms_per_step=1e3 * seconds / max(steps, 1), neighbour_rebuilds=info.get("neighbour_rebuilds"),
              engine_info=info, start_temperature_K=temps[0], end_temperature_K=temps[-1],
              temperature_first_tenth_K=float(np.mean(temps[:tenth])),
              temperature_last_tenth_K=float(np.mean(temps[-tenth:])), mean_temperature_K=float(np.mean(temps)),
              dof=dof, integrator=type(integrator).__name__, friction_per_fs=args.friction_per_fs
              if args.mode == "equilibrate" else None, seed=args.seed)
    manifest = run_manifest.manifest(config=args.config, checkpoint=args.checkpoint, structure=structure, atoms=atoms,
                                     plan=sim.plan, extra=dict(md=md, start_state=structure))
    if args.mode == "equilibrate":
        committed = sim.committed()
        sim.close()
        save_state(args.state, committed, atoms)
        manifest["state_sha256"] = run_manifest.sha256(args.state)
        manifest["energy_rows_step_conserved_kinetic_eV"] = rows
        Path(str(args.state) + ".json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
        print(f"equilibrated {n} atoms {args.ps} ps at {args.temperature_K} K: T last tenth "
              f"{md['temperature_last_tenth_K']:.1f} K, {md['ms_per_step']:.2f} ms/step -> {args.state}")
        return
    sim.close()
    times_ps = [r[0] * args.timestep_fs * 1e-3 for r in rows]
    fit = drift_fit(times_ps, [r[1] / n for r in rows])
    result = dict(drift_meV_per_atom_per_ps=1e3 * fit["slope"], drift_se=1e3 * fit["se"],
                  drift_se_ar1=1e3 * fit["se_ar1"], residual_rho1=fit["rho1"], rms_meV_per_atom=1e3 * fit["rms"],
                  range_meV_per_atom=1e3 * fit["range"], std_meV_per_atom=1e3 * fit["std"],
                  potential_std_meV_per_atom=1e3 * float(np.std([(r[1] - r[2]) / n for r in rows])),
                  samples=len(rows), finite=bool(np.isfinite(rows[-1][1])))
    manifest.update(result=result, timing=dict(schedule=f"{steps} steps in runs of {args.every}, one sample after "
                                                        "each run", estimator="wall clock over the whole NVE run"))
    args.output.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.config}-{args.temperature_K:g}K"
    (args.output / f"nve-{tag}.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    with open(args.output / f"nve-{tag}.csv", "w") as handle:
        handle.write("step,conserved_eV,kinetic_eV,potential_eV\n")
        for step, conserved, kinetic in rows:
            handle.write(f"{step},{conserved:.10f},{kinetic:.10f},{conserved - kinetic:.10f}\n")
    print(f"NVE {tag}: drift {result['drift_meV_per_atom_per_ps']:.4g} +- {result['drift_se_ar1']:.2g} meV/atom/ps, "
          f"rms {result['rms_meV_per_atom']:.3g} meV/atom, T {md['mean_temperature_K']:.1f} K, rebuilds "
          f"{md['neighbour_rebuilds']}, {md['ms_per_step']:.2f} ms/step")


if __name__ == "__main__":
    main()
