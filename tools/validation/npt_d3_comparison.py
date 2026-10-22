"""3b-T05: GPU NPT + D3 against an independent implementation (SPEC §11.2; conductor's choice 7).

The independent side is ASE's ``LangevinBAOAB`` with MACE's own ``MACECalculator`` (cuEquivariance,
float32, MH-1 ``matpes_r2scan``) plus ``torch-dftd`` (float64) on exactly ``fm.D3("r2scan")``'s numbers
(``fastmace.reference.d3.torch_dftd3_calculator``), summed by ASE's ``SumCalculator``. No FastMACE code is in
its force path or its integrator. The FastMACE side (``run-fastmace``) is ``fm.Simulation`` on the default
planner's (production) plan of ``fm.ForceField(MH-1, precision="float32", dispersion=fm.D3("r2scan"))`` with
``fm.integrators.LangevinBAOAB`` on the same settings; its samples come from an output sink and go through the
same arithmetic as the ASE side's.

Settings (``SETTINGS``, recorded with every run; the same on both engines): NbCl5 1x1x3 (216 atoms) from
the r2SCAN source cell, hydrostatic, 0 GPa, 1 fs, friction 0.02 per ASE time unit (T_tau about 0.49 ps),
P_tau 2 ps (a test setting: the library default is 20 T_tau), barostat mass factor 1, the cell
thermostat on. Each run starts from the source cell with Maxwell-Boltzmann velocities at T, runs
``--equilibration-ps`` that the analysis discards, then ``--production-ps``; a sample every
``--sample-every`` steps: temperature, potential and kinetic energy, volume, the instantaneous pressure
with its kinetic part, (2 KE - V Tr sigma) / 3V (the research engine's ``npt_observables``), and the cell.

``analyse`` (the pass rule of 3b-T10's runbook, item 2): per temperature and engine, the seeds' production
samples are cut into blocks of at least twice the integrated autocorrelation time of the volume (the
slowest observable, pooled over runs and engines); the standard error of each mean is that of the block
means pooled over the seeds, and that of the volume fluctuation (the standard deviation of V) a jackknife
over the same blocks. An observable passes when |difference| <= 3 combined standard errors. The drift of
V (slope of a line through each run's production samples) is reported, so a non-stationary 600 K run is
recorded as such.

    python tools/validation/npt_d3_comparison.py run-ase --temperature-K 300 --seed 101 --output OUT/ase-300K-s101
    python tools/validation/npt_d3_comparison.py analyse OUT/* --output OUT/comparison.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SETTINGS = dict(structure="NbCl5", repeat=(1, 1, 3), cell_mode="hydrostatic", pressure_GPa=0.0, timestep_fs=1.0,
                friction_per_ase_time=0.02, barostat_tau_fs=2000.0, barostat_mass_factor=1.0, cell_langevin=True,
                head="matpes_r2scan", d3_functional="r2scan")
COLUMNS = ("step", "time_ps", "temperature_K", "potential_eV_atom", "kinetic_eV_atom", "volume_A3_atom",
           "pressure_GPa", "a_A", "b_A", "c_A")
OBSERVABLES = ("volume_A3_atom", "pressure_GPa", "potential_eV_atom")
SIGMAS = 3.0  # the pass rule: |difference| <= 3 combined standard errors
MIN_BLOCK_TAU = 2.0  # blocks of at least twice the integrated autocorrelation time


# --------------------------------------------------------------------------- observables


def npt_observables(natoms, volume, kinetic, stress):
    """Volume per atom (Å³) and instantaneous pressure (GPa, with the kinetic part) of an NPT state.

    The research engine's ``workflow.npt_observables``; ``stress`` is the potential stress (ASE sign)."""
    from ase import units

    pressure = (2.0 * kinetic - volume * float(np.trace(np.asarray(stress)))) / (3.0 * volume)
    return volume / natoms, pressure / units.GPa


def observables_row(step, timestep_fs, natoms, cell, kinetic, potential, stress):
    """One sample: ``cell`` a 3x3 array (Å), ``kinetic`` and ``potential`` in eV, ``stress`` potential (ASE sign).

    The temperature counts 3N degrees of freedom, as ASE's ``get_temperature`` without constraints."""
    from ase import units

    cell = np.asarray(cell, dtype=float)
    volume = abs(float(np.linalg.det(cell)))
    volume_atom, pressure = npt_observables(natoms, volume, kinetic, stress)
    a, b, c = np.linalg.norm(cell, axis=1)
    return (step, step * timestep_fs / 1000.0, 2.0 * kinetic / (3.0 * natoms * units.kB), potential / natoms,
            kinetic / natoms, volume_atom, pressure, a, b, c)


def sample_row(step, timestep_fs, atoms, potential, stress):
    return observables_row(step, timestep_fs, len(atoms), atoms.cell.array, atoms.get_kinetic_energy(), potential,
                           stress)


# --------------------------------------------------------------------------- the ASE side


def source_atoms(structures_root, settings=SETTINGS):
    """The r2SCAN source cell of the baseline export (``FASTMACE_STRUCTURES``), repeated."""
    from ase.io import read

    path = Path(structures_root) / "Structures" / "Benchmark" / "R2SCAN" / f"{settings['structure']}.cif"
    return read(path).repeat(settings["repeat"])


def d3_record():
    """``fm.D3`` of the comparison: the r2SCAN Becke-Johnson set, every other field the record's default."""
    from fastmace.longrange.d3 import D3

    return D3(functional=SETTINGS["d3_functional"])


def ase_calculator(checkpoint, device="cuda"):
    """MACE's ``MACECalculator`` (cuEq float32, the head) plus torch-dftd (float64) on ``fm.D3``'s numbers."""
    from ase.calculators.mixing import SumCalculator
    from mace.calculators import MACECalculator

    from fastmace.reference.d3 import torch_dftd3_calculator

    mace = MACECalculator(model_paths=str(checkpoint), device=device, default_dtype="float32",
                          head=SETTINGS["head"], enable_cueq=True)
    d3 = torch_dftd3_calculator(d3_record(), dtype="float64", device=device)
    return SumCalculator([mace, d3])


def run_ase(atoms, calculator, temperature_K, seed, steps, sample_every, sink, settings=SETTINGS):
    """ASE ``LangevinBAOAB`` from ``atoms`` (velocities drawn here) for ``steps``; ``sink(row)`` per sample."""
    from ase import units
    from ase.md.langevinbaoab import LangevinBAOAB
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    if settings["cell_mode"] != "hydrostatic" or not settings["cell_langevin"]:
        raise ValueError("the comparison's settings are hydrostatic with the cell thermostat on")
    atoms.calc = calculator
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
        dyn = LangevinBAOAB(atoms, timestep=settings["timestep_fs"] * units.fs, temperature_K=temperature_K,
                            externalstress=-settings["pressure_GPa"] * units.GPa, hydrostatic=True,
                            T_tau=1.0 / settings["friction_per_ase_time"],
                            P_tau=settings["barostat_tau_fs"] * units.fs,
                            P_mass_factor=settings["barostat_mass_factor"], rng=rng)
    for step in range(steps + 1):
        if step:
            dyn.run(1)
        if step % sample_every == 0:
            sink(sample_row(step, settings["timestep_fs"], atoms, atoms.get_potential_energy(),
                            atoms.get_stress(voigt=False)))
    return atoms


def fastmace_side(checkpoint, atoms, temperature_K, seed, steps, sample_every, sink, settings=SETTINGS):
    """``fm.Simulation`` (default planner) from ``atoms`` (velocities drawn here, as ``run_ase``); returns it."""
    from ase import units
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    import fastmace as fm

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=np.random.default_rng(seed))
    model = fm.load(str(checkpoint), head=settings["head"], trusted=True)
    force_field = fm.ForceField(model, atoms, precision="float32", dispersion=d3_record())
    integrator = fm.integrators.LangevinBAOAB(
        timestep_fs=settings["timestep_fs"], temperature_K=temperature_K,
        friction_per_fs=settings["friction_per_ase_time"] * units.fs, pressure_GPa=settings["pressure_GPa"],
        cell_mode=settings["cell_mode"], barostat_tau_fs=settings["barostat_tau_fs"],
        barostat_mass_factor=settings["barostat_mass_factor"],
        disable_cell_langevin=not settings["cell_langevin"])
    simulation = fm.Simulation(atoms, force_field, integrator=integrator, seed=seed,
                               workload=fm.Workload(max_temperature_K=temperature_K))
    natoms = len(atoms)

    def frame(step, values):
        host = {key: np.asarray(value, dtype=float) for key, value in values.items()
                if key in ("cell", "kinetic", "energy", "stress")}
        sink(observables_row(step, settings["timestep_fs"], natoms, host["cell"], float(host["kinetic"].sum()),
                             float(host["energy"].reshape(-1)[0]), host["stress"]))

    simulation.submit_frame(frame)
    simulation.add_sink(frame, sample_every)
    simulation.run(steps)
    simulation.flush_output()
    return simulation


def _versions():
    import importlib.metadata as metadata

    versions = {}
    for package in ("ase", "torch", "mace-torch", "torch-dftd", "cuequivariance", "cuequivariance-torch",
                    "openequivariance", "numpy"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def command_run(args):
    import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

    # isort: split
    import torch

    args.output.mkdir(parents=True, exist_ok=True)
    root = args.structures or os.environ.get("FASTMACE_STRUCTURES", ".")
    atoms = source_atoms(root)
    steps = round((args.equilibration_ps + args.production_ps) * 1000.0 / SETTINGS["timestep_fs"])
    record = dict(engine=args.engine, temperature_K=args.temperature_K, seed=args.seed, steps=steps,
                  equilibration_ps=args.equilibration_ps, production_ps=args.production_ps,
                  sample_every=args.sample_every, settings=SETTINGS, natoms=len(atoms),
                  checkpoint=str(args.checkpoint), machine=args.machine or platform.node(),
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  python=sys.executable, versions=_versions(),
                  fastmace_commit=os.environ.get("FASTMACE_SOURCE_COMMIT"),
                  created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    calculator = ase_calculator(args.checkpoint, args.device) if args.engine == "ase" else None
    started = time.perf_counter()
    steady = {}  # the first sample at or after 10 % of the steps: the rate without start-up and first calls
    with open(args.output / "samples.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)

        def sink(row):
            writer.writerow(row)
            handle.flush()
            if not steady and row[0] >= steps // 10 and row[0] > 0:
                steady.update(step=row[0], time=time.perf_counter())

        if args.engine == "ase":
            final = run_ase(atoms, calculator, args.temperature_K, args.seed, steps, args.sample_every, sink)
        else:
            simulation = fastmace_side(args.checkpoint, atoms, args.temperature_K, args.seed, steps,
                                       args.sample_every, sink)
            record["simulation"] = json.loads(json.dumps(simulation.metadata(), default=str))
    record["wall_s"] = time.perf_counter() - started
    record["s_per_step"] = record["wall_s"] / max(steps, 1)
    if steady and steps > steady["step"]:
        record["steady_s_per_step"] = (started + record["wall_s"] - steady["time"]) / (steps - steady["step"])
    if args.engine == "ase":
        from ase.io import write

        write(args.output / "final.extxyz", final)
    (args.output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    steady_ms = record.get("steady_s_per_step", math.nan) * 1e3
    print(f"{args.engine} {args.temperature_K:g} K seed {args.seed}: {steps} steps, "
          f"{record['s_per_step'] * 1e3:.1f} ms/step, {steady_ms:.1f} after the first 10 %")
    return 0


# --------------------------------------------------------------------------- analysis


def integrated_time(series, c=5.0):
    """Integrated autocorrelation time in samples (Sokal's automatic window, ``c`` times tau)."""
    x = np.asarray(series, dtype=float) - np.mean(series)
    n = len(x)
    if n < 4 or not np.any(x):
        return 1.0
    spectrum = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(spectrum * np.conjugate(spectrum))[:n]
    acf = acf / acf[0]
    tau = 1.0
    for window in range(1, n):
        tau = 1.0 + 2.0 * acf[1:window + 1].sum()
        if window >= c * tau:
            break
    return max(tau, 1.0)


def blocks_of(series, length):
    """Consecutive blocks of ``length`` samples (a trailing partial block is dropped)."""
    count = len(series) // length
    return [np.asarray(series[i * length:(i + 1) * length], dtype=float) for i in range(count)]


def jackknife_std(blocks):
    """The standard deviation of all samples and its jackknife standard error over the blocks."""
    whole = np.concatenate(blocks)
    estimates = np.array([np.concatenate(blocks[:i] + blocks[i + 1:]).std(ddof=1) for i in range(len(blocks))])
    n = len(blocks)
    return float(whole.std(ddof=1)), float(math.sqrt((n - 1) / n * ((estimates - estimates.mean()) ** 2).sum()))


def block_mean(blocks):
    """The mean of all samples and the standard error of the block means."""
    means = np.array([block.mean() for block in blocks])
    return float(np.concatenate(blocks).mean()), float(means.std(ddof=1) / math.sqrt(len(means)))


def drift_per_ps(time_ps, values):
    slope, _ = np.polyfit(np.asarray(time_ps, dtype=float), np.asarray(values, dtype=float), 1)
    return float(slope)


def read_run(directory):
    directory = Path(directory)
    record = json.loads((directory / "run.json").read_text())
    with open(directory / "samples.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    data = {name: np.array([float(row[name]) for row in rows]) for name in COLUMNS}
    keep = data["time_ps"] > record["equilibration_ps"] + 1e-9
    return record, {name: values[keep] for name, values in data.items()}


def analyse(runs, sigmas=SIGMAS, min_block_tau=MIN_BLOCK_TAU):
    """``runs``: (record, production samples) pairs. The comparison per temperature, as a dict."""
    by_temperature = {}
    for record, data in runs:
        by_temperature.setdefault(float(record["temperature_K"]), {}).setdefault(record["engine"], []).append(
            (record, data))
    result = {}
    for temperature, engines in sorted(by_temperature.items()):
        taus = [integrated_time(data["volume_A3_atom"]) for runs_ in engines.values() for _, data in runs_]
        shortest = min(len(data["volume_A3_atom"]) for runs_ in engines.values() for _, data in runs_)
        length = max(1, math.ceil(min_block_tau * max(taus)))
        entry = {"block_samples": length, "tau_volume_samples": max(taus), "engines": {}}
        if shortest // length < 2:
            entry["error"] = f"too few blocks: {shortest} samples per run, blocks of {length}"
            result[f"{temperature:g}K"] = entry
            continue
        for engine, runs_ in sorted(engines.items()):
            stats = {"seeds": sorted(int(record["seed"]) for record, _ in runs_), "blocks": 0}
            for name in OBSERVABLES:
                blocks = [block for _, data in runs_ for block in blocks_of(data[name], length)]
                stats[name] = dict(zip(("mean", "se"), block_mean(blocks), strict=True))
                stats["blocks"] = len(blocks)
            blocks = [block for _, data in runs_ for block in blocks_of(data["volume_A3_atom"], length)]
            stats["volume_std"] = dict(zip(("mean", "se"), jackknife_std(blocks), strict=True))
            stats["temperature_K"] = float(np.mean([data["temperature_K"].mean() for _, data in runs_]))
            stats["volume_drift_A3_atom_per_ps"] = {
                int(record["seed"]): drift_per_ps(data["time_ps"], data["volume_A3_atom"]) for record, data in runs_}
            entry["engines"][engine] = stats
        if len(entry["engines"]) == 2:
            first, second = (entry["engines"][name] for name in sorted(entry["engines"]))
            comparisons = {}
            for name in (*OBSERVABLES, "volume_std"):
                difference = second[name]["mean"] - first[name]["mean"]
                combined = math.hypot(first[name]["se"], second[name]["se"])
                comparisons[name] = dict(difference=difference, combined_se=combined,
                                         sigmas=abs(difference) / combined if combined else math.inf,
                                         passed=abs(difference) <= sigmas * combined)
            entry["comparison"] = comparisons
            entry["passed"] = all(item["passed"] for item in comparisons.values())
        result[f"{temperature:g}K"] = entry
    return result


def command_analyse(args):
    runs = [read_run(directory) for directory in args.runs]
    result = dict(rule=f"|difference| <= {SIGMAS:g} combined SE; blocks >= {MIN_BLOCK_TAU:g} tau(V), pooled over "
                       "seeds; V fluctuation by jackknife over the same blocks", settings=SETTINGS,
                  runs=[str(directory) for directory in args.runs], temperatures=analyse(runs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for temperature, entry in result["temperatures"].items():
        print(temperature, "passed" if entry.get("passed") else entry.get("error", "failed or one engine only"))
        for name, item in entry.get("comparison", {}).items():
            print(f"  {name}: difference {item['difference']:+.4g}, {item['sigmas']:.2f} combined SE")
    return 0 if all(entry.get("passed") for entry in result["temperatures"].values()) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, engine, text in (("run-ase", "ase", "one ASE LangevinBAOAB + MACE + torch-dftd run"),
                               ("run-fastmace", "fastmace", "one fm.Simulation run on the production plan")):
        run = commands.add_parser(name, help=text)
        _run_arguments(run)
        run.set_defaults(handler=command_run, engine=engine)
    compare = commands.add_parser("analyse", help="the comparison of run directories")
    compare.add_argument("runs", nargs="+", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    compare.set_defaults(handler=command_analyse)
    args = parser.parse_args(argv)
    return args.handler(args)


def _run_arguments(run):
    run.add_argument("--temperature-K", type=float, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path,
                     default=Path(os.environ.get("FASTMACE_MODELS", "~/models")).expanduser() / "mace-mh-1.model")
    run.add_argument("--structures", help="the baseline export (default FASTMACE_STRUCTURES)")
    run.add_argument("--equilibration-ps", type=float, default=5.0)
    run.add_argument("--production-ps", type=float, default=20.0)
    run.add_argument("--sample-every", type=int, default=10)
    run.add_argument("--device", default="cuda")
    run.add_argument("--machine", help="the machine's name for the record (e.g. 'Modal L4')")


if __name__ == "__main__":
    sys.exit(main())
