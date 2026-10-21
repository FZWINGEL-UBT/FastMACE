"""Hybrid MH-1 in subprocess-runner children against a fresh process, statistically (2-T23, phase 2 item 4).

2-T07 showed the runner's plain-cuEq float64 case within 1e-12 Å and left the hybrid's statistical
comparison to 2-T23, with 1b-T14's measure. ``run`` builds, per seed, one ``fm.Simulation`` on MH-1
NaCl 3x3x3, hybrid float32, Langevin 2 fs, with thermo rows every --every steps, and runs the target
ramp 300 → 600 K over --steps steps in two ``run()`` calls (one hand-off at half time, as 1b-T14's
interrupted run):

- ``--case runner``: the process first runs a plain-cuEq ``Simulation`` (its compile fixes the
  process's backend family), so the hybrid plan runs in runner children (``plan.runner ==
  "subprocess"``);
- ``--case fresh``: the hybrid plan runs in this process.

``compare`` applies 1b-T14's rule per seed: the runner's mean temperature, mean potential energy and
50 K-bin energies differ from the fresh run's by no more than max(2 x the fresh runs' difference
between seeds, floor). With more than two seeds the difference between seeds is the median over every
pair of fresh seeds (with two seeds, their one difference, as 1b-T14).

The first comparison (20 October 2026, from the 0 K crystal) was dominated by the heating transient of
its first ~1,000 steps. The owner's decision of the same day: repeat from an equilibrated state with
four seeds. ``equilibrate`` makes that state once (in-process hybrid, from the 0 K crystal at 300 K:
first at 0.5 fs, the production step, then at 2 fs), and ``run --start`` begins both cases from it.

    python tools/runner_statistics.py equilibrate --output DIR
    python tools/runner_statistics.py run --case runner --output DIR [--start DIR/start.extxyz --seeds ...]
    python tools/runner_statistics.py compare --fresh DIR_F --runner DIR_R
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics as statistics_module
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

# isort: split
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_step_time  # noqa: E402
import trajectory_versions  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--case", required=True, choices=("runner", "fresh"))
    run.add_argument("--output", required=True, type=Path, help="Fresh directory")
    run.add_argument("--model", default="mh1-r2scan")
    run.add_argument("--structure", default="NaCl")
    run.add_argument("--steps", type=int, default=20_000)
    run.add_argument("--every", type=int, default=20)
    run.add_argument("--seeds", type=int, nargs="+", default=[20261001, 20261101])
    run.add_argument("--targets", type=float, nargs=2, default=[300.0, 600.0])
    run.add_argument("--timestep-fs", type=float, default=2.0)
    run.add_argument("--start", type=Path, help="Start state (extxyz with momenta, from equilibrate)")
    equilibrate = sub.add_parser("equilibrate")
    equilibrate.add_argument("--output", required=True, type=Path, help="Fresh directory")
    equilibrate.add_argument("--model", default="mh1-r2scan")
    equilibrate.add_argument("--structure", default="NaCl")
    equilibrate.add_argument("--temperature-K", type=float, default=300.0)
    equilibrate.add_argument("--seed", type=int, default=20261020)
    # (timestep fs, steps): 2 ps at the production step from 0 K, then 10 ps at the comparison's step.
    equilibrate.add_argument("--stages", type=float, nargs="+", default=[0.5, 4000, 2.0, 5000],
                             help="Pairs of timestep (fs) and steps")
    compare = sub.add_parser("compare")
    compare.add_argument("--fresh", required=True, type=Path)
    compare.add_argument("--runner", required=True, type=Path)
    args = p.parse_args(argv)
    if args.command == "run" and args.steps % 2:
        p.error("--steps must be even (two run() calls)")
    if args.command == "equilibrate" and len(args.stages) % 2:
        p.error("--stages takes pairs of timestep (fs) and steps")
    return args


def read_start(path):
    """The start state: positions, momenta and cell of an extxyz frame (ase.io.read keeps the momenta)."""
    from ase.io import read
    atoms = read(path)
    if not atoms.has("momenta"):
        raise ValueError(f"{path}: no momenta")
    return atoms


def equilibrate(args):
    """One in-process hybrid run at constant target from the 0 K crystal, in stages of (timestep, steps);
    writes start.extxyz (positions and momenta of the last committed step) and each stage's thermo rows."""
    from ase import units
    from ase.io import write

    import fastmace as fm
    from fastmace.config import EngineConfig
    from fastmace.planner.workload import Workload
    args.output.mkdir(parents=True, exist_ok=False)
    meta = dict(tool="runner_statistics equilibrate", status="running", host=os.uname().nodename,
                created_utc=datetime.now(timezone.utc).isoformat(), fastmace_commit=ab_step_time.fastmace_commit(),
                model=args.model, structure=args.structure, temperature_K=args.temperature_K, seed=args.seed,
                stages=[], start=None)
    path = args.output / "equilibrate.json"
    try:
        _, atoms = ab_step_time.read_structure(args.structure, None)
        model = fm.load(args.model)
        workload = Workload(max_temperature_K=args.temperature_K)
        for index in range(0, len(args.stages), 2):
            timestep, steps = float(args.stages[index]), int(args.stages[index + 1])
            integrator = fm.integrators.Langevin(timestep_fs=timestep, temperature_K=args.temperature_K,
                                                 friction_per_fs=0.02 * units.fs)
            ff = fm.ForceField(model, atoms, precision="float32")
            directory = args.output / f"stage-{index // 2}"
            sim = fm.Simulation(atoms.copy(), ff, integrator=integrator, seed=args.seed + index, workload=workload,
                                output=fm.Output(directory, thermo_every=20, trajectory_every=None,
                                                 restart_every=None), engine=EngineConfig(backend="hybrid"))
            sim.run(steps=steps)
            committed = sim.committed()
            rows = thermo_rows(directory)
            tail = rows[len(rows) // 2:]
            meta["stages"].append(dict(timestep_fs=timestep, steps=steps, runner=sim.plan.runner,
                                       backend=sim.plan.engine.backend,
                                       second_half_temperature_K=sum(r["temperature"] for r in tail) / len(tail),
                                       second_half_energy_eV_atom=sum(r["energy"] for r in tail) / len(tail)))
            sim.close()
            atoms = atoms.copy()
            atoms.set_positions(committed.positions)
            atoms.set_cell(committed.cell)
            atoms.set_momenta(committed.momenta)
        write(args.output / "start.extxyz", atoms)
        meta.update(start=str(args.output / "start.extxyz"), status="complete")
    except Exception as exc:
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(meta, indent=1, default=str) + "\n")
    return meta


def thermo_rows(directory):
    """1b-T14's rows (target, temperature, energy per atom) from a Simulation's thermo chunks."""
    from fastmace.io import output as output_io
    header, rows = output_io.read_thermo(directory)
    index = {name: header.index(name) for name in ("step", "target_K", "temperature_K", "potential_eV_atom")}
    return [{"step": int(row[index["step"]]), "target": float(row[index["target_K"]]),
             "temperature": float(row[index["temperature_K"]]), "energy": float(row[index["potential_eV_atom"]])}
            for row in rows if int(row[index["step"]]) > 0]


def run(args):
    from ase import units

    import fastmace as fm
    from fastmace import protocols
    from fastmace.config import EngineConfig
    from fastmace.planner.workload import Workload
    args.output.mkdir(parents=True, exist_ok=False)
    meta = dict(tool="runner_statistics", status="running", case=args.case,
                created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename, python=sys.executable,
                fastmace_commit=ab_step_time.fastmace_commit(), model=args.model, structure=args.structure,
                steps=args.steps, every=args.every, seeds=list(args.seeds), targets=list(args.targets),
                timestep_fs=args.timestep_fs, start=str(args.start) if args.start else None, runs={})
    path = args.output / "statistics.json"

    def save():
        path.write_text(json.dumps(meta, indent=1, default=str) + "\n")

    save()
    try:
        if args.start is not None:
            template = read_start(args.start)
        else:
            _, template = ab_step_time.read_structure(args.structure, None)
        model = fm.load(args.model)
        integrator = fm.integrators.Langevin(timestep_fs=args.timestep_fs, temperature_K=args.targets[0],
                                             friction_per_fs=0.02 * units.fs)
        workload = Workload(max_temperature_K=max(args.targets))
        if args.case == "runner":  # a compiled plain cuEq fixes the family; the hybrid then needs children
            ff = fm.ForceField(model, template, precision="float64")
            first = fm.Simulation(template.copy(), ff, integrator=integrator, seed=1, workload=workload,
                                  engine=EngineConfig(backend="cueq"))
            first.run(steps=20)
            meta["first"] = dict(backend=first.plan.engine.backend, runner=first.plan.runner)
            first.close()
        half = args.steps // 2
        ps = args.steps * args.timestep_fs / 1000.0
        for seed in args.seeds:
            directory = args.output / f"seed-{seed}"
            ff = fm.ForceField(model, template, precision="float32")
            output = fm.Output(directory, thermo_every=args.every, trajectory_every=None, restart_every=None)
            started = time.perf_counter()
            # The hybrid is asked for explicitly, as in 2-T07's test: with backend "auto" the planner keeps the
            # process's plain-cuEq family in-process, and no child would run.
            sim = fm.Simulation(template.copy(), ff, integrator=integrator, seed=seed, workload=workload,
                                output=output, engine=EngineConfig(backend="hybrid"))
            setup_s = time.perf_counter() - started
            started = time.perf_counter()
            sim.run(steps=half, schedule=protocols.ramp(args.targets[0], args.targets[1], ps=ps))
            sim.run(steps=half)
            run_s = time.perf_counter() - started
            metadata = sim.metadata()
            meta["runs"][str(seed)] = dict(backend=sim.plan.engine.backend, dtype=sim.plan.precision.dtype,
                                           runner=sim.plan.runner, setup_s=setup_s, run_s=run_s,
                                           subprocess=metadata.get("subprocess"), report=sim.report.text())
            sim.close()
            save()
        import torch
        meta.update(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, status="complete")
    except Exception as exc:
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    return meta


def compare(fresh, runner):
    metas = {}
    for name, directory in (("fresh", fresh), ("runner", runner)):
        meta = json.loads((Path(directory) / "statistics.json").read_text())
        if meta.get("status") != "complete":
            raise ValueError(f"{directory}: run status {meta.get('status')!r}")
        metas[name] = meta
    seeds = [str(seed) for seed in metas["fresh"]["seeds"]]
    if [str(seed) for seed in metas["runner"]["seeds"]] != seeds or len(seeds) < 2:
        raise ValueError("both runs need the same seeds, at least two")
    if metas["fresh"].get("start") != metas["runner"].get("start"):
        raise ValueError("both runs need the same start state")
    stats = {name: {seed: trajectory_versions.statistics(thermo_rows(Path(directory) / f"seed-{seed}"))
                    for seed in seeds} for name, directory in (("fresh", fresh), ("runner", runner))}
    pairs = [trajectory_versions.differences(stats["fresh"][a], stats["fresh"][b])
             for a, b in itertools.combinations(seeds, 2)]
    between = {key: statistics_module.median(pair[key] for pair in pairs) for key in pairs[0]}
    across = {seed: trajectory_versions.differences(stats["runner"][seed], stats["fresh"][seed]) for seed in seeds}
    limits = {key: max(2.0 * between[key], floor) for key, floor in trajectory_versions.STATISTIC_FLOORS.items()}
    runners = {name: {seed: metas[name]["runs"][seed]["runner"] for seed in seeds} for name in metas}
    backends = {name: {seed: metas[name]["runs"][seed].get("backend") for seed in seeds} for name in metas}
    return dict(runners=runners, backends=backends, between_seeds=between, runner_vs_fresh=across, limits=limits,
                passed=(all(value == "hybrid" for side in backends.values() for value in side.values())
                        and all(value == "subprocess" for value in runners["runner"].values())
                        and all(value == "in-process" for value in runners["fresh"].values())
                        and all(across[seed][key] <= limits[key] for seed in seeds for key in limits)))


def main(argv=None):
    args = parse_args(argv)
    if args.command == "run":
        return run(args)
    if args.command == "equilibrate":
        return equilibrate(args)
    result = compare(args.fresh, args.runner)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
