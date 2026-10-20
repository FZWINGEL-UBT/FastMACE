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
between the two seeds, floor).

    python tools/runner_statistics.py run --case runner --output DIR
    python tools/runner_statistics.py compare --fresh DIR_F --runner DIR_R
"""

from __future__ import annotations

import argparse
import json
import os
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
    compare = sub.add_parser("compare")
    compare.add_argument("--fresh", required=True, type=Path)
    compare.add_argument("--runner", required=True, type=Path)
    args = p.parse_args(argv)
    if args.command == "run" and args.steps % 2:
        p.error("--steps must be even (two run() calls)")
    return args


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
                timestep_fs=args.timestep_fs, runs={})
    path = args.output / "statistics.json"

    def save():
        path.write_text(json.dumps(meta, indent=1, default=str) + "\n")

    save()
    try:
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
    if [str(seed) for seed in metas["runner"]["seeds"]] != seeds or len(seeds) != 2:
        raise ValueError("both runs need the same two seeds")
    stats = {name: {seed: trajectory_versions.statistics(thermo_rows(Path(directory) / f"seed-{seed}"))
                    for seed in seeds} for name, directory in (("fresh", fresh), ("runner", runner))}
    between = trajectory_versions.differences(stats["fresh"][seeds[0]], stats["fresh"][seeds[1]])
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
    result = compare(args.fresh, args.runner)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
