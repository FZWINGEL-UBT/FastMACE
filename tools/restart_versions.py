"""A restart written by an earlier release, resumed by this checkout (4-T13, phase 4 item 5: restart format 4
still loads format 3; 5-T11, phase 5 item 4: a format-4 restart of the phase 4 release resumes here).

One stage per process, CPU float64 by default (the CPU tests' planner: ``device="cpu"``, ``compile_mode="none"``),
``fm.Simulation`` with Langevin at ``--temperature-K`` from seeded Maxwell-Boltzmann momenta:

- ``run --stage uninterrupted`` (the release): ``--steps`` steps in one go, the final state saved;
- ``run --stage write`` (the release): ``--restart-step`` steps, then ``save_restart(<output>/restart)``;
- ``run --stage resume`` (this checkout): ``Simulation.resume`` of that restart, then the remaining steps. A
  plan change (phase 4's new plan fields) is retried with ``allow_plan_change=True`` and recorded;
- ``compare``: the resumed final state against the uninterrupted one, bitwise (step, positions, momenta, cell).

The release stages import the release as the plain package ``fastmace`` from ``--release-root``, a ``git
archive`` of its commit with ``.source_commit``: a restart records the package that wrote it, so the renamed
copies of ``fastmace.bench.versions`` (``fastmace_3b``) cannot stand in for the release here.

    git -C <repository> archive ff80ca2 | tar -x -C <root>
    git -C <repository> rev-parse ff80ca2 > <root>/.source_commit
    python tools/restart_versions.py run --stage write --release-root <root> --expected-commit ff80ca2 \\
        --model <tiny.model> --checkpoint <tiny.model> --output <dir>/write
    ... run --stage uninterrupted ... --output <dir>/uninterrupted
    python tools/restart_versions.py run --stage resume --source <dir>/write --model ... --output <dir>/resume
    python tools/restart_versions.py compare <dir>/uninterrupted <dir>/resume
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
STAGES = ("uninterrupted", "write", "resume")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="One stage in this process")
    run.add_argument("--stage", choices=STAGES, required=True)
    run.add_argument("--output", required=True, type=Path, help="Fresh directory")
    run.add_argument("--release-root", type=Path,
                     help="A git archive of the release with .source_commit (write, uninterrupted)")
    run.add_argument("--expected-commit", default=None, help="Commit (prefix) the release root's .source_commit has")
    run.add_argument("--source", type=Path, help="The write stage's output (resume)")
    run.add_argument("--model", required=True, help="Label or checkpoint path (the tests' tiny model)")
    run.add_argument("--checkpoint", required=True, type=Path)
    run.add_argument("--structure", default="NaCl@2x2x2")
    run.add_argument("--steps", type=int, default=100)
    run.add_argument("--restart-step", type=int, default=50)
    run.add_argument("--temperature-K", type=float, default=300.0)
    run.add_argument("--seed", type=int, default=20261023)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    compare = sub.add_parser("compare", help="The resumed final state against the uninterrupted one")
    compare.add_argument("uninterrupted", type=Path)
    compare.add_argument("resumed", type=Path)
    args = p.parse_args(argv)
    if args.command == "run":
        if args.stage == "resume" and args.source is None:
            p.error("--stage resume needs --source (the write stage's output)")
        if args.stage != "resume" and (args.release_root is None or args.expected_commit is None):
            p.error(f"--stage {args.stage} runs the release: give --release-root and --expected-commit")
        if not 0 < args.restart_step < args.steps:
            p.error("--restart-step must lie inside the run")
    return args


def release_source(root, expected_commit):
    """The release root's commit, checked against ``expected_commit``."""
    source = (Path(root) / ".source_commit").read_text().strip()
    if not source.startswith(expected_commit):
        raise SystemExit(f"{root} is {source}, not {expected_commit}")
    return source


def _state(sim):
    committed = sim.committed()
    return {"step": int(committed.step), "positions": committed.positions.tobytes().hex(),
            "momenta": committed.momenta.tobytes().hex(), "cell": committed.cell.tobytes().hex(),
            "energy": float(committed.energy)}


def run(args):
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    source = root = None
    if args.stage != "resume":  # the release, as the package it was, before anything imports fastmace
        root = args.release_root.expanduser().resolve()
        source = release_source(root, args.expected_commit)
        sys.path.insert(0, str(root))
    import fastmace

    if root is not None and Path(fastmace.__file__).resolve().parent != root / "fastmace":
        raise SystemExit(f"fastmace was imported from {fastmace.__file__}, not from {root}")
    sys.path.append(str(TOOLS))
    import ab_step_time

    from fastmace import simulation as simulations

    args.output.mkdir(parents=True, exist_ok=False)
    record = {"tool": "restart_versions", "stage": args.stage, "fastmace": str(Path(fastmace.__file__).parent),
              "release_source": source, "created_utc": datetime.now(timezone.utc).isoformat(),
              "fastmace_commit": source or ab_step_time.fastmace_commit(), "steps": args.steps,
              "restart_step": args.restart_step, "seed": args.seed, "device": args.device}
    if args.stage == "resume":
        try:
            sim = simulations.Simulation.resume(args.source / "restart")
            record["plan_change"] = False
        except Exception as error:  # noqa: BLE001  (recorded: which refusal a format-3 restart meets)
            record["plan_change"] = f"{type(error).__name__}: {error}"
            sim = simulations.Simulation.resume(args.source / "restart", allow_plan_change=True)
        sim.run(steps=args.steps - int(sim.committed().step))
    else:
        from fastmace import config, integrators
        from fastmace.model import loading
        from fastmace.model.forcefield import ForceField
        from fastmace.planner.workload import Workload

        _, atoms = ab_step_time.read_structure(args.structure, None)
        spec = importlib.util.spec_from_file_location("_ab_frontends", TOOLS / "ab_frontends.py")
        frontends = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(frontends)
        frontends.thermalize(atoms, args.temperature_K, args.seed)
        checkpoint = args.checkpoint.expanduser().resolve(strict=True)
        sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        loaded = loading.load_pristine(args.model, checkpoint=str(checkpoint), sha256=sha256)
        engine = config.EngineConfig(device="cpu", compile_mode="none") if args.device == "cpu" else None
        sim = simulations.Simulation(
            atoms, ForceField(loaded, atoms, precision="float64"),
            integrator=integrators.Langevin(timestep_fs=1.0, temperature_K=args.temperature_K, friction_per_fs=0.01),
            seed=args.seed, workload=Workload(max_temperature_K=args.temperature_K),
            **({} if engine is None else {"engine": engine}))
        if args.stage == "write":
            sim.run(steps=args.restart_step)
            sim.save_restart(args.output / "restart")
        else:
            sim.run(steps=args.steps)
    record.update(final=_state(sim), plan=sim.plan.describe())
    sim.close()
    (args.output / "restart_versions.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"{args.stage} ({record['fastmace']}): step {record['final']['step']}")
    return record


def compare(uninterrupted, resumed):
    a = json.loads((uninterrupted / "restart_versions.json").read_text())["final"]
    b = json.loads((resumed / "restart_versions.json").read_text())["final"]
    keys = ("step", "positions", "momenta", "cell")
    return {"bitwise": all(a[key] == b[key] for key in keys), "differs": [key for key in keys if a[key] != b[key]],
            "energy": [a["energy"], b["energy"]]}


def main(argv=None):
    args = parse_args(argv)
    if args.command == "run":
        run(args)
        return 0
    result = compare(args.uninterrupted, args.resumed)
    print(json.dumps(result, indent=2))
    return 0 if result["bitwise"] else 1


if __name__ == "__main__":
    sys.exit(main())
