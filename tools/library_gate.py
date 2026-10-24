"""The in-library reduced gate (SPEC §12.3, 1b-T16) on one structure for one label (1b-T23, acceptance item 1).

For each --ensembles entry it builds a ``Simulation`` with ``check="gate"`` (owner decision D1: the gate
is opt-in) and records ``sim.gate``: Langevin NVT (2 rattled copies, energy and forces) and Langevin-
BAOAB NPT, hydrostatic (2 rattled and 2 strained copies, stress too). The production half replays the
captured graphs of the plan's engine (graph mode), as a run would. Each ensemble runs in a fresh child
process of this tool, so the gate's references run in-process before any production engine exists
(``references ran: in-process`` in the record); the parent builds no model and imports no torch.

The acceptance criterion is ``status == "passed"`` for every label on one counting structure (plan
1b, acceptance item 1). Output: <output>/metadata.json with each case's gate record, plan, setup times
and the child's log. Exit status (``exit_status``): 0 only when every case passed; 1 when a gate ran and did
not pass; 2 when a case errored (no result, e.g. a missing checkpoint, or a non-zero child exit) or none ran.
4-T12: before, a run whose cases all errored exited 0.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ENSEMBLES = ("nvt", "npt")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, type=Path, help="Fresh directory")
    p.add_argument("--model", required=True, help="Label (its file under FASTMACE_MODELS)")
    p.add_argument("--structure", required=True,
                   help="Benchmark name (under FASTMACE_STRUCTURES) or a file with PATH@NXxNYxNZ")
    p.add_argument("--repeat", nargs=3, type=int, default=[3, 3, 3])
    p.add_argument("--ensembles", nargs="+", choices=ENSEMBLES, default=list(ENSEMBLES))
    p.add_argument("--check", choices=("gate", "allow_unavailable"), default="gate")
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=20261015)
    return p.parse_args(argv)


def case_record(ensemble, sim):
    """What one gated Simulation records: its gate, plan and setup times."""
    gate = sim.gate.to_metadata()
    return dict(ensemble=ensemble, status=gate["status"], gate=gate, plan=sim.plan.describe(full=True),
                setup_s=dict(getattr(sim, "_setup_s", {})))


def read_structure(spec, repeat):
    """(name, periodic supercell): force_stress_gate.py's rule, benchmark names under FASTMACE_STRUCTURES."""
    import importlib.util

    from ase.io import read
    location = Path(__file__).with_name("force_stress_gate.py")
    module_spec = importlib.util.spec_from_file_location("_force_stress_gate", location)
    gate = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(gate)
    name, source, repeat = gate.structure_spec(spec, tuple(repeat), os.environ.get("FASTMACE_STRUCTURES", "."))
    atoms = read(source).repeat(repeat)
    atoms.set_pbc(True)
    return name, atoms


def run_case(args, ensemble):
    """One gated Simulation in this process (a child of ``main``)."""
    from ase import units

    import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)
    from fastmace import integrators
    from fastmace.model import loading
    from fastmace.simulation import Simulation

    loaded = loading.load_pristine(args.model)  # prepares the OEq runtime before MACE is imported
    name, atoms = read_structure(args.structure, args.repeat)
    friction = 0.02 * units.fs  # fm.md's default (main.py's), per fs
    if ensemble == "nvt":
        integrator = integrators.Langevin(timestep_fs=0.5, temperature_K=args.temperature_K, friction_per_fs=friction)
    else:
        integrator = integrators.LangevinBAOAB(timestep_fs=0.5, temperature_K=args.temperature_K,
                                               friction_per_fs=friction, cell_mode="hydrostatic")
    started = time.perf_counter()
    sim = Simulation(atoms, loaded, integrator=integrator, seed=args.seed, check=args.check)
    record = case_record(ensemble, sim)
    record.update(structure=name, natoms=len(atoms), wall_s=time.perf_counter() - started)
    sim.close()
    return record


def main(argv=None, runner=subprocess.run):
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "case":
        args = parse_args(argv[2:])
        record = run_case(args, argv[1])
        print("RESULT " + json.dumps(record, default=str), flush=True)
        return record
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    meta = dict(status="running", tool="library_gate", created_utc=datetime.now(timezone.utc).isoformat(),
                host=os.uname().nodename, python=sys.executable, model=args.model, structure=args.structure,
                check=args.check, seed=args.seed, cases={})
    path = args.output / "metadata.json"

    def save():
        path.write_text(json.dumps(meta, indent=2, default=str) + "\n")

    save()
    for ensemble in args.ensembles:
        log = args.output / f"{ensemble}.log"
        child = [sys.executable, str(Path(__file__).resolve()), "case", ensemble, *map(str, argv)]
        with open(log, "w") as handle:
            done = runner(child, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        lines = [line for line in log.read_text().splitlines() if line.startswith("RESULT ")]
        case = json.loads(lines[-1][7:]) if lines else dict(ensemble=ensemble, status="error")
        case.update(exit_code=done.returncode, log=str(log))
        if done.returncode != 0:
            case["status"] = "error"
        meta["cases"][ensemble] = case
        save()
    meta["passed"] = exit_status(meta) == 0
    meta["status"] = "complete"
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    return meta


def exit_status(meta):
    """0 when every case passed; 2 when any case errored or none ran; 1 otherwise (a gate that did not pass)."""
    cases = list(meta["cases"].values())
    if not cases or any(case.get("status") == "error" for case in cases):
        return 2
    return 0 if all(case.get("status") == "passed" for case in cases) else 1


if __name__ == "__main__":
    result = main()
    sys.exit(exit_status(result) if "cases" in result else 0)
