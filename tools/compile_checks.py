"""Two compile measurements left to 1b-T23 by earlier tasks (plan 1b, "After T14" and "After T04, T13, T18").

``polar-npt``: compiled PolarMACE NPT across canonical restart points (1b-T14 did not cover it). A
``Simulation`` with Langevin-BAOAB (hydrostatic) on POLAR-1-S NaCl 3x3x3 warms up, then writes a restart
(a canonical point: quiesce, rebuild every list, reset the k-space state) before each of --chunks
chunks of --chunk steps. Per chunk it records Dynamo's compile counters, the PolarMACE k-vector record
and the step time, so a recompile at a restart point shows up. Then a fresh child process resumes from
the last restart and runs the last chunk again; its final state is compared with this process's
(reported: the hybrid is not bitwise, plain cuEq float64 should agree to about 1e-12 A).

``calc``: ``fm.calc``'s compile at its first calls (1b-T18 found a second compile at the second call,
4.5 s plain cuEq and 19 s hybrid on NaCl 2x2x2). Measured only: per call the wall time and the compile
counters, over --calls rattled copies. The backend is the planner's for the environment (hybrid in the
OEq environment, plain cuEq in the plain one).

Use an otherwise idle GPU. Output: <output>/metadata.json.
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

import numpy as np

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_step_time  # noqa: E402
import library_gate  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    polar = sub.add_parser("polar-npt")
    polar.add_argument("--output", required=True, type=Path)
    polar.add_argument("--model", default="polar1-s")
    polar.add_argument("--structure", default="NaCl")
    polar.add_argument("--repeat", nargs=3, type=int, default=[3, 3, 3])
    polar.add_argument("--temperature-K", type=float, default=300.0)
    polar.add_argument("--seed", type=int, default=1)
    polar.add_argument("--warmup", type=int, default=300)
    polar.add_argument("--chunk", type=int, default=200)
    polar.add_argument("--chunks", type=int, default=5)
    resume = sub.add_parser("polar-npt-resume")
    resume.add_argument("--restart", required=True, type=Path)
    resume.add_argument("--steps", required=True, type=int)
    calc = sub.add_parser("calc")
    calc.add_argument("--output", required=True, type=Path)
    calc.add_argument("--model", default="mh1-r2scan")
    calc.add_argument("--structure", default="NaCl")
    calc.add_argument("--repeat", nargs=3, type=int, default=[2, 2, 2])
    calc.add_argument("--calls", type=int, default=6)
    calc.add_argument("--seed", type=int, default=1)
    return p.parse_args(argv)


def _counters():
    from torch._dynamo.utils import counters

    return ab_step_time.compile_counters(counters)


def _final(sim):
    committed = sim.committed()
    return dict(step=int(committed.step), positions=np.asarray(committed.positions).tolist(),
                cell=np.asarray(committed.cell).tolist(), energy=float(committed.energy))


def polar_npt(args):
    import torch
    from ase import units

    from fastmace import integrators
    from fastmace.io.output import Output
    from fastmace.model import loading
    from fastmace.simulation import Simulation

    loaded = loading.load_pristine(args.model)
    name, atoms = library_gate.read_structure(args.structure, args.repeat)
    integrator = integrators.LangevinBAOAB(timestep_fs=0.5, temperature_K=args.temperature_K,
                                           friction_per_fs=0.02 * units.fs, cell_mode="hydrostatic")
    args.output.mkdir(parents=True, exist_ok=False)
    output = Output(str(args.output / "run"), trajectory_every=None, restart_every=None)
    meta = dict(command="polar-npt", created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
                fastmace_commit=ab_step_time.fastmace_commit(), model=args.model, structure=name, natoms=len(atoms),
                gpu=torch.cuda.get_device_name(0), torch=torch.__version__, chunks=[])
    started = time.perf_counter()
    sim = Simulation(atoms, loaded, integrator=integrator, output=output, seed=args.seed)
    meta.update(setup_s=time.perf_counter() - started, plan=sim.plan.describe())
    sim.run(steps=args.warmup)
    restart = None
    for index in range(args.chunks):
        restart = args.output / f"restart-{index}"
        before = _counters()
        sim.save_restart(restart)  # a canonical restart point
        torch.cuda.synchronize()
        started = time.perf_counter()
        sim.run(steps=args.chunk)
        torch.cuda.synchronize()
        changes, recompiled = ab_step_time.compile_changes(before, _counters())
        polar = dict(sim._engine.force_field.info()).get("polar")
        meta["chunks"].append(dict(restart_step=args.warmup + index * args.chunk, ms_per_step=(
            time.perf_counter() - started) / args.chunk * 1e3, recompiled=recompiled, changes=changes, polar=polar))
        print(f"chunk {index}: recompiled={recompiled} {meta['chunks'][-1]['ms_per_step']:.3f} ms/step", flush=True)
    final = _final(sim)
    sim.close()
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "polar-npt-resume",
                            "--restart", str(restart), "--steps", str(args.chunk)],
                           capture_output=True, text=True, check=False)
    lines = [line for line in child.stdout.splitlines() if line.startswith("RESULT ")]
    meta["resume"] = dict(exit_code=child.returncode, stderr_tail=child.stderr.splitlines()[-5:])
    if lines:
        resumed = json.loads(lines[-1][7:])
        meta["resume"].update(step=resumed["step"], recompiled_after_resume=resumed["recompiled"],
                              max_position_difference_A=float(np.abs(np.array(resumed["positions"])
                                                                     - np.array(final["positions"])).max()),
                              max_cell_difference_A=float(np.abs(np.array(resumed["cell"])
                                                                 - np.array(final["cell"])).max()),
                              energy_difference_eV=resumed["energy"] - final["energy"])
    meta["recompiled_at_a_restart_point"] = any(chunk["recompiled"] for chunk in meta["chunks"])
    return meta


def polar_npt_resume(args):
    import torch

    from fastmace.simulation import Simulation

    sim = Simulation.resume(args.restart)
    sim.run(steps=1)  # compiles and captures, as any new process does
    torch.cuda.synchronize()
    before = _counters()
    sim.run(steps=args.steps - 1)
    _, recompiled = ab_step_time.compile_changes(before, _counters())
    record = dict(_final(sim), recompiled=recompiled)
    sim.close()
    print("RESULT " + json.dumps(record), flush=True)
    return record


def calc_compiles(args):
    import torch

    from fastmace.ase import calc as fm_calc

    calculator = fm_calc(args.model)
    name, atoms = library_gate.read_structure(args.structure, args.repeat)
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    meta = dict(command="calc", created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
                fastmace_commit=ab_step_time.fastmace_commit(), model=args.model, structure=name, natoms=len(atoms),
                gpu=torch.cuda.get_device_name(0), torch=torch.__version__, calls=[])
    for index in range(args.calls):
        copy = atoms.copy()
        copy.positions += 0.05 * rng.standard_normal(copy.positions.shape)
        copy.calc = calculator
        before = _counters()
        started = time.perf_counter()
        copy.get_forces()
        copy.get_stress()
        torch.cuda.synchronize()
        changes, recompiled = ab_step_time.compile_changes(before, _counters())
        meta["calls"].append(dict(index=index, wall_s=time.perf_counter() - started, compiled=recompiled,
                                  changes=changes))
        print(f"call {index}: {meta['calls'][-1]['wall_s']:.2f} s compiled={recompiled}", flush=True)
    meta.update(plan=calculator.plan.describe() if calculator.plan is not None else None, notes=list(calculator.notes),
                compiles=[call["index"] for call in meta["calls"] if call["compiled"]])
    return meta


def main(argv=None):
    args = parse_args(argv)
    if args.command == "polar-npt-resume":
        return polar_npt_resume(args)
    meta = polar_npt(args) if args.command == "polar-npt" else calc_compiles(args)
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (args.output / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return meta


if __name__ == "__main__":
    main()
