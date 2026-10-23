"""One annealing frame where GPU and ASE LBFGS find different minima: two basins, or a defect? (plan task 4-T08).

For a frame that fails SPEC decision 46's energy clause, one process per side (the tiers must not share a
process, SPEC §8.7), with ``tools/bench/optimizer_frames.py``'s settings (ASE 3.27 LBFGS defaults, fmax 0.01
eV/A, at most 5,000 steps, fixed cell, MH-1 ``matpes_r2scan``):

- the relaxation from the frame again, one step at a time, with the positions and energy after every step
  (for where the two trajectories first diverge). On the GPU, ``sim.relax(steps=1)`` repeated is one
  relaxation: each call continues with the history, as ASE's ``run`` does;
- the other side's minimum (``--other``, its ``.npz`` from the 4-T08 run) re-relaxed by this side's optimiser:
  does it stay put?
- ``--precision float64`` for the GPU side's float64 run.

    PYTHONPATH=$PWD ~/environments/mace_cueq_oeq/bin/python tools/validation/anneal_frame_probe.py --side gpu \\
        --checkpoint ~/models/mace-mh-1.model --traj NaTaCl6=<data>/NaTaCl6/captured.traj --frame 12 \\
        --other <runs>/NaTaCl6/ase.npz --output <dir>/gpu-float32

Writes ``<output>.json`` (per-step energies, steps, the re-relaxation) and ``<output>.npz`` (per-step positions).
``compare`` reads two outputs and reports the first step where the trajectories differ by more than each of
1e-6, 1e-4, 1e-2 and 0.1 A, and the two minima's difference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

THRESHOLDS_A = (1e-6, 1e-4, 1e-2, 0.1)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--side", choices=("ase", "gpu"), required=True)
    run.add_argument("--checkpoint", required=True, type=Path)
    run.add_argument("--model", default="mh1")
    run.add_argument("--head", default="matpes_r2scan")
    run.add_argument("--traj", required=True, help="NAME=PATH of a captured.traj")
    run.add_argument("--frame", required=True, type=int)
    run.add_argument("--other", type=Path, default=None, help="the other side's .npz of the 4-T08 run")
    run.add_argument("--other-key", default=None,
                     help="key in --other (default NAME/FRAME or NAME/graph-skinauto/FRAME)")
    run.add_argument("--precision", choices=("float32", "float64"), default="float32")
    run.add_argument("--fmax", type=float, default=0.01)
    run.add_argument("--steps", type=int, default=5000)
    run.add_argument("--output", required=True, type=Path)
    compare = sub.add_parser("compare")
    compare.add_argument("a", type=Path)
    compare.add_argument("b", type=Path)
    return p.parse_args(argv)


def divergence(cell, a, b):
    """Per step, the largest minimum-image distance between trajectories ``a`` and ``b`` (steps x N x 3), over
    the steps both have; and for each of ``THRESHOLDS_A`` the first step above it (``None`` if never)."""
    import numpy as np

    n = min(len(a), len(b))
    cell = np.asarray(cell, dtype=float)
    fractional = np.linalg.solve(cell.T, (b[:n] - a[:n]).reshape(-1, 3).T).T
    fractional -= np.round(fractional)
    largest = np.linalg.norm(fractional @ cell, axis=1).reshape(n, -1).max(axis=1)
    first = {f"{t:g}": next((int(i) for i in range(n) if largest[i] > t), None) for t in THRESHOLDS_A}
    return largest, first


def _other_positions(args, name):
    import numpy as np

    saved = dict(np.load(args.other))
    keys = [args.other_key] if args.other_key else [f"{name}/{args.frame}", f"{name}/graph-skinauto/{args.frame}"]
    for key in keys:
        if key in saved:
            return key, saved[key]
    raise KeyError(f"no {name} frame {args.frame} in {args.other}")


def ase_side(args, loaded, atoms, name, record, trajectory):
    import numpy as np
    from ase.optimize import LBFGS

    from fastmace.ase.calculator import FastMACECalculator

    calc = FastMACECalculator(loaded, precision=args.precision)

    def relax(start, keep):
        relaxed = atoms.copy()
        relaxed.set_positions(start)
        relaxed.set_momenta(np.zeros((len(relaxed), 3)))
        relaxed.calc = calc
        started = time.perf_counter()
        run = LBFGS(relaxed, logfile=None)
        energies, converged = [], False
        for converged in run.irun(fmax=args.fmax, steps=args.steps):
            energies.append(float(relaxed.get_potential_energy()) / len(relaxed))
            if keep:
                trajectory.append(relaxed.positions.copy())
        return relaxed, {"converged": bool(converged), "steps": int(run.get_number_of_steps()),
                         "seconds": time.perf_counter() - started, "energies_eV_atom": energies,
                         "fmax_eV_A": float(np.linalg.norm(relaxed.get_forces(), axis=1).max())}

    relaxed, record["relaxation"] = relax(atoms.positions, keep=True)
    record["minimum_positions_saved"] = True
    if args.other is not None:
        key, start = _other_positions(args, name)
        again, summary = relax(start, keep=False)
        summary["start"] = key
        summary["displacement_from_start_A"] = float(
            divergence(atoms.cell.array, start[None], again.positions[None])[0][0])
        record["re_relaxation"] = summary
    return relaxed.positions


def gpu_side(args, loaded, atoms, name, record, trajectory):
    import numpy as np
    import torch

    import fastmace as fm
    from fastmace.config import NeighbourConfig
    from fastmace.model.forcefield import ForceField

    sim = fm.Simulation(atoms.copy(), ForceField(loaded, atoms, precision=args.precision),
                        optimizer=fm.optimize.LBFGS(), seed=0, neighbours=NeighbourConfig(skin="auto"))
    record["plan"] = {"tier": sim.plan.tier, "skin_A": sim.plan.neighbours.skin, "precision": args.precision}

    def relax(start, keep):
        sim.update(cell=atoms.cell.array, positions=start)
        started = time.perf_counter()
        energies, steps, summary = [], 0, None
        if keep:  # one step per call: the positions after every step
            trajectory.append(np.asarray(start, dtype=float).copy())
            while steps < args.steps:
                relaxed, summary = sim.relax(fmax=args.fmax, steps=1)
                if not energies:
                    energies.append(summary["initial_eV_atom"])
                if summary["steps"] == 0:  # converged at the check before the step
                    break
                steps += summary["steps"]
                energies.append(summary["energy_eV_atom"])
                trajectory.append(relaxed.positions.copy())
            converged = summary["converged"]
        else:
            relaxed, summary = sim.relax(fmax=args.fmax, steps=args.steps)
            steps, converged, energies = summary["steps"], summary["converged"], [summary["energy_eV_atom"]]
        torch.cuda.synchronize()
        return relaxed, {"converged": bool(converged), "steps": int(steps), "seconds": time.perf_counter() - started,
                         "energies_eV_atom": energies, "fmax_eV_A": summary["fmax_eV_A"]}

    relaxed, record["relaxation"] = relax(atoms.positions, keep=True)
    minimum = relaxed.positions
    if args.other is not None:
        key, start = _other_positions(args, name)
        again, summary = relax(start, keep=False)
        summary["start"] = key
        summary["displacement_from_start_A"] = float(
            divergence(atoms.cell.array, start[None], again.positions[None])[0][0])
        record["re_relaxation"] = summary
    sim.close()
    return minimum


def run(args):
    from fastmace.model import compile as compiling
    from fastmace.model import loading

    allocator = compiling.configure_allocator()
    import numpy as np
    import torch
    from ase.io import read

    oeq = str(loading.prepare_hybrid())
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    loaded = loading.load_pristine(args.model, args.head, checkpoint=str(checkpoint))
    name, _, path = args.traj.partition("=")
    atoms = read(path, index=args.frame)
    atoms.calc = None
    atoms.set_constraint()
    record = {"schema": 1, "task": "4-T08 probe", "side": args.side, "structure": name, "traj": path,
              "frame": args.frame, "precision": args.precision, "fmax": args.fmax, "steps_limit": args.steps,
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "ase": __import__("ase").__version__,
              "cuda_allocator": allocator, "oeq_runtime": oeq, "checkpoint": str(checkpoint)}
    trajectory = []
    minimum = (ase_side if args.side == "ase" else gpu_side)(args, loaded, atoms, name, record, trajectory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_suffix(".npz"), trajectory=np.asarray(trajectory), minimum=minimum,
             cell=atoms.cell.array)
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    r = record["relaxation"]
    print(f"{name} frame {args.frame} {args.side} {args.precision}: {r['steps']} steps, converged {r['converged']}, "
          f"E {r['energies_eV_atom'][-1]:.7f} eV/atom", flush=True)
    if "re_relaxation" in record:
        q = record["re_relaxation"]
        print(f"  re-relaxed from {q['start']}: {q['steps']} steps, E {q['energies_eV_atom'][-1]:.7f} eV/atom, "
              f"moved {q['displacement_from_start_A']:.4g} A", flush=True)
    return 0


def compare(a, b):
    """The first steps where trajectories ``a`` and ``b`` part, and the difference of their minima."""
    import numpy as np

    sa, sb = np.load(a.with_suffix(".npz")), np.load(b.with_suffix(".npz"))
    ra = json.loads(a.with_suffix(".json").read_text())["relaxation"]
    rb = json.loads(b.with_suffix(".json").read_text())["relaxation"]
    largest, first = divergence(sa["cell"], sa["trajectory"], sb["trajectory"])
    final = float(divergence(sa["cell"], sa["minimum"][None], sb["minimum"][None])[0][0])
    n = min(len(ra["energies_eV_atom"]), len(rb["energies_eV_atom"]))
    de = [1000 * (rb["energies_eV_atom"][i] - ra["energies_eV_atom"][i]) for i in range(n)]
    return {"first_step_above_A": first, "largest_by_step_A": [float(x) for x in largest],
            "dE_by_step_meV_atom": de, "steps": [ra["steps"], rb["steps"]],
            "minima_max_distance_A": final,
            "minima_dE_meV_atom": 1000 * (rb["energies_eV_atom"][-1] - ra["energies_eV_atom"][-1])}


def main(argv=None):
    args = parse_args(argv)
    if args.command == "run":
        return run(args)
    result = compare(args.a, args.b)
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("by_step_A") and k != "dE_by_step_meV_atom"},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
