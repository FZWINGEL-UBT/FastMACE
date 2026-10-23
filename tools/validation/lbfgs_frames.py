"""GPU LBFGS on annealing quench frames against ASE's LBFGS on ``fm.calc`` (plan task 4-T05; SPEC §10.3).

Frames from the research engine's ``captured.traj`` (the frames its 13 October quench relaxed with ASE's
LBFGS) are relaxed by one side per process, since the two tiers must not share a process (SPEC §8.7):

- ``--side ase``: ASE 3.27's ``LBFGS`` on ``fm.calc`` (the compiled tier, same model, head and precision),
  what ``protocols.quench.relax`` runs today;
- ``--side gpu``: ``Simulation(optimizer=fm.optimize.LBFGS())`` per ``--modes`` (``graph``: captured graphs,
  padded edges, rebuilds between replays; ``compiled``: the compiled tier, exact edge counts) and ``--skins``
  (Angstrom): conductor's choice 6, the mode and skin of optimiser plans. One engine per (mode, skin) relaxes
  every frame in turn, each set with ``Simulation.update(cell=, positions=)`` as 4-T09's quench will;
  ``--compare`` (the ASE side's JSON) adds |dE| (meV/atom) and the largest per-atom distance (minimum image)
  against ASE's minimum.

Each relaxation records steps, converged, wall seconds, start and final energy (eV/atom), fmax, neighbour
rebuilds and re-captures. LBFGS fixed cell, fmax 0.01 eV/A, memory 100, maxstep 0.2 A, at most 5,000 steps.
Hybrid float32 by default; run each side on an otherwise idle GPU (``tools/ci/local_gpu.py run``) with the OEq
environment's Python:

    PYTHONPATH=$PWD ~/environments/mace_cueq_oeq/bin/python tools/validation/lbfgs_frames.py --side ase \\
        --checkpoint ~/models/mace-mh-1.model --output results/4-T05-<machine>-<date>/ase \\
        --traj NaCl=<dir>/NaCl/captured.traj NbCl5=<dir>/NbCl5/captured.traj --frames 0 10 20
    ... --side gpu --output results/4-T05-<machine>-<date>/gpu --compare results/4-T05-<machine>-<date>/ase.json

Writes ``<output>.json`` and ``<output>.npz`` (the minima's positions).
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--side", choices=("ase", "gpu"), required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--model", default="mh1")
    p.add_argument("--head", default="matpes_r2scan")
    p.add_argument("--traj", nargs="+", required=True, help="NAME=PATH of a captured.traj")
    p.add_argument("--frames", nargs="+", type=int, default=[0, 10, 20])
    p.add_argument("--modes", nargs="+", choices=("graph", "compiled"), default=["graph", "compiled"])
    p.add_argument("--skins", nargs="+", type=float, default=[0.3, 0.6, 1.0])
    p.add_argument("--precision", choices=("float32", "float64"), default="float32")
    p.add_argument("--fmax", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--compare", type=Path, default=None, help="The ASE side's JSON (gpu side)")
    return p.parse_args(argv)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def mic_distance(cell, a, b):
    """The largest per-atom distance between positions ``a`` and ``b`` in ``cell``, minimum image."""
    import numpy as np

    fractional = np.linalg.solve(cell.T, (b - a).T).T
    fractional -= np.round(fractional)
    return float(np.linalg.norm(fractional @ cell, axis=1).max())


def read_frames(args):
    from ase.io import read

    out = {}
    for spec in args.traj:
        name, _, path = spec.partition("=")
        frames = {}
        for index in args.frames:
            atoms = read(path, index=index)
            atoms.calc = None
            atoms.set_constraint()
            frames[index] = atoms
        out[name] = (path, frames)
    return out


def ase_side(args, loaded, frames, meta, minima):
    import numpy as np
    from ase.optimize import LBFGS

    from fastmace.ase.calculator import FastMACECalculator

    calc = FastMACECalculator(loaded, precision=args.precision)
    for name, (path, by_index) in frames.items():
        record = meta["structures"].setdefault(name, {"traj": path, "frames": {}})
        for index, atoms in by_index.items():
            relaxed = atoms.copy()
            relaxed.set_momenta(np.zeros((len(relaxed), 3)))
            relaxed.calc = calc
            started = time.perf_counter()
            initial = float(relaxed.get_potential_energy())
            run = LBFGS(relaxed, logfile=None)
            converged = bool(run.run(fmax=args.fmax, steps=args.steps))
            seconds = time.perf_counter() - started
            result = {"converged": converged, "steps": int(run.get_number_of_steps()), "seconds": seconds,
                      "initial_eV_atom": initial / len(relaxed),
                      "energy_eV_atom": float(relaxed.get_potential_energy()) / len(relaxed),
                      "fmax_eV_A": float(np.linalg.norm(relaxed.get_forces(), axis=1).max()),
                      "ms_per_step": 1000 * seconds / max(run.get_number_of_steps(), 1)}
            record["frames"][str(index)] = result
            minima[f"{name}/{index}"] = relaxed.positions.copy()
            print(f"{name} frame {index} ASE: {result['steps']} steps {seconds:.1f} s "
                  f"E {result['energy_eV_atom']:.6f} eV/atom", flush=True)
            yield


def gpu_side(args, loaded, frames, meta, minima):
    import gc

    import torch

    import fastmace as fm
    from fastmace.config import NeighbourConfig
    from fastmace.model.forcefield import ForceField

    compare = json.loads(args.compare.read_text()) if args.compare else None
    reference = dict(__import__("numpy").load(args.compare.with_suffix(".npz"))) if args.compare else {}
    for name, (path, by_index) in frames.items():
        record = meta["structures"].setdefault(name, {"traj": path, "runs": {}})
        first = next(iter(by_index.values()))
        for mode in args.modes:
            for skin in args.skins:
                key = f"{mode}-skin{skin:g}"
                started = time.perf_counter()
                sim = fm.Simulation(first.copy(), ForceField(loaded, first, precision=args.precision),
                                    optimizer=fm.optimize.LBFGS(), seed=0, mode=None if mode == "graph" else mode,
                                    neighbours=NeighbourConfig(skin=skin))
                run = record["runs"].setdefault(key, {"setup_s": time.perf_counter() - started,
                                                      "tier": sim.plan.tier, "skin_A": skin, "frames": {}})
                for index, atoms in by_index.items():
                    sim.update(cell=atoms.cell.array, positions=atoms.positions)
                    relaxed, summary = sim.relax(fmax=args.fmax, steps=args.steps)
                    torch.cuda.synchronize()
                    result = {k: summary[k] for k in ("converged", "steps", "seconds", "initial_eV_atom",
                                                      "energy_eV_atom", "fmax_eV_A", "neighbour_rebuilds",
                                                      "recaptures")}
                    result["ms_per_step"] = 1000 * summary["seconds"] / max(summary["steps"], 1)
                    if compare is not None and str(index) in compare["structures"].get(name, {}).get("frames", {}):
                        ase = compare["structures"][name]["frames"][str(index)]
                        result["dE_vs_ase_meV_atom"] = 1000 * (summary["energy_eV_atom"] - ase["energy_eV_atom"])
                        result["dsteps_vs_ase"] = summary["steps"] - ase["steps"]
                        result["max_distance_vs_ase_A"] = mic_distance(atoms.cell.array,
                                                                       reference[f"{name}/{index}"],
                                                                       relaxed.positions)
                    run["frames"][str(index)] = result
                    minima[f"{name}/{key}/{index}"] = relaxed.positions.copy()
                    print(f"{name} frame {index} {key}: {result['steps']} steps {result['seconds']:.2f} s "
                          f"({result['ms_per_step']:.2f} ms/step) rebuilds {result['neighbour_rebuilds']} "
                          f"recaptures {result['recaptures']} dE {result.get('dE_vs_ase_meV_atom', float('nan')):.4f} "
                          f"meV/atom dmax {result.get('max_distance_vs_ase_A', float('nan')):.4f} A", flush=True)
                    yield
                run["engine"] = {key: value for key, value in sim._engine.info().items()
                                 if key in ("edge_sizes_used", "neighbour_rebuilds", "recaptures", "graphs",
                                            "recapture_log", "pool_resets")}
                sim.close()
                del sim
                gc.collect()
                torch.cuda.empty_cache()


def main(argv=None):
    args = parse_args(argv)
    import numpy as np
    import torch

    from fastmace.model import loading

    oeq = str(loading.prepare_hybrid())
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    loaded = loading.load_pristine(args.model, args.head, checkpoint=str(checkpoint))
    frames = read_frames(args)
    meta = {"schema": 2, "task": "4-T05", "side": args.side,
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "fastmace_commit": git_commit(),
            "machine": socket.gethostname(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "ase": __import__("ase").__version__, "precision": args.precision, "oeq_runtime": oeq,
            "model": args.model, "head": args.head, "checkpoint": str(checkpoint), "fmax": args.fmax,
            "steps": args.steps, "optimizer": "LBFGS(maxstep=0.2, memory=100, damping=1.0, alpha=70.0)",
            "structures": {}}
    minima = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    side = ase_side if args.side == "ase" else gpu_side
    for _ in side(args, loaded, frames, meta, minima):
        args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        np.savez(args.output.with_suffix(".npz"), **minima)
    meta["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
