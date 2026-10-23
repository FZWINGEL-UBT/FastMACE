"""GPU optimisers on annealing quench frames against ASE's on ``fm.calc`` (plan tasks 4-T05, 4-T06; SPEC §10.3).

``--optimizer lbfgs`` (default) or ``fire``, ASE 3.27's defaults on both sides. Frames from the research
engine's ``captured.traj`` (the frames its 13 October quench relaxed with ASE's LBFGS) are relaxed by one side
per process, since the two tiers must not share a process (SPEC §8.7):

- ``--side ase``: ASE 3.27's ``LBFGS`` or ``FIRE`` on ``fm.calc`` (the compiled tier, same model, head and
  precision), what ``protocols.quench.relax`` runs today for LBFGS;
- ``--side gpu``: ``Simulation(optimizer=fm.optimize.LBFGS())`` (or ``FIRE()``) per ``--modes`` (``graph``:
  captured graphs, padded edges, rebuilds between replays; ``compiled``: the compiled tier, exact edge
  counts) and ``--skins`` (Angstrom): conductor's choice 6, the mode and skin of optimiser plans. One engine
  per (mode, skin) relaxes every frame in turn, each set with ``Simulation.update(cell=, positions=)`` as
  4-T09's quench will; ``--compare`` (the ASE side's JSON) adds |dE| (meV/atom), the step difference and the
  largest per-atom distance (minimum image) against ASE's minimum.

Each relaxation records steps, converged, wall seconds, start and final energy (eV/atom), fmax, neighbour
rebuilds and re-captures; FIRE also its switch flips, the steps that took ASE's ``v = 0``, ``dt *= fdec``
branch (on the GPU the optimiser's ``restarts`` counter, on the ASE side the steps where ``dt`` fell). The
first frame of each side includes its warm-up (compiles, captures). Fixed cell, fmax 0.01 eV/A, at most
5,000 steps. Hybrid float32 by default; run each side on an otherwise idle GPU (``tools/ci/local_gpu.py run``)
with the OEq environment's Python:

    PYTHONPATH=$PWD ~/environments/mace_cueq_oeq/bin/python tools/bench/optimizer_frames.py --side ase \\
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
    p.add_argument("--optimizer", choices=("lbfgs", "fire"), default="lbfgs")
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
    from ase.optimize import FIRE, LBFGS

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
            run = (LBFGS if args.optimizer == "lbfgs" else FIRE)(relaxed, logfile=None)
            converged, flips, dt = False, 0, getattr(run, "dt", None)
            for converged in run.irun(fmax=args.fmax, steps=args.steps):
                if dt is not None:  # FIRE's switch flip: v = 0, dt *= fdec
                    flips += int(run.dt < dt)
                    dt = run.dt
            seconds = time.perf_counter() - started
            steps = int(run.get_number_of_steps())
            result = {"converged": bool(converged), "steps": steps, "seconds": seconds,
                      "initial_eV_atom": initial / len(relaxed),
                      "energy_eV_atom": float(relaxed.get_potential_energy()) / len(relaxed),
                      "fmax_eV_A": float(np.linalg.norm(relaxed.get_forces(), axis=1).max()),
                      "ms_per_step": 1000 * seconds / max(steps, 1),
                      "switch_flips": flips if args.optimizer == "fire" else None}
            record["frames"][str(index)] = result
            minima[f"{name}/{index}"] = relaxed.positions.copy()
            print(f"{name} frame {index} ASE {args.optimizer}: {steps} steps {seconds:.1f} s "
                  f"E {result['energy_eV_atom']:.6f} eV/atom flips {result['switch_flips']}", flush=True)
            yield


def gpu_side(args, loaded, frames, meta, minima):
    import gc

    import numpy as np
    import torch

    import fastmace as fm
    from fastmace.config import NeighbourConfig
    from fastmace.model.forcefield import ForceField

    compare = json.loads(args.compare.read_text()) if args.compare else None
    reference = dict(np.load(args.compare.with_suffix(".npz"))) if args.compare else {}
    record_class = fm.optimize.LBFGS if args.optimizer == "lbfgs" else fm.optimize.FIRE
    for name, (path, by_index) in frames.items():
        record = meta["structures"].setdefault(name, {"traj": path, "runs": {}})
        first = next(iter(by_index.values()))
        for mode in args.modes:
            for skin in args.skins:
                key = f"{mode}-skin{skin:g}"
                started = time.perf_counter()
                sim = fm.Simulation(first.copy(), ForceField(loaded, first, precision=args.precision),
                                    optimizer=record_class(), seed=0, mode=None if mode == "graph" else mode,
                                    neighbours=NeighbourConfig(skin=skin))
                run = record["runs"].setdefault(key, {"setup_s": time.perf_counter() - started,
                                                      "tier": sim.plan.tier, "skin_A": skin, "frames": {}})
                optimizer = sim._engine.optimizer
                for index, atoms in by_index.items():
                    sim.update(cell=atoms.cell.array, positions=atoms.positions)
                    restarts = getattr(optimizer, "restarts", None)  # FIRE's switch flips (4-T06), cumulative
                    before = None if restarts is None else int(restarts)
                    relaxed, summary = sim.relax(fmax=args.fmax, steps=args.steps)
                    torch.cuda.synchronize()
                    result = {k: summary[k] for k in ("converged", "steps", "seconds", "initial_eV_atom",
                                                      "energy_eV_atom", "fmax_eV_A", "neighbour_rebuilds",
                                                      "recaptures")}
                    result["ms_per_step"] = 1000 * summary["seconds"] / max(summary["steps"], 1)
                    result["switch_flips"] = None if before is None else int(optimizer.restarts) - before
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
                          f"recaptures {result['recaptures']} flips {result['switch_flips']} "
                          f"dE {result.get('dE_vs_ase_meV_atom', float('nan')):.4f} meV/atom "
                          f"dmax {result.get('max_distance_vs_ase_A', float('nan')):.4f} A", flush=True)
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
    from fastmace.model import compile as compiling
    from fastmace.model import loading

    # The engine's process constraint: the allocator setting before anything touches CUDA. The ASE side
    # takes it too, so that both sides run with the same allocator.
    allocator = compiling.configure_allocator()
    import numpy as np
    import torch

    oeq = str(loading.prepare_hybrid())
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    loaded = loading.load_pristine(args.model, args.head, checkpoint=str(checkpoint))
    frames = read_frames(args)
    meta = {"schema": 3, "task": "4-T05/4-T06", "side": args.side, "optimizer": args.optimizer,
            "optimizer_settings": "ASE 3.27 defaults", "cuda_allocator": allocator,
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "fastmace_commit": git_commit(),
            "machine": socket.gethostname(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "ase": __import__("ase").__version__, "precision": args.precision, "oeq_runtime": oeq,
            "model": args.model, "head": args.head, "checkpoint": str(checkpoint), "fmax": args.fmax,
            "steps": args.steps, "structures": {}}
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
