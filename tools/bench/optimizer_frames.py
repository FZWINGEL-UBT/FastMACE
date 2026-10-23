"""GPU optimisers on annealing quench frames against ASE's on ``fm.calc`` (plan tasks 4-T05, 4-T06, 4-T08;
SPEC §10.3).

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
first frame of each side and process includes its warm-up (compiles, captures; ``warm_up`` in its
record). Fixed cell, fmax 0.01 eV/A, at most 5,000 steps. Hybrid float32 by default; run each side on an
otherwise idle GPU (``tools/ci/local_gpu.py run``) with the OEq environment's Python:

    PYTHONPATH=$PWD ~/environments/mace_cueq_oeq/bin/python tools/bench/optimizer_frames.py --side ase \\
        --checkpoint ~/models/mace-mh-1.model --output results/4-T05-<machine>-<date>/ase \\
        --traj NaCl=<dir>/NaCl/captured.traj NbCl5=<dir>/NbCl5/captured.traj --frames 0 10 20
    ... --side gpu --output results/4-T05-<machine>-<date>/gpu --compare results/4-T05-<machine>-<date>/ase.json

Writes ``<output>.json`` and ``<output>.npz`` (the minima's positions, and with a cell filter their cells),
after every frame.

**Variable cell (4-T07):** ``--cell-filter frechet`` (``--pressure-GPa``) relaxes with ASE's
``FrechetCellFilter`` on the ASE side and ``fm.optimize.FrechetCell`` on the GPU side. A variable-cell engine
takes no host cell write, so the GPU side builds one ``Simulation`` per frame. ``--structures NAME ...``
replaces ``--traj``: the benchmark r2SCAN CIFs under ``--structures-root`` (ResearchModule,
``Structures/Benchmark/R2SCAN``), repeated ``--repeat``, the cell scaled isotropically by ``1 + --strain``
with the atoms, then rattled by ``--rattle`` Angstrom (``Atoms.rattle``, seed ``--seed`` plus the CRC-32 of the
name): one frame, index 0. ``--compare`` then also gives the largest cell-length difference (Angstrom) and
the relative volume difference against ASE's minimum. ``--dispersion d3bj`` adds D3 on both sides. Energies
are E_pot per atom on both sides; the GPU side's summary energy with a pressure is E + pV (ASE's filter
energy), and the pV term is taken off.

    ... --side ase --cell-filter frechet --structures NaCl NbCl5 --output results/4-T07-<machine>-<date>/ase-lbfgs

For 4-T08 (all 210 frames, ``tools/validation/anneal_minima.py``): ``--frames all`` relaxes every frame of
each ``captured.traj``, ``--skins auto`` leaves the skin to the plan (rule 11's optimiser skin), and
``--resume`` continues an interrupted output, skipping the frames it has. ``FASTMACE_COMMIT`` gives the
commit where ``git rev-parse`` cannot (a Windows worktree seen from WSL).
"""

from __future__ import annotations

import argparse
import json
import os
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
    p.add_argument("--head", default="matpes_r2scan", help="\"none\" for a single-head model (POLAR-1-S)")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--traj", nargs="+", help="NAME=PATH of a captured.traj")
    source.add_argument("--structures", nargs="+", help="benchmark names, strained and rattled (4-T07)")
    p.add_argument("--structures-root", type=Path, default=Path("/mnt/g/Margraf/ResearchModule"))
    p.add_argument("--repeat", nargs=3, type=int, default=[3, 3, 3])
    p.add_argument("--strain", type=float, default=0.02)
    p.add_argument("--rattle", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cell-filter", choices=("none", "frechet"), default="none")
    p.add_argument("--pressure-GPa", type=float, default=0.0)
    p.add_argument("--dispersion", choices=("none", "d3bj"), default="none")
    p.add_argument("--frames", nargs="+", default=["0", "10", "20"], help="frame indices, or all")
    p.add_argument("--modes", nargs="+", choices=("graph", "compiled"), default=["graph", "compiled"])
    p.add_argument("--skins", nargs="+", type=_skin, default=[0.3, 0.6, 1.0], help="Angstrom, or auto")
    p.add_argument("--precision", choices=("float32", "float64"), default="float32")
    p.add_argument("--fmax", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--compare", type=Path, default=None, help="The ASE side's JSON (gpu side)")
    p.add_argument("--resume", action="store_true", help="continue <output>, skipping the frames it has")
    return p.parse_args(argv)


def _skin(text):
    return text if text == "auto" else float(text)


def run_key(mode, skin):
    return f"{mode}-skin{skin:g}" if skin != "auto" else f"{mode}-skinauto"


def frame_indices(spec, count):
    """``--frames`` for a trajectory of ``count`` frames: ``["all"]`` or indices."""
    if "all" in spec:
        if len(spec) != 1:
            raise ValueError("--frames all takes no indices beside it")
        return list(range(count))
    indices = [int(index) for index in spec]
    for index in indices:
        if not -count <= index < count:
            raise IndexError(f"frame {index} of a trajectory of {count} frames")
    return indices


def load_previous(output):
    """The record and minima an interrupted run left at ``output`` (``(None, {})`` if none)."""
    import numpy as np

    path = Path(output).with_suffix(".json")
    if not path.is_file():
        return None, {}
    saved = Path(output).with_suffix(".npz")
    return json.loads(path.read_text(encoding="utf-8")), dict(np.load(saved)) if saved.is_file() else {}


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("FASTMACE_COMMIT")


def mic_distance(cell, a, b):
    """The largest per-atom distance between positions ``a`` and ``b`` in ``cell``, minimum image."""
    import numpy as np

    fractional = np.linalg.solve(cell.T, (b - a).T).T
    fractional -= np.round(fractional)
    return float(np.linalg.norm(fractional @ cell, axis=1).max())


def dispersion(args):
    return None if args.dispersion == "none" else args.dispersion


def cell_result(cell):
    """The relaxed cell's lengths (Angstrom), angles and volume, for the JSON."""
    import numpy as np
    from ase.geometry import cell_to_cellpar

    parameters = cell_to_cellpar(cell)
    return {"cell_lengths_A": [float(value) for value in parameters[:3]],
            "cell_angles_deg": [float(value) for value in parameters[3:]], "volume_A3": abs(float(np.linalg.det(cell)))}


def read_frames(args):
    from ase.io import read

    out = {}
    for name in args.structures or ():
        import zlib

        path = args.structures_root / "Structures" / "Benchmark" / "R2SCAN" / f"{name}.cif"
        atoms = read(path).repeat(tuple(args.repeat))
        atoms.set_cell(atoms.cell.array * (1.0 + args.strain), scale_atoms=True)
        atoms.rattle(args.rattle, seed=args.seed + zlib.crc32(name.encode()))
        out[name] = (str(path), {0: atoms})
    for spec in args.traj or ():
        name, _, path = spec.partition("=")
        frames = {}
        trajectory = read(path, index=":")
        for index in frame_indices(args.frames, len(trajectory)):
            atoms = trajectory[index]
            atoms.calc = None
            atoms.set_constraint()
            frames[index] = atoms
        out[name] = (path, frames)
    return out


def ase_side(args, loaded, frames, meta, minima):
    import numpy as np
    from ase.optimize import FIRE, LBFGS

    from fastmace.ase.calculator import FastMACECalculator

    calc = FastMACECalculator(loaded, precision=args.precision, dispersion=dispersion(args))
    warm = False  # the process's first relaxation carries the compile
    for name, (path, by_index) in frames.items():
        record = meta["structures"].setdefault(name, {"traj": path, "frames": {}})
        for index, atoms in by_index.items():
            if str(index) in record["frames"]:  # --resume
                continue
            relaxed = atoms.copy()
            relaxed.set_momenta(np.zeros((len(relaxed), 3)))
            relaxed.calc = calc
            started = time.perf_counter()
            initial = float(relaxed.get_potential_energy())
            target = relaxed
            if args.cell_filter == "frechet":
                from ase import units
                from ase.filters import FrechetCellFilter

                target = FrechetCellFilter(relaxed, scalar_pressure=args.pressure_GPa * units.GPa)
            run = (LBFGS if args.optimizer == "lbfgs" else FIRE)(target, logfile=None)
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
                      "fmax_eV_A": float(np.linalg.norm(target.get_forces(), axis=1).max()),
                      "ms_per_step": 1000 * seconds / max(steps, 1),
                      "switch_flips": flips if args.optimizer == "fire" else None, "warm_up": not warm}
            warm = True
            if target is not relaxed:
                result.update(cell_result(relaxed.cell.array))
            record["frames"][str(index)] = result
            minima[f"{name}/{index}"] = relaxed.positions.copy()
            minima[f"{name}/{index}/cell"] = relaxed.cell.array.copy()
            print(f"{name} frame {index} ASE {args.optimizer}: {steps} steps {seconds:.1f} s "
                  f"E {result['energy_eV_atom']:.6f} eV/atom flips {result['switch_flips']}", flush=True)
            yield


def gpu_side(args, loaded, frames, meta, minima):
    import gc

    import numpy as np
    import torch
    from ase import units

    import fastmace as fm
    from fastmace.config import NeighbourConfig
    from fastmace.model.forcefield import ForceField

    compare = json.loads(args.compare.read_text()) if args.compare else None
    reference = dict(np.load(args.compare.with_suffix(".npz"))) if args.compare else {}
    record_class = fm.optimize.LBFGS if args.optimizer == "lbfgs" else fm.optimize.FIRE
    variable_cell = args.cell_filter == "frechet"
    optimizer_record = (record_class(cell_filter=fm.optimize.FrechetCell(scalar_pressure_GPa=args.pressure_GPa))
                        if variable_cell else record_class())

    def simulation(atoms, mode, skin):
        return fm.Simulation(atoms.copy(), ForceField(loaded, atoms, precision=args.precision,
                                                      dispersion=dispersion(args)),
                             optimizer=optimizer_record, seed=0, mode=None if mode == "graph" else mode,
                             neighbours=NeighbourConfig(skin=skin))

    for name, (path, by_index) in frames.items():
        record = meta["structures"].setdefault(name, {"traj": path, "runs": {}})
        first = next(iter(by_index.values()))
        for mode in args.modes:
            for skin in args.skins:
                key = run_key(mode, skin)
                done = record["runs"].get(key, {}).get("frames", {})
                if all(str(index) in done for index in by_index):  # --resume
                    continue
                started = time.perf_counter()
                sim = simulation(first, mode, skin)
                run = record["runs"].setdefault(key, {"setup_s": time.perf_counter() - started,
                                                      "tier": sim.plan.tier, "skin_A": skin, "frames": {}})
                run["planned_skin_A"] = sim.plan.neighbours.skin
                optimizer = sim._engine.optimizer
                warm = False  # the engine's first relaxation carries its captures
                for index, atoms in by_index.items():
                    if str(index) in run["frames"]:  # --resume
                        continue
                    if variable_cell:  # no host cell write on a variable-cell engine: one Simulation per frame
                        if atoms is not first:
                            sim.close()
                            sim = simulation(atoms, mode, skin)
                            optimizer = sim._engine.optimizer
                            warm = False
                    else:
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
                    result["warm_up"], warm = not warm, True
                    if variable_cell:  # the summary's energies are E + pV (ASE's filter energy): take pV off
                        pv = args.pressure_GPa * units.GPa * summary["volume_A3"] / len(atoms)
                        result["enthalpy_eV_atom"] = result["energy_eV_atom"]
                        result["energy_eV_atom"] -= pv
                        result.update(cell_result(relaxed.cell.array))
                    if compare is not None and str(index) in compare["structures"].get(name, {}).get("frames", {}):
                        ase = compare["structures"][name]["frames"][str(index)]
                        result["dE_vs_ase_meV_atom"] = 1000 * (result["energy_eV_atom"] - ase["energy_eV_atom"])
                        result["dsteps_vs_ase"] = summary["steps"] - ase["steps"]
                        ase_cell = reference.get(f"{name}/{index}/cell", atoms.cell.array)
                        result["max_distance_vs_ase_A"] = mic_distance(ase_cell, reference[f"{name}/{index}"],
                                                                       relaxed.positions)
                        if variable_cell:
                            result["max_cell_length_diff_A"] = max(
                                abs(a - b) for a, b in zip(result["cell_lengths_A"], ase["cell_lengths_A"]))
                            result["volume_rel_diff"] = result["volume_A3"] / ase["volume_A3"] - 1.0
                    run["frames"][str(index)] = result
                    minima[f"{name}/{key}/{index}"] = relaxed.positions.copy()
                    minima[f"{name}/{key}/{index}/cell"] = relaxed.cell.array.copy()
                    print(f"{name} frame {index} {key}: {result['steps']} steps {result['seconds']:.2f} s "
                          f"({result['ms_per_step']:.2f} ms/step) rebuilds {result['neighbour_rebuilds']} "
                          f"recaptures {result['recaptures']} flips {result['switch_flips']} "
                          f"dE {result.get('dE_vs_ase_meV_atom', float('nan')):.4f} meV/atom "
                          f"dmax {result.get('max_distance_vs_ase_A', float('nan')):.4f} A"
                          f" dcell {result.get('max_cell_length_diff_A', float('nan')):.2e} A"
                          f" dV {result.get('volume_rel_diff', float('nan')):.2e}", flush=True)
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
    head = None if args.head == "none" else args.head
    loaded = loading.load_pristine(args.model, head, checkpoint=str(checkpoint))
    frames = read_frames(args)
    previous, minima = load_previous(args.output) if args.resume else (None, {})
    meta = {"schema": 4, "task": "4-T05/4-T06/4-T07/4-T08", "side": args.side, "optimizer": args.optimizer,
            "cell_filter": args.cell_filter, "pressure_GPa": args.pressure_GPa, "dispersion": args.dispersion,
            "structure_settings": None if not args.structures else {
                "root": str(args.structures_root), "repeat": args.repeat, "strain": args.strain,
                "rattle_A": args.rattle, "seed": args.seed, "seed_rule": "seed + crc32(name)"},
            "optimizer_settings": "ASE 3.27 defaults", "cuda_allocator": allocator,
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "fastmace_commit": git_commit(),
            "machine": socket.gethostname(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "ase": __import__("ase").__version__, "precision": args.precision, "oeq_runtime": oeq,
            "model": args.model, "head": args.head, "checkpoint": str(checkpoint), "fmax": args.fmax,
            "steps": args.steps, "structures": {}}
    if previous is not None:
        if previous["side"] != args.side or previous["optimizer"] != args.optimizer:
            raise SystemExit(f"--resume: {args.output} is a {previous['side']} {previous['optimizer']} run")
        meta["structures"] = previous["structures"]
        meta["resumed"] = previous.get("resumed", []) + [meta["started"]]
        meta["started"] = previous["started"]
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
