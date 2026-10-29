"""Relaxation throughput: fresh engines per frame against one reused engine per composition (relax-throughput A).

Energy_Hull step 3a relaxed 430 melt-quenched glass frames (84 and 112 atoms) with variable cell, each on a
fresh engine (``relax_set.py``), and spent about 70 % of the wall time on setup. This script relaxes a
manifest of frames (CSV ``name,path``) with Energy_Hull's pass A settings (float64,
``fp64_emulation_mantissa_bits="dynamic"``, ``radial_tables="off"``, no D3, gpu-LBFGS with
``FrechetCell(scalar_pressure_GPa=0)``, fmax 0.01 eV/Å, 5000 steps) in blocks, all in one process:

- ``fresh``: a new ``ForceField`` and ``Simulation`` per frame, as ``relax_set.py`` did. Recorded per frame:
  the ``ForceField`` time, the ``Simulation`` construction with its setup parts (``plan``, ``gate``,
  ``build``, ``capture``, ``provenance``), the relaxation loop and the wall time.
- ``reused``: ``fastmace.protocols.quench.GPURelaxer``, one engine per run of frames with the same atomic
  numbers, moved to each next frame by ``Simulation.update(cell=, positions=)`` (relax-throughput A-T01).
  Recorded per frame: whether it built, the wall time, the loop, re-captures and neighbour rebuilds.

``--blocks`` orders the blocks (default ``fresh,reused,reused,fresh``: both orders in one process). Frames
are sorted by atom count and atomic numbers, so frames of one composition follow each other, as
``relax_set.py`` sorts them. Every relaxed frame's energy, cell and positions go into ``frames.npz`` so the
report compares the modes per frame against the 4-T07 relaxation limits (cell lengths 1e-3 Å, volume
0.01 %, energy 0.025 meV/atom). Output, under ``--output``: ``rows.jsonl`` (one row per frame and block,
written as it finishes), ``frames.npz`` and ``run.json`` (settings, GPU, host, versions, commit).

Run on an otherwise idle GPU (workstations: the idle check first, ResearchModule ``AGENTS.md``).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path


def _commit():
    root = Path(__file__).resolve().parents[2]
    marker = root / ".source_commit"
    if marker.exists():
        return marker.read_text(encoding="utf-8").strip()
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions():
    from importlib.metadata import PackageNotFoundError, version

    found = {}
    for package in ("torch", "mace-torch", "cuequivariance", "cuequivariance-torch", "openequivariance", "e3nn",
                    "ase", "numpy"):
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            pass
    return found


def _sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with columns name,path")
    parser.add_argument("--checkpoint", required=True, help="the MH-1 checkpoint (mace-mh-1.model)")
    parser.add_argument("--head", default="matpes_r2scan")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", default="fresh,reused,reused,fresh")
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--fp64-mantissa", default="dynamic")
    parser.add_argument("--radial-tables", choices=("off", "auto", "on"), default="off")
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=None, help="only the first N frames (smoke runs)")
    args = parser.parse_args(argv)
    blocks = [block.strip() for block in args.blocks.split(",") if block.strip()]
    if not blocks or set(blocks) - {"fresh", "reused"}:
        parser.error(f"--blocks takes fresh and reused, got {args.blocks!r}")

    import fastmace as fm  # before torch (soft file locks)
    import numpy as np
    import torch
    from ase.io import read

    from fastmace.config import EngineConfig, RewriteConfig
    from fastmace.model.forcefield import ForceField
    from fastmace.protocols.quench import GPURelaxer
    from fastmace.simulation import Simulation

    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, newline="", encoding="utf-8") as handle:
        entries = list(csv.DictReader(handle))
    frames = [(entry["name"], read(entry["path"])) for entry in entries]
    frames.sort(key=lambda item: (len(item[1]), item[1].numbers.tobytes()))
    if args.limit is not None:
        frames = frames[:args.limit]

    run = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "argv": sys.argv,
           "host": socket.gethostname(), "python": platform.python_version(), "executable": sys.executable,
           "fastmace_commit": _commit(), "versions": _versions(), "blocks": blocks,
           "frames": [name for name, _ in frames], "precision": args.precision,
           "fp64_emulation_mantissa_bits": args.fp64_mantissa, "radial_tables": args.radial_tables,
           "fmax_eV_A": args.fmax, "max_steps": args.steps, "optimizer": "gpu-LBFGS",
           "cell_filter": "FrechetCell(scalar_pressure_GPa=0)", "dispersion": None,
           "environment": {k: os.environ.get(k) for k in ("PYTHONPATH", "FASTMACE_CACHE", "FASTMACE_MODELS",
                                                           "TRITON_CACHE_DIR", "CUDA_VISIBLE_DEVICES")}}
    started = time.perf_counter()
    model = fm.load("mh1", head=args.head, checkpoint=args.checkpoint)
    run["load_s"] = time.perf_counter() - started
    run["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    (args.output / "run.json").write_text(json.dumps(run, indent=2, default=str) + "\n", encoding="utf-8")

    mantissa = int(args.fp64_mantissa) if args.fp64_mantissa.isdigit() else args.fp64_mantissa
    engine = EngineConfig(fp64_emulation_mantissa_bits=mantissa) if args.precision == "float64" else None
    rewrites = "auto" if args.radial_tables == "auto" else RewriteConfig(radial_tables=args.radial_tables)
    record = fm.optimize.LBFGS(cell_filter=fm.optimize.FrechetCell(scalar_pressure_GPa=0.0))
    arrays = {}
    rows_path = args.output / "rows.jsonl"

    def keep(row, block_index, name, relaxed, summary):
        key = f"{block_index}/{name}"
        arrays[f"{key}/cell"] = np.asarray(relaxed.cell.array)
        arrays[f"{key}/positions"] = np.asarray(relaxed.positions)
        row.update({field: summary[field] for field in ("steps", "converged", "seconds", "initial_eV_atom",
                                                   "energy_eV_atom", "fmax_eV_A", "volume_A3", "recaptures",
                                                   "neighbour_rebuilds")})
        with open(rows_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
        print(f"{time.strftime('%H:%M:%S')} {row['mode']:6s} {name} {row['natoms']} atoms: "
              f"wall {row['wall_s']:.1f} s, loop {summary['seconds']:.1f} s, {summary['steps']} steps, "
              f"E {summary['energy_eV_atom']:.6f}", flush=True)

    for block_index, mode in enumerate(blocks):
        relaxer = None
        block_started = time.perf_counter()
        for name, atoms in frames:
            row = {"block": block_index, "mode": mode, "name": name, "natoms": len(atoms),
                   "formula": atoms.get_chemical_formula()}
            _sync()
            wall = time.perf_counter()
            if mode == "fresh":
                t = time.perf_counter()
                field = ForceField(model, atoms, precision=args.precision, rewrites=rewrites)
                row["forcefield_s"] = time.perf_counter() - t
                t = time.perf_counter()
                sim = Simulation(atoms.copy(), field, optimizer=record, seed=0, engine=engine)
                _sync()
                row["simulation_s"] = time.perf_counter() - t
                row["setup_parts_s"] = dict(sim._setup_s)
                relaxed, summary = sim.relax(fmax=args.fmax, steps=args.steps)
                row["built"] = True
                row["captures"] = sim._engine.info().get("captures")
                sim.close()
            else:
                if relaxer is None:
                    relaxer = GPURelaxer(model, "gpu-LBFGS", precision=args.precision, engine=engine,
                                         cell_filter=record.cell_filter, rewrites=rewrites)
                builds = relaxer.builds
                relaxed, summary = relaxer(atoms.copy(), args.fmax, args.steps)
                row["built"] = relaxer.builds > builds
                row["captures"] = relaxer._sim._engine.info().get("captures")
            _sync()
            row["wall_s"] = time.perf_counter() - wall
            keep(row, block_index, name, relaxed, summary)
        if relaxer is not None:
            row_builds = relaxer.builds
            relaxer.close()
            print(f"block {block_index} ({mode}): {row_builds} engine builds", flush=True)
        print(f"block {block_index} ({mode}): {time.perf_counter() - block_started:.1f} s for {len(frames)} frames",
              flush=True)
        np.savez(args.output / "frames.npz", **arrays)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (args.output / "run.json").write_text(json.dumps(run, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
