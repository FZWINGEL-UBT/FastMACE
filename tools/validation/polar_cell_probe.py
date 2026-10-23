"""PolarMACE under cell changes on ``fm.calc`` (plan task 4-T12b; 4-T07's diverging LBFGS with the cell filter).

4-T07: ASE's ``LBFGS(FrechetCellFilter(atoms))`` on ``fm.calc`` with POLAR-1-S on NbCl5 3x3x3 (hybrid float32)
rose from -10694.843 to -10693.860 eV/atom in 5,000 steps. This probe starts from the same configuration
(``tools/bench/optimizer_frames.py``: the benchmark cell repeated, scaled by 1 + strain, rattled with seed
seed + crc32(name)) and records, on one ``FastMACECalculator`` (the compiled tier, as 4-T07):

- ``lbfgs``: the first ``--lbfgs-steps`` steps of the same relaxation: energy, filter fmax, smallest
  interatomic distance, volume, the included k-set size (graph_longrange's rule on the host), and whether
  the energy rose;
- ``forces``: at the start and at the first step whose energy rose, -dE/dt along the normalised forces by
  central differences (t = +-``--force-step`` Angstrom) against |F|;
- ``scans``: E and the stress along hydrostatic and xx strains in +-``--scan-max`` (``--scan-points``), central
  differences of E against V sigma at every interior point, and the k-set size at each strain;
- ``fresh``: at the start and at the first rising step, the reused calculator against a fresh
  ``FastMACECalculator`` built at that configuration (energy, forces, stress), for stale state.

Usage (the local GPU through tools/ci/local_gpu.py run, mace_cueq_oeq):
  python tools/validation/polar_cell_probe.py --output results/RUN/probe.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

# isort: split
import numpy as np

HERE = Path(__file__).resolve().parent


def _fd_tool():
    spec = importlib.util.spec_from_file_location("_polar_stress_fd", HERE.parent / "polar_stress_fd.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="polar1-s")
    p.add_argument("--structure", default="NbCl5")
    p.add_argument("--repeat", nargs=3, type=int, default=[3, 3, 3])
    p.add_argument("--strain", type=float, default=0.02)
    p.add_argument("--rattle", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision", default="float32")
    p.add_argument("--lbfgs-steps", type=int, default=300)
    p.add_argument("--force-step", type=float, default=1e-3)
    p.add_argument("--scan-max", type=float, default=2e-3)
    p.add_argument("--scan-points", type=int, default=21)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args(argv)


def start_atoms(args):
    from ase.io import read

    root = Path(os.environ.get("FASTMACE_STRUCTURES", "."))
    atoms = read(root / "Structures" / "Benchmark" / "R2SCAN" / f"{args.structure}.cif").repeat(tuple(args.repeat))
    atoms.set_cell(atoms.cell.array * (1.0 + args.strain), scale_atoms=True)
    atoms.rattle(args.rattle, seed=args.seed + zlib.crc32(args.structure.encode()))
    return atoms


def min_distance(atoms):
    from ase.neighborlist import neighbor_list

    distances = neighbor_list("d", atoms, 3.0)
    return float(distances.min()) if len(distances) else None


def evaluate(calc, atoms):
    atoms = atoms.copy()
    atoms.calc = calc
    energy = float(atoms.get_potential_energy())
    return energy, atoms.get_forces().copy(), atoms.get_stress(voigt=False).copy()


def force_check(calc, atoms, step):
    """-dE/dt along d = F/|F| by central differences, against |F| (they are equal for consistent forces)."""
    energy, forces, _ = evaluate(calc, atoms)
    norm = float(np.linalg.norm(forces))
    direction = forces / norm
    values = {}
    for sign in (1, -1):
        moved = atoms.copy()
        moved.positions += sign * step * direction
        values[sign] = evaluate(calc, moved)[0]
    derivative = -(values[1] - values[-1]) / (2 * step)
    return {"force_norm_eV_A": norm, "fd_eV_A": derivative, "relative_difference": (derivative - norm) / norm,
            "energy_eV": energy, "step_A": step}


def scan(calc, atoms, kind, scan_max, points, fd):
    """E and stress along a strain path; central differences of E against V sigma; k-set size per strain."""
    strains = np.linspace(-scan_max, scan_max, points)
    rows = []
    for e in strains:
        strain = np.identity(3) * e if kind == "hydrostatic" else np.diag([e, 0.0, 0.0])
        moved = atoms.copy()
        moved.set_cell(atoms.cell.array @ (np.identity(3) + strain), scale_atoms=True)
        energy, _, stress = evaluate(calc, moved)
        kset = sum(1 for included, _ in fd.included_kset(moved.cell.array).values() if included)
        analytic = float(np.trace(stress)) if kind == "hydrostatic" else float(stress[0, 0])
        rows.append({"strain": float(e), "energy_eV": energy, "volume_A3": float(moved.get_volume()),
                     "stress_eV_A3": stress.tolist(), "dE_de_analytic_eV": analytic * float(moved.get_volume()),
                     "k_vectors": kset})
    for index in range(1, len(rows) - 1):
        de = rows[index + 1]["strain"] - rows[index - 1]["strain"]
        # E(cell (I + e)) with the reference cell fixed: dE/de = V(e) sigma (I + e)^-1 terms; at |e| <= 2e-3 the
        # relative correction is below 2e-3, far under what this looks for.
        rows[index]["dE_de_fd_eV"] = (rows[index + 1]["energy_eV"] - rows[index - 1]["energy_eV"]) / de
    return rows


def fresh_check(calc, atoms, factory):
    reused = evaluate(calc, atoms)
    fresh = evaluate(factory(), atoms)
    return {"energy_eV": [reused[0], fresh[0]], "energy_difference_eV": reused[0] - fresh[0],
            "max_force_difference_eV_A": float(np.abs(reused[1] - fresh[1]).max()),
            "max_stress_difference_eV_A3": float(np.abs(reused[2] - fresh[2]).max())}


def main(argv=None):
    args = parse_args(argv)
    from ase.filters import FrechetCellFilter
    from ase.optimize import LBFGS

    from fastmace.ase.calculator import FastMACECalculator
    from fastmace.model import loading

    fd = _fd_tool()
    loaded = loading.load_pristine(args.model)

    def factory():
        return FastMACECalculator(loaded, precision=args.precision)

    calc = factory()
    atoms = start_atoms(args)
    out = {"tool": "tools/validation/polar_cell_probe.py", "args": {k: str(v) for k, v in vars(args).items()},
           "started": datetime.now(timezone.utc).isoformat(),
           "fastmace_commit": os.environ.get("FASTMACE_SOURCE_COMMIT"), "natoms": len(atoms)}
    try:
        import torch

        out["gpu"] = torch.cuda.get_device_name()
    except Exception:  # noqa: BLE001 - informational
        out["gpu"] = None
    start = atoms.copy()
    out["forces_start"] = force_check(calc, start, args.force_step)
    out["fresh_start"] = fresh_check(calc, start, factory)
    out["scans_start"] = {kind: scan(calc, start, kind, args.scan_max, args.scan_points, fd)
                          for kind in ("hydrostatic", "xx")}
    relaxed = start.copy()
    relaxed.calc = calc
    target = FrechetCellFilter(relaxed)
    run = LBFGS(target, logfile=None)
    steps, previous, first_rise = [], None, None
    started = time.perf_counter()
    for index, _ in enumerate(run.irun(fmax=0.01, steps=args.lbfgs_steps)):
        energy = float(relaxed.get_potential_energy())
        fmax = float(np.linalg.norm(target.get_forces(), axis=1).max())
        row = {"step": index, "energy_eV": energy, "fmax_eV_A": fmax,
               "volume_A3": float(relaxed.get_volume()), "min_distance_A": min_distance(relaxed),
               "k_vectors": sum(1 for included, _ in fd.included_kset(relaxed.cell.array).values() if included),
               "rose": previous is not None and energy > previous}
        steps.append(row)
        if row["rose"] and first_rise is None:
            first_rise = (index, relaxed.copy())
        previous = energy
    out["lbfgs"] = {"steps": steps, "seconds": time.perf_counter() - started,
                    "first_rise": None if first_rise is None else first_rise[0]}
    if first_rise is not None:
        bad = first_rise[1]
        out["forces_first_rise"] = force_check(calc, bad, args.force_step)
        out["fresh_first_rise"] = fresh_check(calc, bad, factory)
    final = relaxed.copy()
    out["forces_final"] = force_check(calc, final, args.force_step)
    out["fresh_final"] = fresh_check(calc, final, factory)
    out["finished"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({key: out[key] for key in out if key.startswith(("forces", "fresh"))}, indent=1))
    print("lbfgs first rise:", out["lbfgs"]["first_rise"], "last:", steps[-1] if steps else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
