"""Gate and setup time of ``generic_radial_tables`` on a generic model (MACE-OMAT-0 medium by default).

Per structure (rattled and strained supercells), three force fields through the minimal planner on the same
configuration: float32 with the generic radial tables (``RewriteConfig(radial_tables="on")``), float32 without them
(the untabled model it replaces) and float64 without them (the reference). Reported: energy per atom, forces and
stress of each float32 side against float64, and tables against the network. The tables' certificate and pieces
come from the force field's applied records.

Setup time: the tabled side is built twice, first into an empty table cache (``--cache``: fit and certificate) and
again from it (cached), each the build and first evaluation, against the untabled float32 side's.

Run as ``run.sh NAME results/generic-radial-20261102/gate.py --cache DIR``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)


def structures(rattle, strain, seed):
    import numpy as np
    from ase import Atoms
    from ase.build import bulk

    srtio3 = Atoms("SrTiO3", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5),
                                               (0, 0.5, 0.5)], cell=[3.905] * 3, pbc=True).repeat((6, 6, 6))
    nacl = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((4, 4, 4))
    out = {}
    for name, atoms in (("SrTiO3", srtio3), ("NaCl", nacl)):
        rng = np.random.default_rng(seed)
        e = rng.uniform(-strain, strain, (3, 3))
        atoms.set_cell(atoms.cell.array @ (np.eye(3) + 0.5 * (e + e.T)), scale_atoms=True)
        atoms.positions = atoms.positions + rng.normal(0.0, rattle, atoms.positions.shape)
        out[name] = atoms
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="omat0-medium")
    p.add_argument("--rattle", type=float, default=0.05)
    p.add_argument("--strain", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cache", required=True, type=Path, help="An empty table cache for the setup timing")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    import torch

    import fastmace as fm
    from fastmace.config import RewriteConfig

    shutil.rmtree(args.cache, ignore_errors=True)
    args.cache.mkdir(parents=True)
    os.environ["FASTMACE_CACHE"] = str(args.cache)
    loaded = fm.load(args.model)
    record = dict(model=args.model, device=torch.cuda.get_device_name(), host=os.uname().nodename,
                  source_commit=os.environ.get("FASTMACE_SOURCE_COMMIT"), rattle=args.rattle, strain=args.strain,
                  structures={})

    def side(atoms, precision, tables):
        rewrites = RewriteConfig(radial_tables="on") if tables else "auto"
        torch.cuda.synchronize()
        started = time.perf_counter()
        field = fm.ForceField(loaded, atoms, precision=precision, rewrites=rewrites)
        energy, forces, stress = field.evaluate(atoms, stress=True)
        torch.cuda.synchronize()
        applied = [str(a) for a in getattr(field._evaluator(atoms)[1], "applied", ())]
        return dict(seconds=time.perf_counter() - started, energy=float(energy), forces=forces, stress=stress,
                    applied=applied)

    for name, atoms in structures(args.rattle, args.strain, args.seed).items():
        n = len(atoms)
        first = side(atoms, "float32", True)  # empty cache: fit and certificate
        cached = side(atoms, "float32", True)
        network = side(atoms, "float32", False)
        reference = side(atoms, "float64", False)

        def diff(a, b):
            return dict(energy_meV_per_atom=1e3 * (a["energy"] - b["energy"]) / n,
                        max_force=float(abs(a["forces"] - b["forces"]).max()),
                        max_stress=float(abs(a["stress"] - b["stress"]).max()))

        record["structures"][name] = dict(
            atoms=n, max_abs_force=float(abs(reference["forces"]).max()),
            tables_vs_network=diff(cached, network), tables_vs_float64=diff(cached, reference),
            network_vs_float64=diff(network, reference), first_vs_cached=diff(first, cached),
            setup_seconds=dict(tables_fit=first["seconds"], tables_cached=cached["seconds"],
                               network=network["seconds"], float64=reference["seconds"]),
            applied=[a for a in first["applied"] if "radial" in a])
        print(json.dumps({name: {k: v for k, v in record["structures"][name].items() if k != "applied"}}))
        print("  ", record["structures"][name]["applied"])
    record["cache_files"] = sorted(str(p.relative_to(args.cache)) for p in args.cache.rglob("*.pt"))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"gate-{args.model}.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
