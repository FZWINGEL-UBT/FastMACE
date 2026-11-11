"""Idea 11 on the GPU, round 2 (float32, OMAT-0 medium): SrTiO3 6x6x6 at the origin, +200 A and +1,500 A, perfect
and rattled. Paths: the whole-system call with WHOLE_SYSTEM_FLOAT64_EDGES off and on, and the tiled call (mode
padded, R 256) with TILE_LOCAL_FRAME on. Per row: the largest force component (perfect: the residual), and against
the same path's origin run the forces' and the stress's largest difference (relative).
Usage: idea11_gpu2.py --output DIR
"""
import fastmace  # noqa: F401  (before torch)
import argparse
import json
import os
import sys

import numpy as np
import torch

import fastmace as fm
from fastmace.model import forces as forces_module
from fastmace.model import tiled as tiled_module
from fastmace.model.tiled import TileConfig, TiledForceField
from fastmace.planner import minimal

sys.path.insert(0, "tools/bench")
import tiled_capacity  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
args = p.parse_args()
os.makedirs(args.output, exist_ok=True)
torch.cuda.set_per_process_memory_fraction(0.9)
base = {"perfect": tiled_capacity.supercell("SrTiO3", (6, 6, 6), 0.0, 0.0),
        "rattled": tiled_capacity.supercell("SrTiO3", (6, 6, 6), 0.05, 0.0)}
model = fm.load("omat0-medium")
rows, reference = [], {}
tiled_module.TILE_LOCAL_FRAME = True
for offset in (0.0, 200.0, 1500.0):
    for kind, atoms0 in base.items():
        atoms = atoms0.copy()
        atoms.positions = atoms.positions + offset
        results = {}
        for switch in (False, True):
            forces_module.WHOLE_SYSTEM_FLOAT64_EDGES = switch
            user = fm.ForceField(model, atoms, precision="float32")
            planner = minimal.default_planner(user)
            field = planner.build_evaluation(planner.plan_evaluation(user, atoms), user, atoms)
            assert field.float64_edges is switch, field.float64_edges
            results[f"whole-f64-{'on' if switch else 'off'}"] = field.evaluate(atoms, stress=True)
            if not switch:
                tiled = TiledForceField(field, TileConfig(256, mode="padded", budget_receivers=()))
                tiled.evaluate(atoms, stress=True)
                results["tiled-frame-on"] = tiled.evaluate(atoms, stress=True)
                assert tiled.last.get("local_frame") == "tiles", tiled.last.get("local_frame")
                del tiled
            del field, user
            torch.cuda.empty_cache()
        for path, (energy, forces, stress) in results.items():
            forces, stress = np.asarray(forces, np.float64), np.asarray(stress, np.float64)
            row = dict(offset=offset, kind=kind, path=path, energy=float(energy),
                       max_force=float(np.abs(forces).max()), max_stress=float(np.abs(stress).max()))
            if offset == 0.0:
                reference[(kind, path)] = (energy, forces, stress)
            else:
                e0, f0, s0 = reference[(kind, path)]
                row["energy_diff"] = float(energy - e0)
                row["max_diff_from_origin"] = float(np.abs(forces - f0).max())
                row["rel_diff_from_origin"] = row["max_diff_from_origin"] / float(np.abs(f0).max())
                row["stress_diff_from_origin"] = float(np.abs(stress - s0).max())
            rows.append(row)
            print(json.dumps(row), flush=True)
with open(os.path.join(args.output, "idea11-gpu2.json"), "w") as f:
    json.dump(rows, f, indent=1)
print("summary")
for row in rows:
    print(f"{row['offset']:>7.0f} {row['kind']:>8} {row['path']:>15} max|F| {row['max_force']:.3e} "
          f"diff {row.get('max_diff_from_origin', 0):.3e} rel {row.get('rel_diff_from_origin', 0):.2e} "
          f"dE {row.get('energy_diff', 0):.2e} dS {row.get('stress_diff_from_origin', 0):.2e}")
