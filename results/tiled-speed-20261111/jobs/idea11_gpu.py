"""Idea 11 on the GPU (float32, OMAT-0 medium): SrTiO3 6x6x6 at the origin and moved by +200 A (the whole-system
A100-80 limit's coordinates, about 57^3 = 226 A) and +1,500 A (capacity scale), perfect and rattled.
For the whole-system call and the tiled call (mode padded, R 256) with TILE_LOCAL_FRAME off and on: the largest force
component on the perfect crystal, and the rattled forces' largest difference from the origin run (relative).
Usage: idea11_gpu.py --output DIR
"""
import fastmace  # noqa: F401  (before torch)
import argparse
import json
import os
import sys

import numpy as np
import torch

import fastmace as fm
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
for offset in (0.0, 200.0, 1500.0):
    for kind, atoms0 in base.items():
        atoms = atoms0.copy()
        atoms.positions = atoms.positions + offset
        user = fm.ForceField(model, atoms, precision="float32")
        planner = minimal.default_planner(user)
        field = planner.build_evaluation(planner.plan_evaluation(user, atoms), user, atoms)
        results = {"whole": field.evaluate(atoms, stress=True)}
        for frame in (False, True):
            tiled_module.TILE_LOCAL_FRAME = frame
            tiled = TiledForceField(field, TileConfig(256, mode="padded", budget_receivers=()))
            tiled.evaluate(atoms, stress=True)
            results[f"tiled-frame-{'on' if frame else 'off'}"] = tiled.evaluate(atoms, stress=True)
            assert tiled.last.get("local_frame") == ("tiles" if frame else None), tiled.last.get("local_frame")
        for path, (energy, forces, stress) in results.items():
            forces = np.asarray(forces, np.float64)
            row = dict(offset=offset, kind=kind, path=path, energy=float(energy), max_force=float(np.abs(forces).max()))
            if offset == 0.0:
                reference[(kind, path)] = forces
            else:
                ref = reference[(kind, path)]
                row["max_diff_from_origin"] = float(np.abs(forces - ref).max())
                row["rel_diff_from_origin"] = row["max_diff_from_origin"] / float(np.abs(ref).max())
            rows.append(row)
            print(json.dumps(row), flush=True)
        del field, user
        torch.cuda.empty_cache()
with open(os.path.join(args.output, "idea11-gpu.json"), "w") as f:
    json.dump(rows, f, indent=1)
print("summary")
for row in rows:
    print(f"{row['offset']:>7.0f} {row['kind']:>8} {row['path']:>15} max|F| {row['max_force']:.3e} "
          f"diff {row.get('max_diff_from_origin', 0):.3e} rel {row.get('rel_diff_from_origin', 0):.2e}")
