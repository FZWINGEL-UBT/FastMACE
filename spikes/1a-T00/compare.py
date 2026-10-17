"""Compare probe runs against a reference run: forces, energy and evidence fields.

Usage: compare.py REFERENCE_RUN RUN [RUN ...]   (run ids under ~/fastmace-spike-cache)
"""
import json
import sys
from pathlib import Path

import numpy as np

base = Path.home() / "fastmace-spike-cache"


def load(run):
    out = base / run / "out"
    result = json.loads((out / "result.json").read_text())
    forces = np.load(out / "forces.npy") if (out / "forces.npy").is_file() else None
    positions = np.load(out / "positions.npy") if (out / "positions.npy").is_file() else None
    return result, forces, positions


reference, ref_forces, ref_positions = load(sys.argv[1])
for run in sys.argv[1:]:
    result, forces, positions = load(run)
    row = {"run": run, "mode": result["mode"], "ok": result["ok"], "BUILT_EXTENSION": result.get("BUILT_EXTENSION"),
           "oeq_import_s": result.get("oeq_import_s"), "wall_s": result.get("wall_s"),
           "nvcc_attempts": len(result.get("nvcc_attempts") or []),
           "extension_in_fresh_cache": str(result.get("extension_so", "")).startswith(str(base / run)),
           "energy_eV": result.get("energy_eV")}
    if forces is not None and ref_forces is not None:
        assert np.array_equal(positions, ref_positions)
        diff = np.abs(forces - ref_forces)
        row["max_abs_dforce"] = float(diff.max())
        row["bitwise_equal"] = bool(np.array_equal(forces, ref_forces))
        row["dE_eV"] = result["energy_eV"] - reference["energy_eV"]
    row["loaded"] = [Path(p).name + " <- " + str(Path(p).parent) for p in result.get("loaded_libraries", [])]
    print(json.dumps(row, indent=1))
