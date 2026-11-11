"""Diagnose TILE_LOCAL_FRAME at large receiver caps (job 58: R 16,384 at 20^3 forces 1.2e-4 eV/A off).

Per receiver cap: the tiled call (mode padded, frame on, cap fixed) with _slot_local wrapped to record per slot the
largest |row|, the largest |remaining shift|, the edges with a nonzero remaining shift, and the largest error of the
float32 vectors against float64 ones; then the largest force on the perfect crystal.
Usage: python diag_frame.py REPEAT R [R ...]
"""
import json
import sys

sys.path.insert(0, "tools")
import fastmace  # noqa: F401  (before torch)
import torch

import fastmace as fm
from fastmace.model import tiled as tm
from fastmace.model.tiled import TileConfig, TiledForceField
from fastmace.planner import minimal

sys.path.insert(0, "tools/bench")
import tiled_capacity  # noqa: E402

repeat = tuple(int(x) for x in sys.argv[1].split("x"))
caps = [int(r) for r in sys.argv[2:]]
atoms = tiled_capacity.supercell("SrTiO3", repeat, 0.0, 0.0)
stats = []
plain = tm._slot_local_vectors


def wrapped(positions64, nodes, node_mask, local, edge_mask, shifts, cell, inverse, receivers, dtype, far):
    v32 = plain(positions64, nodes, node_mask, local, edge_mask, shifts, cell, inverse, receivers, dtype, far)
    rows, remaining = tm._slot_local(positions64, nodes, node_mask, local, edge_mask, shifts, cell, inverse,
                                     receivers, dtype)
    m = edge_mask > 0
    e64 = positions64[nodes[local[1]]] - positions64[nodes[local[0]]] + torch.round(shifts.double() @ inverse) @ cell
    nonzero = (remaining.abs().sum(1) > 0) & m
    error = (v32.double() - e64)[m].abs()
    stats.append(dict(receivers=int(receivers), rows_max=float(rows.abs().max()),
                      remaining_max=float(remaining[m].abs().max()), remaining_edges=int(nonzero.sum()),
                      edges=int(m.sum()), vec_err=float(error.max()),
                      vec_err_remaining=float((v32.double() - e64)[nonzero].abs().max()) if nonzero.any() else 0.0))
    return v32


tm._slot_local_vectors = wrapped
tm.TILE_LOCAL_FRAME = True
tm.TILE_EPILOGUE = False  # the wrapper runs eagerly
user = fm.ForceField(fm.load("omat0-medium"), atoms, precision="float32")
planner = minimal.default_planner(user)
plan = planner.plan_evaluation(user, atoms)
out = {}
for cap in caps:
    with tiled_capacity.no_global_list(True):
        field = planner.build_evaluation(plan, user, atoms, call_rows=cap, release_node_attrs=True,
                                         release_whole_system=True, global_list=False)
        tiled = TiledForceField(field, TileConfig(cap, None, mode="padded", neighbours="tiles", budget_receivers=(),
                                                  species="index"))
        positions = field._prepare_one_off(atoms)
    stats.clear()
    result = tiled(positions)
    out[cap] = dict(max_force=float(result.forces.abs().max()), local_frame=tiled.last.get("local_frame"),
                    slots=[dict(s) for s in stats])
    worst = max(stats, key=lambda s: s["vec_err"])
    print(cap, "max|F|", out[cap]["max_force"], "frame", out[cap]["local_frame"], "slot calls", len(stats),
          "worst", json.dumps(worst), flush=True)
    del tiled, field, result
    torch.cuda.empty_cache()
print(json.dumps(out))
