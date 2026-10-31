"""Offline refit of rule 12: counted edges (the crystal's pairs within r_max, plus the edge bucket) instead of the
uniform-density count, and the memory-rescue records as fractions of the remaining activations they keep.
    python refit2.py FM_SOURCE MATRIX_CASES.json
"""

import json
import math
import sys
from pathlib import Path

FM = Path(sys.argv[1])
sys.path.insert(0, str(FM))
sys.path.insert(0, str(FM / "tools" / "bench"))
import capacity_scan  # noqa: E402
from ase.neighborlist import neighbor_list  # noqa: E402

from fastmace.model.capabilities import ModelDims  # noqa: E402
from fastmace.planner import memory, minimal  # noqa: E402

DATA = FM / "tests/_support/data"
FIT = json.loads((DATA / "rule12_fit.json").read_text())
CAP = json.loads((DATA / "rule12_capacity.json").read_text())
MAT = json.loads(Path(sys.argv[2]).read_text())
registry = minimal._rewrite_registry()
BUCKET = 0.05
RESCUE = ("recompute_conv_weights", "recompute_first_table", "recompute_node_segments", "recompute_node_region",
          "gate_epilogue")
# Keep fractions (MH-1, the matrix: NbCl5 and NaCl, 21,600-121,824 atoms).
KEEP = {"recompute_conv_weights": 0.665, "recompute_first_table": 0.967, "recompute_node_segments": 0.653,
        "recompute_node_region": 0.69}
GE = {"none": 1.04, "segments": 1.176, "region": 1.0}

per_atom_cache = {}


def counted_per_atom(structure, r):
    key = (structure, round(r, 4))
    if key not in per_atom_cache:
        _, cell = capacity_scan.base_cell(structure)
        cell.set_pbc(True)
        per_atom_cache[key] = len(neighbor_list("i", cell, r)) / len(cell)
    return per_atom_cache[key]


def volume_per_atom(structure):
    _, cell = capacity_scan.base_cell(structure)
    return cell.get_volume() / len(cell)


def r_from_uniform(case):
    n, edges = case["nodes"], case["edges"]
    v = volume_per_atom(case["structure"]) * n
    # edges = 2 * n^2 (4/3 pi r^3) / (2 V) * (1 + bucket), rounded up
    return (edges / (1 + BUCKET) * v / (n * n * 4.0 / 3.0 * math.pi)) ** (1 / 3)


def estimate(case, dims, edges, safety, keep_model):
    names = case["rewrites"]
    additive = [n for n in names if not (keep_model and n in RESCUE)]
    saved = memory.SAFETY
    memory.SAFETY = safety
    try:
        e = memory.estimate(dims, edges=edges, nodes=case["nodes"], dtype=case["dtype"], capacity=case["capacity"],
                            list_dtype=case["dtype"], fused_conv=case["fused_conv"],
                            rewrite_terms=memory.rewrite_memory_terms(registry, dims, additive, case["dtype"]),
                            selfcheck_terms=memory.selfcheck_memory_terms(registry, dims, case["selfchecks"],
                                                                          case["dtype"], case["sms"]),
                            resident_terms=memory.resident_memory_terms(registry, dims, case["resident"],
                                                                        case["dtype"]))
    finally:
        memory.SAFETY = saved
    if not keep_model:
        return e.total
    terms = dict(e.terms)
    remaining = terms["activations"] + terms["rewrites"]
    factor = 1.0
    for n in names:
        factor *= KEEP.get(n, 1.0)
    if "gate_epilogue" in names:
        factor *= GE["region" if "recompute_node_region" in names else
                      "segments" if "recompute_node_segments" in names else "none"]
    return e.total - remaining + remaining * factor


def run(safety, keep_model, counted):
    groups = {}
    for group, cases in (("fit", FIT["cases"]), ("capacity", CAP), ("matrix", MAT)):
        for c in cases:
            dims = ModelDims.from_record(FIT["dims"][c["label"]])
            edges = c["edges"]
            if counted:
                r = r_from_uniform(c)
                edges = int(math.ceil(counted_per_atom(c["structure"], r) * c["nodes"] * (1 + BUCKET)))
            ratio = estimate(c, dims, edges, safety, keep_model) / c["peak_bytes"]
            key = group if group != "matrix" else f"matrix {c.get('stage')}"
            groups.setdefault(key, []).append(ratio)
    return groups



import numpy as np  # noqa: E402

FACTORS = ["recompute_conv_weights", "recompute_first_table", "recompute_node_segments", "recompute_node_region",
           "ge_none", "ge_segments"]


def parts(c, dims, edges, safety):
    names = c["rewrites"]
    additive = [n for n in names if n not in RESCUE]
    saved = memory.SAFETY
    memory.SAFETY = safety
    try:
        e = memory.estimate(dims, edges=edges, nodes=c["nodes"], dtype=c["dtype"], capacity=c["capacity"],
                            list_dtype=c["dtype"], fused_conv=c["fused_conv"],
                            rewrite_terms=memory.rewrite_memory_terms(registry, dims, additive, c["dtype"]),
                            selfcheck_terms=memory.selfcheck_memory_terms(registry, dims, c["selfchecks"],
                                                                          c["dtype"], c["sms"]),
                            resident_terms=memory.resident_memory_terms(registry, dims, c["resident"], c["dtype"]))
    finally:
        memory.SAFETY = saved
    terms = dict(e.terms)
    remaining = terms["activations"] + terms["rewrites"]
    row = [1.0 if f in names else 0.0 for f in FACTORS[:4]]
    ge = "gate_epilogue" in names and "recompute_node_region" not in names
    row += [1.0 if ge and "recompute_node_segments" not in names else 0.0,
            1.0 if ge and "recompute_node_segments" in names else 0.0]
    return e.total - remaining, remaining, row


def counted_edges(c):
    r = r_from_uniform(c)
    return int(math.ceil(counted_per_atom(c["structure"], r) * c["nodes"] * (1 + BUCKET)))


for safety in (0.95, 0.96, 0.97, 1.0):
    dims = ModelDims.from_record(FIT["dims"]["mh1-r2scan"])
    data = [(c, *parts(c, dims, counted_edges(c), safety)) for c in MAT]
    base = [fixed + rem for c, fixed, rem, row in data if c["stage"] == "s0-default"]
    target = np.mean([b / c["peak_bytes"] for b, (c, *_ ) in zip(base, [d for d in data if d[0]["stage"] == "s0-default"])])
    A, y = [], []
    for c, fixed, rem, row in data:
        if c["stage"] == "s0-default":
            continue
        want = (target * c["peak_bytes"] - fixed) / rem
        A.append(row)
        y.append(math.log(want))
    A, y = np.array(A), np.array(y)
    # ge_none is not separable from the default on the matrix (every stage up to s3 has it): fix it at 1.04
    A_fit = A[:, [0, 1, 2, 3, 5]]
    y_fit = y - A[:, 4] * math.log(1.04)
    sol, *_ = np.linalg.lstsq(A_fit, y_fit, rcond=None)
    keep = dict(zip(["conv", "table", "segments", "region", "ge_segments"], np.exp(sol).round(3)))
    # ratios with the fitted factors, ge_none 1.04 also on the default
    out = {}
    for c, fixed, rem, row in data:
        logf = sum(r * s for r, s in zip(row[:4] + [row[5]], sol)) + row[4] * math.log(1.04)
        ratio = (fixed + rem * math.exp(logf)) / c["peak_bytes"]
        out.setdefault(c["stage"], []).append(ratio)
    g = run(safety, False, True)
    print(f"SAFETY {safety}: target {target:.3f} keep {keep} | fit {min(g['fit']):.2f}-{max(g['fit']):.2f} "
          f"capacity {min(g['capacity']):.2f}-{max(g['capacity']):.2f} | "
          + "  ".join(f"{k} {min(v):.2f}-{max(v):.2f}" for k, v in sorted(out.items())), flush=True)
