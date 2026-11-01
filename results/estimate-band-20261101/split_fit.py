"""Candidate model for rule 12: separate scales for the per-edge and per-node remainders instead of one SAFETY, and
each memory-rescue record keeping a fraction of its own side only (edge: recompute_conv_weights, recompute_first_table;
node: recompute_node_segments, recompute_node_region, gate_epilogue). Fitted on MH-1 float32 (rescue matrix and
capacity cases: NaCl, NbCl5, SrTiO3) by least squares of log(estimate / peak) towards a common target ratio; then the
ratio band of every set, the fit set's other labels included.
    python split_fit.py FM_SOURCE
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

FM = Path(sys.argv[1])
sys.path.insert(0, str(FM))
from fastmace.model.capabilities import ModelDims  # noqa: E402
from fastmace.planner import memory, minimal  # noqa: E402

DATA = FM / "tests/_support/data"
FIT = json.loads((DATA / "rule12_fit.json").read_text())
CAPACITY = json.loads((DATA / "rule12_capacity.json").read_text())
RESCUE = json.loads((DATA / "rule12_rescue.json").read_text())
registry = minimal._rewrite_registry()
EDGE_RUNGS = ("recompute_conv_weights", "recompute_first_table")
NODE_RUNGS = ("recompute_node_segments", "recompute_node_region")
RESCUE = [c for c in RESCUE]


def run(case, dims, names, edges, nodes, capacity):
    dtype = case["dtype"]
    e = memory.estimate(dims, edges=edges, nodes=nodes, dtype=dtype, capacity=capacity, list_dtype=dtype,
                        fused_conv=case["fused_conv"],
                        rewrite_terms=memory.rewrite_memory_terms(registry, dims, names, dtype),
                        selfcheck_terms=memory.selfcheck_memory_terms(registry, dims, case["selfchecks"], dtype,
                                                                      case["sms"]),
                        resident_terms=memory.resident_memory_terms(registry, dims, case["resident"], dtype))
    return e.run, e.build


def pieces(case):
    """fixed (with the list), the per-edge and per-node remainders without the rescue fractions (SAFETY included),
    the build estimate, and which rungs are on."""
    dims = ModelDims.from_record(FIT["dims"][case["label"]])
    additive = [n for n in case["rewrites"] if n not in EDGE_RUNGS + NODE_RUNGS + ("gate_epilogue",)]
    e, n, c = case["edges"], case["nodes"], case["capacity"]
    fixed, build = run(case, dims, additive, 0, 0, c)
    edge = run(case, dims, additive, e, 0, c)[0] - fixed
    node = run(case, dims, additive, 0, n, c)[0] - fixed
    _, build = run(case, dims, case["rewrites"], e, n, c)
    on = set(case["rewrites"])
    return dict(fixed=fixed, edge=edge, node=node, build=build, on=on, peak=case["peak_bytes"],
                structure=case["structure"], stage=case.get("stage", "default"), label=case["label"],
                dtype=case["dtype"], atoms=case["nodes"])


NAMES = ["edge_scale", "node_scale", "conv", "table", "segments", "region", "ge", "ge_segments"]


def predict(p, x):
    se, sn, conv, table, seg, region, ge, ge_seg = x
    ke = (conv if "recompute_conv_weights" in p["on"] else 1.0) * (table if "recompute_first_table" in p["on"]
                                                                   else 1.0)
    kn = 1.0
    if "recompute_node_segments" in p["on"]:
        kn *= seg
    if "recompute_node_region" in p["on"]:
        kn *= region
    if "gate_epilogue" in p["on"] and "recompute_node_region" not in p["on"]:
        kn *= ge_seg if "recompute_node_segments" in p["on"] else ge
    run_ = p["fixed"] + se * ke * p["edge"] + sn * kn * p["node"]
    return max(run_, p["build"])


mh1 = [pieces(c) for c in RESCUE] + [pieces(c) for c in CAPACITY if c["dtype"] == "float32"]
target = math.log(1.07)


def residual(x):
    return np.array([math.log(predict(p, x) / p["peak"]) - target for p in mh1])


x0 = np.array([1.0, 1.0, 0.6, 0.97, 0.65, 0.66, 1.04, 1.17])
# gate_epilogue without the segments is on in every default point, so it is not separable from the node scale:
# fixed at the probe's 1.04 (12.3 kB per atom of about 300, docs/validation/region-norm.md).
lo = [0.3, 0.3, 0.05, 0.3, 0.1, 0.1, 1.0399, 0.9]
hi = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0401, 2.0]
fit = least_squares(residual, x0, bounds=(lo, hi))
x = fit.x
print("fitted:", ", ".join(f"{k} {v:.3f}" for k, v in zip(NAMES, x)))
groups = {}
for p in mh1:
    groups.setdefault((p["structure"], p["stage"]), []).append(predict(p, x) / p["peak"])
for k in sorted(groups):
    v = groups[k]
    print(f"  {k[0]:6s} {k[1]:13s} n={len(v):2d} {min(v):.3f}-{max(v):.3f}")
allr = [r for v in groups.values() for r in v]
print(f"MH-1 float32 band {min(allr):.3f}-{max(allr):.3f} (spread {max(allr) / min(allr):.3f})")
# the other sets with the same edge and node scales (no rescue records there)
for name, cases in (("fit (60, 14 labels)", FIT["cases"]),
                    ("capacity f64", [c for c in CAPACITY if c["dtype"] == "float64"])):
    r = [predict(pieces(c), x) / c["peak_bytes"] for c in cases]
    print(f"{name}: {min(r):.3f}-{max(r):.3f}")
Path("split_fit.json").write_text(json.dumps(dict(names=NAMES, x=list(x)), indent=1))
