"""Rule 12's residual by component (owner's item, 1 November 2026): for every case of the three fit sets (60 small,
28 capacity, 37 rescue), the estimate split into its fixed part (weights, workspace, resident buffers), the
neighbour list, the per-edge part and the per-node part (activations and additive rewrite terms, after the rescue
fractions), by evaluating it with edges, nodes or the list capacity set to zero (the estimate is linear in each).
Then the ratio estimate / measured by group, and a least-squares fit of the measured peak on the four parts per
group: a coefficient below 1 means that part is overstated.
    python residuals.py FM_SOURCE
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

FM = Path(sys.argv[1])
sys.path.insert(0, str(FM))
from fastmace.model.capabilities import ModelDims  # noqa: E402
from fastmace.planner import memory, minimal  # noqa: E402

DATA = FM / "tests/_support/data"
FIT = json.loads((DATA / "rule12_fit.json").read_text())
CAPACITY = json.loads((DATA / "rule12_capacity.json").read_text())
RESCUE = json.loads((DATA / "rule12_rescue.json").read_text())
registry = minimal._rewrite_registry()


def total(case, dims, edges, nodes, capacity):
    dtype = case["dtype"]
    e = memory.estimate(dims, edges=edges, nodes=nodes, dtype=dtype, capacity=capacity, list_dtype=dtype,
                        fused_conv=case["fused_conv"],
                        rewrite_terms=memory.rewrite_memory_terms(registry, dims, case["rewrites"], dtype),
                        selfcheck_terms=memory.selfcheck_memory_terms(registry, dims, case["selfchecks"], dtype,
                                                                      case["sms"]),
                        resident_terms=memory.resident_memory_terms(registry, dims, case["resident"], dtype))
    return e.run, e.build


def parts(case):
    dims = ModelDims.from_record(FIT["dims"][case["label"]])
    e, n, c = case["edges"], case["nodes"], case["capacity"]
    run, build = total(case, dims, e, n, c)
    fixed, _ = total(case, dims, 0, 0, 0)
    with_list, _ = total(case, dims, 0, 0, c)
    edge_part = total(case, dims, e, 0, 0)[0] - fixed
    node_part = total(case, dims, 0, n, 0)[0] - fixed
    return dict(run=run, build=build, total=max(run, build), fixed=fixed, list=with_list - fixed,
                edge=edge_part, node=node_part, edges_per_atom=e / n)


rows = []
for group, cases in (("fit", FIT["cases"]), ("capacity", CAPACITY), ("rescue", RESCUE)):
    for c in cases:
        p = parts(c)
        key = c.get("stage", c.get("backend", "")) if group != "capacity" else c["dtype"]
        rows.append(dict(group=group, label=c["label"], structure=c["structure"], key=key, atoms=c["nodes"],
                         dtype=c["dtype"], peak=c["peak_bytes"], ratio=p["total"] / c["peak_bytes"], **p))

print("== ratio estimate / measured by group")
by = defaultdict(list)
for r in rows:
    by[(r["group"], r["structure"], r["key"])].append(r)
for k in sorted(by):
    v = by[k]
    print(f"{k[0]:8s} {k[1]:6s} {str(k[2]):14s} n={len(v):2d} ratio {min(x['ratio'] for x in v):.3f}-"
          f"{max(x['ratio'] for x in v):.3f}  edges/atom {v[0]['edges_per_atom']:.1f}  "
          f"edge share {np.mean([x['edge'] / x['total'] for x in v]):.2f}  node share "
          f"{np.mean([x['node'] / x['total'] for x in v]):.2f}  fixed+list share "
          f"{np.mean([(x['fixed'] + x['list']) / x['total'] for x in v]):.2f}")

print("\n== least squares: measured = a*fixed + b*list + c*edge + d*node (per group; MH-1 only for fit)")
for name, sel in (("rescue", lambda r: r["group"] == "rescue"),
                  ("capacity f32", lambda r: r["group"] == "capacity" and r["dtype"] == "float32"),
                  ("capacity f64", lambda r: r["group"] == "capacity" and r["dtype"] == "float64"),
                  ("fit MH-1", lambda r: r["group"] == "fit" and r["label"].startswith("mh1")),
                  ("fit other", lambda r: r["group"] == "fit" and not r["label"].startswith("mh1"))):
    s = [r for r in rows if sel(r)]
    if len(s) < 4:
        continue
    A = np.array([[r["fixed"], r["list"], r["edge"], r["node"]] for r in s])
    y = np.array([r["peak"] for r in s])
    w = 1.0 / y  # relative errors
    coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    pred = A @ coef
    print(f"{name:13s} n={len(s):2d} a(fixed) {coef[0]:.2f} b(list) {coef[1]:.2f} c(edge) {coef[2]:.3f} "
          f"d(node) {coef[3]:.3f}  fitted ratio {min(pred / y):.3f}-{max(pred / y):.3f}")

print("\n== rescue: ratio against edges/atom and stage (the crystal difference)")
for r in sorted((r for r in rows if r["group"] == "rescue"), key=lambda r: (r["key"], r["structure"], r["atoms"])):
    print(f"  {r['key']:13s} {r['structure']:6s} {r['atoms']:7d} ratio {r['ratio']:.3f}  "
          f"edge {r['edge'] / r['peak']:.2f} node {r['node'] / r['peak']:.2f} "
          f"fixed+list {(r['fixed'] + r['list']) / r['peak']:.2f} (of the peak)")

Path(sys.argv[2] if len(sys.argv) > 2 else "residuals.json").write_text(json.dumps(rows, indent=1))
