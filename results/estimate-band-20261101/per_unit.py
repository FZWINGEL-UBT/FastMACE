"""The measured live peak (max allocated) per stage as fixed + per-atom x N + per-edge x E, against rule 12's own
per-atom and per-edge parts before the rescue fractions: what each rung frees per edge or per atom (the difference
between consecutive stages), and what the estimate counts.
    python per_unit.py FM_SOURCE POINT_DIR...
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

FM = Path(sys.argv[1])
sys.path.insert(0, str(FM))
sys.path.insert(0, str(Path(__file__).parent))
import split_fit  # noqa: E402  (pieces: fixed, edge, node before the rescue fractions)

points = {}
for d in sys.argv[2:]:
    for f in Path(d).rglob("point-*.json"):
        points[f.name] = json.loads(f.read_text())

rows = []
for c in split_fit.RESCUE + [c for c in split_fit.CAPACITY if c["dtype"] == "float32"]:
    p = points.get(Path(c["source"]).name)
    if p is None:
        continue
    ph = [v for v in p["phases"].values() if isinstance(v, dict) and v.get("max_allocated")]
    q = split_fit.pieces(c)
    rows.append(dict(gpu=p["gpu"][-12:], structure=c["structure"], stage=c.get("stage", "default"), N=c["nodes"],
                     E=c["edges"], alloc=max(v["max_allocated"] for v in ph),
                     reserved=max(v["max_reserved"] for v in ph), fixed=q["fixed"], edge=q["edge"], node=q["node"]))
by = defaultdict(list)
for r in rows:
    by[r["stage"]].append(r)
print("stage          n  per atom kB  per edge kB  fixed GiB   (least squares on alloc; fit residual)  "
      "estimate before fractions: per atom kB, per edge kB")
coef = {}
for stage in sorted(by):
    v = by[stage]
    A = np.array([[1.0, r["N"], r["E"]] for r in v])
    y = np.array([r["alloc"] for r in v])
    w = 1 / y
    if len(v) >= 3:
        c, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        res = A @ c / y
        coef[stage] = c
        fit = f"{c[1] / 1e3:8.1f}  {c[2] / 1e3:8.3f}  {c[0] / 2**30:8.2f}   {res.min():.3f}-{res.max():.3f}"
    else:
        fit = "(too few)"
    en = np.mean([r["node"] / r["N"] for r in v]) / 1e3
    ee = np.mean([r["edge"] / r["E"] for r in v]) / 1e3
    print(f"{stage:13s} {len(v):2d}  {fit}   est {en:7.1f} {ee:7.3f}")
print("\nper point: alloc per atom (kB), edges per atom, reserved/alloc")
for r in sorted(rows, key=lambda r: (r["stage"], r["structure"], r["N"])):
    print(f"  {r['gpu']:12s} {r['structure']:6s} {r['stage']:13s} {r['N']:7d} alloc/N {r['alloc'] / r['N'] / 1e3:6.1f} "
          f"E/N {r['E'] / r['N']:5.1f}  res/alloc {r['reserved'] / r['alloc']:.3f}  "
          f"(alloc - est fixed)/N {(r['alloc'] - r['fixed']) / r['N'] / 1e3:6.1f}")
