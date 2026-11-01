"""Where the measured peak sits above the live tensors: for every rescue and capacity case with its point JSON, the
peak reserved (rule 12's target), the peak allocated (live tensors), and per phase the reserve (reserved - allocated
at that phase's peak), with the outside-allocator bytes (context, cuBLAS workspace, ...). Then rule 12's estimate
against both, by platform, crystal and stage.
    python reserve.py FM_SOURCE POINT_DIR...
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

FM = Path(sys.argv[1])
sys.path.insert(0, str(FM))
from fastmace.model.capabilities import ModelDims  # noqa: E402
from fastmace.planner import memory, minimal  # noqa: E402

DATA = FM / "tests/_support/data"
FIT = json.loads((DATA / "rule12_fit.json").read_text())
registry = minimal._rewrite_registry()
points = {}
for d in sys.argv[2:]:
    for f in Path(d).rglob("point-*.json"):
        points[f.name] = json.loads(f.read_text())


def estimate(case):
    dims = ModelDims.from_record(FIT["dims"][case["label"]])
    dt = case["dtype"]
    e = memory.estimate(dims, edges=case["edges"], nodes=case["nodes"], dtype=dt, capacity=case["capacity"],
                        list_dtype=dt, fused_conv=case["fused_conv"],
                        rewrite_terms=memory.rewrite_memory_terms(registry, dims, case["rewrites"], dt),
                        selfcheck_terms=memory.selfcheck_memory_terms(registry, dims, case["selfchecks"], dt,
                                                                      case["sms"]),
                        resident_terms=memory.resident_memory_terms(registry, dims, case["resident"], dt))
    return max(e.run, e.build), e.run, e.build


rows = []
for name, cases in (("rescue", json.loads((DATA / "rule12_rescue.json").read_text())),
                    ("capacity", json.loads((DATA / "rule12_capacity.json").read_text()))):
    for c in cases:
        p = points.get(Path(c["source"]).name)
        if p is None:
            continue
        ph = {k: v for k, v in p["phases"].items() if isinstance(v, dict) and v.get("max_reserved")}
        alloc = max(v["max_allocated"] for v in ph.values())
        res = max(v["max_reserved"] for v in ph.values())
        est, run, build = estimate(c)
        rows.append(dict(set=name, gpu=p["gpu"][:22], host=p["host"], structure=c["structure"], dtype=c["dtype"],
                         stage=c.get("stage", "default"), atoms=c["nodes"], est=est, run=run, build=build,
                         reserved=res, allocated=alloc, peak=c["peak_bytes"],
                         outside=max(v.get("outside_allocator", 0) for v in ph.values()),
                         phases={k: (v["max_allocated"], v["max_reserved"]) for k, v in ph.items()}))
print(f"{len(rows)} cases with points")
G = 2 ** 30
for r in sorted(rows, key=lambda r: (r["set"], r["gpu"], r["structure"], r["dtype"], r["stage"], r["atoms"])):
    b = r["phases"].get("build", (0, 0))
    s = r["phases"].get("steady", r["phases"].get("warmup", (0, 0)))
    w = r["phases"].get("warmup", (0, 0))
    print(f"{r['set']:8s} {r['gpu']:22s} {r['structure']:6s} {r['dtype']:7s} {r['stage']:13s} {r['atoms']:7d} "
          f"est/res {r['est'] / r['reserved']:.3f} est/alloc {r['est'] / r['allocated']:.3f} "
          f"res/alloc {r['reserved'] / r['allocated']:.3f}  build a/r {b[0] / G:5.2f}/{b[1] / G:5.2f} "
          f"warm a/r {w[0] / G:5.2f}/{w[1] / G:5.2f}  outside {r['outside'] / G:.2f}")
by = defaultdict(list)
for r in rows:
    by[(r["set"], r["gpu"], r["dtype"])].append(r)
print()
for k, v in sorted(by.items()):
    q = [r["reserved"] / r["allocated"] for r in v]
    e1 = [r["est"] / r["reserved"] for r in v]
    e2 = [r["est"] / r["allocated"] for r in v]
    print(f"{k[0]:8s} {k[1]:22s} {k[2]:7s} n={len(v):2d} res/alloc {min(q):.3f}-{max(q):.3f}  est/res "
          f"{min(e1):.3f}-{max(e1):.3f} ({max(e1) / min(e1):.3f})  est/alloc {min(e2):.3f}-{max(e2):.3f} "
          f"({max(e2) / min(e2):.3f})")
Path("reserve.json").write_text(json.dumps(rows, indent=1))
