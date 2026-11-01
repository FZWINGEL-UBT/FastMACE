"""The rescue records' side fractions from the measured live peaks, and rule 12's band per crystal and GPU.

1. Per stage of the rescue ladder, least squares of the live peak (max allocated over build, warm-up and steady) on
   fixed + per atom x N + per edge x E, over the fit crystals (every crystal but HELD_OUT).
2. The fractions: conv_weights and first_table from the per-edge slopes of consecutive stages; the node segments,
   the node region and gate_epilogue (alone, with the segments) from the per-atom slopes.
3. With those fractions (set on fastmace.rewrites.memory), rule 12's estimate against the measured peak (max
   reserved) and the live peak, per crystal, GPU and stage, the held-out crystal separately; and the fit and
   capacity sets.
    python band.py FM_SOURCE CASES.json POINT_DIR...      (CASES.json: matrix_cases.py's output, every crystal)
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
from fastmace.rewrites import memory as terms  # noqa: E402

HELD_OUT = () if "--all" in sys.argv else ("NaNbO3",)  # --all: the final constants, every crystal
# Below this the build's fixed transient (about 1.7 GB, the build term's) sets the live peak, not the force call.
MIN_ATOMS = 15_000
DATA = FM / "tests/_support/data"
FIT = json.loads((DATA / "rule12_fit.json").read_text())
registry = minimal._rewrite_registry()
cases = json.loads(Path(sys.argv[2]).read_text())
points = {}
for d in sys.argv[3:]:
    for f in Path(d).rglob("point-*.json"):
        points[(f.parent.parent.name, f.name)] = json.loads(f.read_text())  # file names repeat across runs


def peaks(case):
    source = Path(case["source"])
    p = points[(source.parent.parent.name, source.name)]
    ph = [v for v in p["phases"].values() if isinstance(v, dict) and v.get("max_allocated")]
    return max(v["max_allocated"] for v in ph), max(v["max_reserved"] for v in ph), p["gpu"]


for c in cases:
    c["alloc"], c["reserved"], c["gpu"] = peaks(c)

# 1. slopes per stage over the fit crystals
slope = {}
print("stage          n  fixed GiB  per atom kB  per edge kB  fit residual")
for stage in sorted({c["stage"] for c in cases}):
    v = [c for c in cases if c["stage"] == stage and c["structure"] not in HELD_OUT and c["nodes"] >= MIN_ATOMS]
    A = np.array([[1.0, c["nodes"], c["edges"]] for c in v])
    y = np.array([c["alloc"] for c in v], dtype=float)
    coef, *_ = np.linalg.lstsq(A / y[:, None], np.ones(len(v)), rcond=None)
    slope[stage] = coef
    r = A @ coef / y
    print(f"{stage:13s} {len(v):2d}  {coef[0] / 2**30:8.2f}  {coef[1] / 1e3:10.1f}  {coef[2] / 1e3:10.3f}   "
          f"{r.min():.3f}-{r.max():.3f}")
    held = [c for c in cases if c["stage"] == stage and c["structure"] in HELD_OUT and c["nodes"] >= MIN_ATOMS]
    if held:
        r = np.array([[1.0, c["nodes"], c["edges"]] for c in held]) @ coef / np.array([c["alloc"] for c in held])
        print(f"{'':13s} held out {', '.join(HELD_OUT)}: predicted/measured live peak {r.min():.3f}-{r.max():.3f}")

# 2. the fractions
n = {s: slope[s][1] for s in slope}
e = {s: slope[s][2] for s in slope}
base_node = np.mean([n["s0-default"], n["s1-conv"], n["s2-table"]])  # gate_epilogue on, no node rung
ge_alone = (base_node + 0.0) / (base_node - 12.3e3)  # region-norm.md's 12.3 kB per atom, too small to fit here
node0 = base_node / ge_alone
fractions = dict(
    _CONV_WEIGHTS_KEEP=e["s1-conv"] / e["s0-default"],
    _FIRST_TABLE_KEEP=e["s2-table"] / e["s1-conv"],
    _NODE_SEGMENTS_KEEP=n["s4-geoff"] / node0,
    _GATE_EPILOGUE_SEGMENTS_KEEP=n["s3-segments"] / n["s4-geoff"],
    _NODE_REGION_KEEP=n["s5-region"] / n["s4-geoff"],
    _GATE_EPILOGUE_KEEP=ge_alone,
)
print("\nfractions:", ", ".join(f"{k} {v:.3f}" for k, v in fractions.items()),
      f"(region with gate_epilogue / without: {n['s6-region-ge'] / n['s5-region']:.3f})")
if "--keep-code" not in sys.argv:
    for k, v in fractions.items():
        setattr(terms, k, round(float(v), 3))
# --mh1-extra=F: F more of MH-1's stock per-edge floats that the fused path does not keep (float32), so the per-edge
# part's overstatement matches the per-node part's at s0
for arg in sys.argv:
    if arg.startswith("--mh1-extra="):
        extra = int(arg.split("=", 1)[1])
        terms._MH1_STREAMING_SHARE = {4: (2000 + extra) / 8265, 8: terms._MH1_STREAMING_SHARE[8]}
        print(f"MH-1 streaming share, float32: {2000 + extra} of 8265 floats")


# 3. the per-stage allocator reserve: the factor on the kept remainder that puts the stage's lowest fit point at its
# measured peak (the upper envelope over the fit crystals, every size), rounded up to 0.005
def full(c):
    dims = ModelDims.from_record(FIT["dims"][c["label"]])
    dt = c["dtype"]
    est = memory.estimate(dims, edges=c["edges"], nodes=c["nodes"], dtype=dt, capacity=c["capacity"], list_dtype=dt,
                          fused_conv=c["fused_conv"],
                          rewrite_terms=memory.rewrite_memory_terms(registry, dims, c["rewrites"], dt),
                          selfcheck_terms=memory.selfcheck_memory_terms(registry, dims, c["selfchecks"], dt, c["sms"]),
                          resident_terms=memory.resident_memory_terms(registry, dims, c["resident"], dt))
    return est


memory.RESCUE_RESERVE = tuple((name, 1.0) for name, _ in memory.RESCUE_RESERVE)
need = defaultdict(list)
for c in cases:
    est = full(c)
    if est.build >= c["reserved"]:
        continue
    kept = est.term("activations") + est.term("rewrites")
    need[memory.rescue_stage(n for n, _ in memory.rewrite_memory_terms(registry, ModelDims.from_record(
        FIT["dims"][c["label"]]), c["rewrites"], c["dtype"]))].append(
        ((c["reserved"] - (est.run - kept)) / kept, c["structure"] in HELD_OUT, c["structure"], c["nodes"]))
factors = {}
for stage, v in sorted(need.items()):
    fit = max(x for x in v if not x[1])
    held = max((x for x in v if x[1]), default=None)
    factors[stage] = float(np.ceil(fit[0] * 200) / 200)
    print(f"reserve {stage or 'default':25s} envelope {fit[0]:.3f} ({fit[2]} {fit[3]}) -> {factors[stage]:.3f}"
          + (f"; held out needs {held[0]:.3f} ({held[3]})" if held else ""))
if "--no-reserve" not in sys.argv:
    memory.RESCUE_RESERVE = tuple((name, factors.get(name, 1.0)) for name, _ in memory.RESCUE_RESERVE)


def estimate(c):
    return full(c).total


groups = defaultdict(list)
for c in cases:
    est = estimate(c)
    c["ratio"], c["ratio_alloc"] = est / c["reserved"], est / c["alloc"]
    groups[(c["structure"] in HELD_OUT, c["gpu"][-14:], c["structure"])].append(c)
print("\n                               n  est/reserved    est/allocated  reserved/allocated  edges/atom")
for (held, gpu, s), v in sorted(groups.items()):
    q = lambda k: f"{min(c[k] for c in v):.3f}-{max(c[k] for c in v):.3f}"  # noqa: E731
    rs = [c["reserved"] / c["alloc"] for c in v]
    print(f"{'held ' if held else 'fit  '}{gpu:14s} {s:7s} {len(v):2d}  {q('ratio'):12s}  {q('ratio_alloc'):12s}  "
          f"{min(rs):.3f}-{max(rs):.3f}      {v[0]['edges'] / v[0]['nodes']:.1f}")
by_stage = defaultdict(list)
for c in cases:
    by_stage[c["stage"]].append(c)
print("\nby stage:", "; ".join(f"{s} {min(c['ratio'] for c in v):.3f}-{max(c['ratio'] for c in v):.3f}"
                              for s, v in sorted(by_stage.items())))
allr = [c["ratio"] for c in cases]
print(f"all rescue points: est/reserved {min(allr):.3f}-{max(allr):.3f}; est/allocated "
      f"{min(c['ratio_alloc'] for c in cases):.3f}-{max(c['ratio_alloc'] for c in cases):.3f}")
for name, file in (("fit (60)", "rule12_fit.json"), ("capacity (28)", "rule12_capacity.json")):
    data = json.loads((DATA / file).read_text())
    data = data["cases"] if isinstance(data, dict) else data
    r = [estimate(c) / c["peak_bytes"] for c in data]
    print(f"{name}: {min(r):.3f}-{max(r):.3f}")
print("RESCUE_RESERVE =", memory.RESCUE_RESERVE)
Path("band.json").write_text(json.dumps(dict(fractions=fractions, reserve=memory.RESCUE_RESERVE, cases=[
    {k: c[k] for k in ("structure", "gpu", "stage", "nodes", "edges", "alloc", "reserved", "ratio", "ratio_alloc")}
    for c in cases]), indent=1))
