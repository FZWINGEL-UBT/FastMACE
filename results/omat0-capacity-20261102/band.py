"""Rule 12's estimate against the measured peak (max reserved) for OMAT-0 matrix cases, with today's code, per stage
and crystal; and the factor per stage that would put each stage's lowest case at its peak.
    python band.py FM_SOURCE CASES.json...
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

FM = Path(sys.argv[1])
sys.path.insert(0, str(FM))
from fastmace.model.capabilities import ModelDims  # noqa: E402
from fastmace.planner import memory, minimal  # noqa: E402

FIT = json.loads((FM / "tests/_support/data/rule12_fit.json").read_text())
registry = minimal._rewrite_registry()
groups = defaultdict(list)
for file in sys.argv[2:]:
    for c in json.loads(Path(file).read_text()):
        dims = ModelDims.from_record(FIT["dims"][c["label"]])
        dt = c["dtype"]
        names = [n for n in c["rewrites"] if n != "recompute_conv_weights"]  # MH-1's record: not this model's
        est = memory.estimate(dims, edges=c["edges"], nodes=c["nodes"], dtype=dt, capacity=c["capacity"],
                              list_dtype=dt, fused_conv=c["fused_conv"],
                              rewrite_terms=memory.rewrite_memory_terms(registry, dims, names, dt),
                              selfcheck_terms=memory.selfcheck_memory_terms(registry, dims, c["selfchecks"], dt,
                                                                            c["sms"]),
                              resident_terms=memory.resident_memory_terms(registry, dims, c["resident"], dt))
        kept = est.term("activations") + est.term("rewrites")
        need = (c["peak_bytes"] - (est.run - kept)) / kept  # the factor on the kept part that meets the peak
        groups[c["stage"]].append((c["structure"], c["nodes"], c["gpu"][-8:], est.total / c["peak_bytes"], need))
for stage, rows in sorted(groups.items()):
    print(stage, f"{min(r[3] for r in rows):.3f}-{max(r[3] for r in rows):.3f}",
          "envelope (cells >= 15k)", f"{max((r[4] for r in rows if r[1] >= 15000), default=float('nan')):.3f}")
    for r in sorted(rows):
        print(f"   {r[0]:7s} {r[1]:7d} {r[2]} est/peak {r[3]:.3f} factor needed {r[4]:.3f}")
