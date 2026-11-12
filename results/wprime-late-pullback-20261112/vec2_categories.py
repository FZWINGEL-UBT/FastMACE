"""Two channels per lane on top of LATE_PULLBACK (conductor, 12 Nov 2026): where the instructions per edge go, lp
(skip + LATE_PULLBACK + hoist) against lp + CHANNEL_VECTOR 2 (sources from census_followup.py), sm_90 and sm_80, by
census_sass category and by opcode (the largest changes). -> vec2_categories.json
"""
import collections
import json

from common import HERE
from census_sass import census

out = {}
for arch in (90, 80):
    rows = {}
    for name, trips in (("fu-h200-lp", 4), ("fu-h200-lp-vec2", 2)):
        r = census(str(HERE / "sources" / f"{name}.cu"), arch, (), str(HERE / "sass"), trips=trips)
        rows[name] = r
    a, b = rows["fu-h200-lp"], rows["fu-h200-lp-vec2"]
    cats = {k: (a["per_edge"]["categories"].get(k, 0), b["per_edge"]["categories"].get(k, 0))
            for k in set(a["per_edge"]["categories"]) | set(b["per_edge"]["categories"])}
    ops = collections.Counter()
    for k in set(a["per_edge"]["ops"]) | set(b["per_edge"]["ops"]):
        ops[k] = b["per_edge"]["ops"].get(k, 0) - a["per_edge"]["ops"].get(k, 0)
    out[f"sm{arch}"] = dict(
        registers=(a["registers"], b["registers"]),
        per_edge=(a["per_edge"]["instructions"], b["per_edge"]["instructions"]),
        channel_rest=(a["channel"]["instructions"], b["channel"]["instructions"]),
        edge_rest=(a.get("edge_rest", {}).get("instructions"), b.get("edge_rest", {}).get("instructions")),
        categories=dict(sorted(cats.items(), key=lambda kv: kv[1][1] - kv[1][0])),
        largest_changes=dict(sorted(ops.items(), key=lambda kv: kv[1])[:8] + sorted(ops.items(), key=lambda kv: kv[1])[-8:]))
    print(arch, json.dumps(out[f"sm{arch}"]))
(HERE / "vec2_categories.json").write_text(json.dumps(out, indent=1) + "\n")
