import json
from collections import defaultdict

ours = {"NaCl": 0.240, "NbCl3O": 0.615, "NbCl5": 1.792, "NaNbO3": 2.840}  # streaming backward, t00 medians (MH-1)
table = defaultdict(dict)
bad = defaultdict(list)
for line in open("sweep.jsonl"):
    if not line.startswith("{"):
        continue
    d = json.loads(line)
    cell = d["file"].split("-")[-1][:-3]
    t = d.get("backward_direct_ms")
    err = max(d.get(k, 1) for k in ("direct_dx_vs", "direct_dsh_vs", "direct_dw_vs"))
    table[d["config"]][cell] = (t, err)
cells = ["NaCl", "NbCl3O", "NbCl5", "NaNbO3"]
print("config".ljust(18), "  ".join(c.rjust(14) for c in cells), "   (ms, ratio to ours; * = error > 1e-5)")
for config, row in table.items():
    parts = []
    for c in cells:
        t, err = row.get(c, (None, None))
        parts.append("-".rjust(14) if t is None else f"{t:6.3f} {t / ours[c]:5.2f}{'*' if err > 1e-5 else ' '}")
    print(config.ljust(18), "  ".join(parts))
