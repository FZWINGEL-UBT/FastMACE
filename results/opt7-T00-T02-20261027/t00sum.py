import json
import statistics
from collections import defaultdict

rows = defaultdict(lambda: defaultdict(list))
checks = {}
for line in open("t00.jsonl"):
    if not line.startswith("{"):
        continue
    d = json.loads(line)
    key = d["file"].replace("ops-", "").replace(".pt", "")
    for field in ("forward_ms", "backward_ms", "create_graph_ms"):
        if field in d:
            rows[key][(d["side"], field)].append(d[field])
    if d["side"] == "lunatp":
        checks[key] = max(d.get(k, 0) for k in ("forward_vs_streaming", "dx_vs", "dsh_vs", "dw_vs"))
        if "backward_error" in d:
            checks[key] = d["backward_error"]
print(f"{'case':24s} {'fwd ours':>8s} {'fwd luna':>8s} {'ratio':>6s} {'bwd ours':>8s} {'bwd luna':>8s} {'ratio':>6s} "
      f"{'graph':>7s} {'max rel diff':>12s}")
for key, values in sorted(rows.items()):
    m = {k: statistics.median(v) for k, v in values.items()}
    fo, fl = m.get(("streaming", "forward_ms")), m.get(("lunatp", "forward_ms"))
    bo, bl = m.get(("streaming", "backward_ms")), m.get(("lunatp", "backward_ms"))
    g = m.get(("lunatp", "create_graph_ms"))
    print(f"{key:24s} {fo:8.3f} {fl:8.3f} {fl / fo:6.2f} {bo:8.3f} {(bl or float('nan')):8.3f} "
          f"{(bl or float('nan')) / bo:6.2f} {g:7.2f} {checks.get(key)!s:>12.12s}")
