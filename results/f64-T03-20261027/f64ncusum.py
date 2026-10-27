import csv
import io
from collections import defaultdict

rows = defaultdict(dict)
order = []
for line in open("f64ncu.csv"):
    if not line.startswith('"'):
        continue
    r = next(csv.reader(io.StringIO(line)))
    if len(r) < 15 or r[0] == "ID":
        continue
    key = (r[0], r[4][:60])
    if key not in rows:
        order.append(key)
    rows[key][r[12]] = r[14]
agg = defaultdict(lambda: defaultdict(float))
count = defaultdict(int)
for key in order:
    m = rows[key]
    name = key[1]
    count[name] += 1
    t = float(m.get("gpu__time_duration.sum", "0").replace(",", "")) / 1e3
    agg[name]["us"] += t
    for metric, label in (("sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active", "fp64%"),
                          ("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "mem%"),
                          ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "sm%"),
                          ("sm__warps_active.avg.pct_of_peak_sustained_active", "occ%")):
        try:
            agg[name][label] += float(m.get(metric, "nan").replace(",", "")) * t
        except ValueError:
            pass
for name, a in sorted(agg.items(), key=lambda kv: -kv[1]["us"])[:14]:
    t = a["us"]
    print(f"{t:9.0f} us x{count[name]:<3d} fp64 {a['fp64%'] / t:5.1f}%  mem {a['mem%'] / t:5.1f}%  sm {a['sm%'] / t:5.1f}%  "
          f"occ {a['occ%'] / t:5.1f}%  {name}")
