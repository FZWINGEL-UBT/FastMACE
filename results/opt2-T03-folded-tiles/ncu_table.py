"""OPT2-T03: the table of an ``ncu`` CSV from ``local.sh ncu`` (the third, warm launch of each structure and rule).

    python ncu_table.py ROOT/ncu.log
"""

import collections
import csv
import sys

KEYS = (("Duration", "duration_us", 1e-3), ("DRAM Throughput", "dram_pct", 1), ("L1/TEX Cache Throughput", "l1_pct", 1),
        ("Compute (SM) Throughput", "sm_pct", 1), ("Achieved Occupancy", "achieved_occupancy_pct", 1),
        ("Theoretical Occupancy", "theoretical_occupancy_pct", 1), ("Registers Per Thread", "registers", 1),
        ("Waves Per SM", "waves", 1))
LABELS = [(structure, rule) for structure in ("NaCl", "NbCl5", "NaNbO3") for rule in ("fixed", "waves")]


def table(path):
    rows = list(csv.reader(open(path, encoding="utf-8", errors="replace")))
    start = next(i for i, row in enumerate(rows) if row and row[0] == "ID")
    header = rows[start]
    launches = collections.defaultdict(dict)
    for row in rows[start + 1:]:
        if len(row) < 15:  # metric rows stop after "Metric Value" and an empty rule column; rule rows are longer
            continue
        entry = dict(zip(header, row))
        launch = launches[int(entry["ID"])]
        launch[entry["Metric Name"]] = entry["Metric Value"]
        launch["grid"], launch["block"] = entry["Grid Size"], entry["Block Size"]
    out = []
    for index, (structure, rule) in enumerate(LABELS):
        launch = launches[3 * index + 2]
        record = {"structure": structure, "rule": rule, "grid": launch["grid"], "block": launch["block"]}
        for metric, key, scale in KEYS:
            value = launch.get(metric)
            record[key] = None if value in (None, "") else round(float(value.replace(",", "")) * scale, 2)
        out.append(record)
    return out


if __name__ == "__main__":
    print("| Structure | Tiles | Grid × block | Duration (µs) | DRAM % | L1 % | SM % | Occupancy achieved / theoretical % "
          "| Registers | Waves |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in table(sys.argv[1]):
        print(f"| {r['structure']} | {r['rule']} | {r['grid']} × {r['block']} | {r['duration_us']} | {r['dram_pct']} | "
              f"{r['l1_pct']} | {r['sm_pct']} | {r['achieved_occupancy_pct']} / {r['theoretical_occupancy_pct']} | "
              f"{r['registers']} | {r['waves']} |")
