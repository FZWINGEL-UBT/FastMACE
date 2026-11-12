"""Speed table of tools/ab_rewrites.py outputs: analyse_speed.py DIR (ab-ab.json, ab-ba.json) -> markdown on stdout.
Per structure and build order: side a (defaults) and b (PACKED8 off) median, mean, min-max of the block times (ms per
MD step), the ratio of medians b/a and the range of the per-pair ratios; every block time is in the JSON."""
import json
import sys
from pathlib import Path

rows = ["| structure | order | a median / mean (min-max) ms | b median / mean (min-max) ms | b/a | pair ratios | valid |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
for order in ("ab", "ba"):
    path = Path(sys.argv[1]) / f"ab-{order}.json"
    if not path.is_file():
        continue
    record = json.loads(path.read_text())
    for name, entry in record["structures"].items():
        sides = entry["ab"]["sides"]
        cell = lambda s: f"{s['median_ms']:.3f} / {s['mean_ms']:.3f} ({s['min_ms']:.3f}-{s['max_ms']:.3f})"
        pairs = entry["pair_ratios"]  # b/a per pair (the inner record names its first-built side first)
        rows.append(f"| {name} ({entry['natoms']}) | {order} | {cell(sides['a'])} | {cell(sides['b'])} | "
                    f"{entry['ratio_b_over_a']:.4f} | {min(pairs):.4f}-{max(pairs):.4f} | {entry['valid']} |")
print(f"GPU {record['gpu']}, protocol {record['protocol']}\n")
print("\n".join(rows))
