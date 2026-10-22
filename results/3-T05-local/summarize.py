"""Markdown rows of tools/ab_skin.py results: python3 summarize.py <json> ... (3-T05)."""

import json
import sys


def rows(path):
    record = json.load(open(path))
    for name, entry in record["structures"].items():
        sides = entry["ab"]["sides"]
        timed, pairs = entry["timed"], entry["pair_ratios"]
        grew = any(timed[s]["capacity_after"] != timed[s]["capacity_before"] for s in "ab")
        yield (f"| {name} {entry['natoms']} | {record['temperature_K']:g} K | {''.join(record['order'])}, "
               f"{record['pattern'].upper()} | {entry['sides']['b']['skin_A']:g} / "
               f"{entry['plans']['b']['capacity_headroom']:g} | {sides['a']['median_ms']:.3f} | "
               f"{sides['b']['median_ms']:.3f} | **{entry['ratio_b_over_a']:.4f}** | "
               f"{sum(r < 1 for r in pairs)}/{len(pairs)}, {min(pairs):.3f}–{max(pairs):.3f} | "
               f"{timed['a']['neighbour_rebuilds']} → {timed['b']['neighbour_rebuilds']} | "
               f"{entry['steps_per_rebuild']['a']:.1f} → {entry['steps_per_rebuild']['b']:.1f} | "
               f"{'yes' if grew else 'no'} | {'yes' if entry['valid'] else 'NO'} |")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        for row in rows(path):
            print(row)
