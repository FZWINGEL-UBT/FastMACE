"""Derived statistics of 2-T12's noise-floor processes (the columns of 1a-T18's summary.json).

    python summarize.py <results directory>   # reads */metadata.json, writes summary.json
"""

import json
import statistics
import sys
from pathlib import Path


def side_info(info, *keys):
    for key in keys:
        if isinstance(info, dict) and key in info:
            return info[key]
    return None


def main(root):
    root = Path(root)
    runs = sorted(p.parent.name for p in root.glob("nf-*/metadata.json"))  # not the smoke run
    out = dict(runs=runs, structures={})
    for run in runs:
        meta = json.loads((root / run / "metadata.json").read_text())
        for name, entry in meta["structures"].items():
            ab = entry["ab"]
            a, b = ab["names"]
            sa, sb = ab["sides"][a], ab["sides"][b]
            pairs = ab["pair_ratios_b_over_a"]
            windows = [statistics.median(sb["block_ms"][i:i + 4]) / statistics.median(sa["block_ms"][i:i + 4])
                       for i in range(0, len(pairs), 4)]
            sides = entry["sides"]
            out["structures"].setdefault(name, dict(natoms=entry["natoms"], processes={}))
            out["structures"][name]["processes"][run] = dict(
                pattern=ab["pattern"], side_kind=meta["side_kinds"][a],
                median_ms={a: sa["median_ms"], b: sb["median_ms"]},
                block_spread={a: sa["spread"], b: sb["spread"]},
                block_stdev_rel={a: sa["stdev_ms"] / sa["median_ms"], b: sb["stdev_ms"] / sb["median_ms"]},
                ratio_b_over_a=ab["ratio_b_over_a"], pair_ratio_min=min(pairs), pair_ratio_max=max(pairs),
                window4_ratios=windows, observables_difference=entry["observables_difference"],
                compile_during_timing=entry["compile_during_timing"],
                other_gpu_processes=entry["other_gpu_processes"], dynamo_counters=entry["dynamo_counters"],
                setup_s={s: sides[s].get("setup_s") for s in (a, b)},
                rebuilds={s: side_info(sides[s].get("info"), "neighbour_rebuilds", "rebuilds") for s in (a, b)},
                edge_sizes={s: side_info(sides[s].get("info"), "edge_sizes_used", "edge_sizes") for s in (a, b)})
    for name, s in out["structures"].items():
        for kind in sorted({p["side_kind"] for p in s["processes"].values()}):
            procs = [p for p in s["processes"].values() if p["side_kind"] == kind]
            ratios = [p["ratio_b_over_a"] for p in procs]
            windows = [w for p in procs for w in p["window4_ratios"]]
            pairs_lo = min(p["pair_ratio_min"] for p in procs)
            pairs_hi = max(p["pair_ratio_max"] for p in procs)
            a_medians = [p["median_ms"][next(iter(p["median_ms"]))] for p in procs]
            s.setdefault("floor", {})[kind] = dict(
                processes=len(procs), ratio_12_max_dev=max(abs(r - 1) for r in ratios),
                ratio_12=[min(ratios), max(ratios)], window4=[min(windows), max(windows)],
                window4_max_dev=max(abs(w - 1) for w in windows), pair_ratio=[pairs_lo, pairs_hi],
                a_median_spread_across_processes=(max(a_medians) - min(a_medians)) / statistics.median(a_medians))
    (root / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    for name, s in out["structures"].items():
        print(name, json.dumps(s["floor"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
