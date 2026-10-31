"""M1's table: the top kernels of one eager force call by time, with their L2 hit rate and DRAM bytes against an
ideal (each input tensor read once, from the profiled op's input shapes, float32/int32 = 4 bytes per element).
Reads step_breakdown's ``ncu_kernels.csv`` (the joined Nsight Compute rows) and ``ncu_raw.csv`` (for
``lts__t_sector_hit_rate.pct``, which the join does not carry), in the same kernel order.
    python m1_table.py CELL_DIR [--top 15]
"""

import argparse
import ast
import collections
import csv
import math
from pathlib import Path


def short(name):
    name = name.replace("void ", "")
    return name.split("<")[0].split("(")[0][:60]


def ideal_bytes(dims):
    try:
        shapes = ast.literal_eval(dims)
    except (ValueError, SyntaxError):
        return None
    total = 0
    for shape in shapes:
        if isinstance(shape, list) and shape and all(isinstance(n, int) for n in shape):
            total += math.prod(shape) * 4
    return total or None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cell", type=Path)
    p.add_argument("--top", type=int, default=15)
    args = p.parse_args()
    rows = list(csv.DictReader((args.cell / "ncu_kernels.csv").open()))
    raw = [r for r in csv.DictReader((args.cell / "ncu_raw.csv").open()) if r.get("ID", "").isdigit()]
    hits = [float(r.get("lts__t_sector_hit_rate.pct") or "nan") for r in raw]
    if len(hits) != len(rows):
        print(f"warning: {len(hits)} raw rows, {len(rows)} joined rows; hit rates by position")
    groups = collections.defaultdict(lambda: dict(us=0.0, dram=0.0, l2=0.0, ideal=0.0, ideal_known=True, n=0,
                                                   hit_weighted=0.0, category=""))
    total_us = 0.0
    for i, r in enumerate(rows):
        us = float(r["duration_us"] or 0)
        total_us += us
        g = groups[short(r["ncu_kernel"] or r["kernel"])]
        g["us"] += us
        g["n"] += 1
        g["dram"] += float(r["dram_read_bytes"] or 0) + float(r["dram_write_bytes"] or 0)
        g["l2"] += float(r["l2_bytes"] or 0)
        ideal = ideal_bytes(r["input_dims"])
        if ideal is None:
            g["ideal_known"] = False
        else:
            g["ideal"] += ideal
        if i < len(hits) and not math.isnan(hits[i]):
            g["hit_weighted"] += hits[i] * us
        g["category"] = r["category"]
    print(f"{args.cell.name}: force call kernel time {total_us / 1e3:.3f} ms, {len(rows)} kernels")
    print("| Kernel | Launches | ms | Share | L2 hit % | DRAM MB | Ideal MB (inputs once) | DRAM / ideal |")
    print("|---|---|---|---|---|---|---|---|")
    for name, g in sorted(groups.items(), key=lambda kv: -kv[1]["us"])[:args.top]:
        hit = g["hit_weighted"] / g["us"] if g["us"] else float("nan")
        ideal = g["ideal"] if g["ideal_known"] and g["ideal"] else None
        ratio = f"{g['dram'] / ideal:.2f}" if ideal else "–"
        print(f"| `{name}` | {g['n']} | {g['us'] / 1e3:.3f} | {100 * g['us'] / total_us:.1f} % | {hit:.1f} | "
              f"{g['dram'] / 1e6:.1f} | {ideal / 1e6 if ideal else float('nan'):.1f} | {ratio} |")


if __name__ == "__main__":
    main()
