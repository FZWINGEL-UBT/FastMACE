"""The step-layout-cleanup rows (µs per step) from attribution-<s>.json and gaps-radix-<s>.json of one machine.

  python layout_rows.py DIR   (DIR holds attribution-{SrTiO3,NbCl5}.json and gaps-radix-{SrTiO3,NbCl5}.json)
"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
ROWS = [("outer-sum Memset, tabled_outer_sum.py:600", lambda k: "tabled_outer_sum.py:600" in k and "[memset" in k),
        ("backward gather+transpose, tabled_outer_sum.py:739", lambda k: "tabled_outer_sum.py:739" in k),
        ("backward, tabled_outer_sum.py:757", lambda k: "tabled_outer_sum.py:757" in k),
        ("tf32x3_gemm.py:171 clone/slice_backward", lambda k: "tf32x3_gemm.py:171" in k),
        ("generic/modules.py:211 (forward)", lambda k: "generic/modules.py:211" in k),
        ("generic/modules.py:530 (forward)", lambda k: "generic/modules.py:530" in k and "lean_forward" in k),
        ("generic/modules.py:530 (backward)", lambda k: "generic/modules.py:530" in k and "compute_forces" in k),
        ("per-step compose, mh1 refresh_composed/_compose_weights", lambda k: "_compose_weights" in k or "refresh_composed" in k)]
print("| Row | " + " | ".join(("SrTiO3 10³", "NbCl5")) + " |\n| --- | --- | --- |")
data = {s: json.loads((d / f"attribution-{s}.json").read_text()) for s in ("SrTiO3", "NbCl5")}
gaps = {s: json.loads((d / f"gaps-radix-{s}.json").read_text()) if (d / f"gaps-radix-{s}.json").exists() else None
        for s in data}


def cell(us, total):
    return f"{us:.0f} ({100 * us / total:.2f}%)"


for name, test in ROWS:
    cells = []
    for s, a in data.items():
        us = sum(r["us_per_step"] for r in a["by_site"] if r["class"] not in ("ours: NVRTC", "ours: Triton")
                 and test(r["key"])) if "compose" not in name else \
            sum(r["us_per_step"] for r in a["by_site"] if test(r["key"]))
        cells.append(cell(us, a["kernel_us_per_step"]))
    print(f"| {name} | " + " | ".join(cells) + " |")
cells = []
for s, g in gaps.items():
    if g is None:
        cells.append("n/a")
        continue
    graph = [r for r in g["radix"] if "None" not in r["graph_ids"]]
    us, n = sum(r["us_per_step"] for r in graph), sum(r["per_step"] for r in graph)
    cells.append(cell(us, data[s]["kernel_us_per_step"]) + f", {n:g} sorts/step in graph")
print("| species_plan argsort, species_gemm.py:249 (radix sort in graph) | " + " | ".join(cells) + " |")
print("| **Not ours, total** | " + " | ".join(cell(a["not_ours_us_per_step"], a["kernel_us_per_step"]) for a in data.values())
      + " |")
print("| GPU kernel time per step | " + " | ".join(f"{a['kernel_us_per_step']:.0f}" for a in data.values()) + " |")
for s, g in gaps.items():
    if g:
        print(f"{s}: steady gaps > 20 µs {g['gaps_over_threshold_us_per_step']} µs/step; window-start artefacts "
              f"{[x['gap_us'] for x in g['window_start_artefacts']]}; top: "
              + "; ".join(f"{r['us_per_step']} µs {r['pattern'][:110]}" for r in g["gap_patterns"][:3]))
