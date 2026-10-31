"""Rule 12's fit data with counted edges: each case's ``edges`` becomes the pairs within the model's r_max counted in
the structure (both directions) plus the 5 % edge bucket, as ``memory.counted_edges`` gives; the uniform-density
count stays as ``uniform_edges``. r_max is recovered from the uniform count. The matrix cases become
rule12_rescue.json.
    python data_update.py FM_SOURCE DATA_DIR MATRIX_CASES.json
"""

import json
import math
import sys
from pathlib import Path

FM = Path(sys.argv[1])
sys.path.insert(0, str(FM / "tools" / "bench"))
import capacity_scan  # noqa: E402
from ase.neighborlist import neighbor_list  # noqa: E402

DATA = Path(sys.argv[2])
BUCKET = 0.05
cache = {}


def per_atom(structure, r):
    key = (structure, round(r, 3))
    if key not in cache:
        _, cell = capacity_scan.base_cell(structure)
        cell.set_pbc(True)
        cache[key] = len(neighbor_list("i", cell, r)) / len(cell)
    return cache[key]


def volume_per_atom(structure):
    _, cell = capacity_scan.base_cell(structure)
    return cell.get_volume() / len(cell)


def r_max(case, uniform):
    n = case["nodes"]
    v = volume_per_atom(case["structure"]) * n
    r = (uniform / (1 + BUCKET) * v / (n * n * 4.0 / 3.0 * math.pi)) ** (1 / 3)
    return round(r * 2) / 2  # the labels' cutoffs are whole or half Å


def update(case):
    uniform = case.get("uniform_edges", case["edges"])
    r = r_max(case, uniform)
    case["uniform_edges"] = uniform
    case["r_max"] = r
    case["edges"] = int(math.ceil(per_atom(case["structure"], r) * case["nodes"] * (1 + BUCKET)))
    return case


fit = json.loads((DATA / "rule12_fit.json").read_text())
fit["cases"] = [update(c) for c in fit["cases"]]
(DATA / "rule12_fit.json").write_text(json.dumps(fit, indent=1) + "\n")
cap = [update(c) for c in json.loads((DATA / "rule12_capacity.json").read_text())]
(DATA / "rule12_capacity.json").write_text(json.dumps(cap, indent=1) + "\n")
rescue = []
for c in json.loads(Path(sys.argv[3]).read_text()):
    c = update(dict(c))
    c["source"] = c["source"].split("fastmace-capacity/", 1)[-1]
    rescue.append(c)
(DATA / "rule12_rescue.json").write_text(json.dumps(rescue, indent=1) + "\n")
print("r_max:", sorted({(c["label"], c["r_max"]) for c in fit["cases"] + cap + rescue}))
print("per atom:", {k: round(v, 3) for k, v in cache.items()})
