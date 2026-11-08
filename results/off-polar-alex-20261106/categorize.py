"""Groups a module_profile sources.csv into time sinks: categorize.py <profile dir>..."""
import ast
import csv
import sys
from collections import defaultdict


def dims(r):
    try:
        return ast.literal_eval(r["input_dims"])
    except Exception:
        return []


def first_rows(d):
    out = []
    for x in d:
        if isinstance(x, list) and x and isinstance(x[0], int):
            out.append(x[0])
    return out


def category(r, edges, kcount, natoms):
    k, op, d = r["kernel"], r["op"], dims(r)
    rows = first_rows(d)
    flat = str(d)
    if kcount and (str(kcount) in flat):
        return "k-space (cos/sin, structure factors)"
    if "jit_conv" in op:
        return "stock block-0 conv (OEq jit)"
    if "streaming_conv" in op or "fused_folded_forward" in op or "wprime" in op:
        return "streaming conv (later blocks)"
    if op == "aten::bmm" or ("clone_permute_slice" in k) or (op == "aten::mm" and len(d) > 1 and d[0][1:] == [10]
                                                            and d[1][1:2] and d[1][1] >= 4096):
        return "stock block-0 skip_tp (per-atom matrices)"
    if "cuequivariance" in op:
        return "symmetric contraction (cuEq)"
    if "folded_linear_input_grad" in op or (edges and rows and rows[0] == edges):
        return "radial MLPs (edges)"
    if "silu" in k:  # SiLU on atom rows: node MLPs (POLAR's heads and field maps), not the radial MLPs
        return "node MLPs (silu on atom rows)"
    if natoms and any(x == [natoms, natoms] or (isinstance(x, list) and len(x) == 2 and x[0] in (kcount,)) for x in d):
        return "k-space (cos/sin, structure factors)"
    if "tf32x3" in k or "grouped" in op or "tf32x3" in op or (op == "aten::mm" and rows and rows[0] in
                                                                  (natoms, 3 * natoms, 5 * natoms, 7 * natoms)):
        return "node linears (product/skip/readout)"
    if "edge_sh" in r["source"] or "edge_sh" in k:
        return "edge harmonics"
    return "other (elementwise, copies, small)"


for path in sys.argv[1:]:
    rows = list(csv.DictReader(open(path + "/sources.csv")))
    natoms = 1944 if "NbCl5" in path else 216 if "NaCl" in path else 648
    edges = kcount = 0
    for r in rows:
        lead = first_rows(dims(r))
        if "silu" in r["kernel"] and lead and lead[0] > natoms:  # the first SiLU on more rows than atoms: edges
            edges = lead[0]
            break
    for r in rows:
        for x in dims(r):
            if isinstance(x, list) and len(x) == 2 and x[1] == natoms and x[0] not in (natoms, edges) and x[0] > 500:
                kcount = x[0]
    totals = defaultdict(float)
    for r in rows:
        totals[category(r, edges, kcount, natoms)] += float(r["us_per_call"])
    total = sum(totals.values())
    print(f"== {path}: atoms {natoms} edges {edges} k {kcount}  kernel total {total:.0f} us")
    for name, us in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"   {us:8.0f} us  {100 * us / total:5.1f} %  {name}")
