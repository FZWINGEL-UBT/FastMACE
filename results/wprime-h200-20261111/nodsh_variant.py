"""Idea 4 falsifier (tensor cores for w''s dsh channel reduction, 12 Nov 2026): the H200 default w' source
(skip+hoist) with every dsh contribution removed, for census_sass.py. What a tensor-core dsh could remove at most:
the v_j sums, the w v_j updates, SCALAR_PATHS' per-term w x0 c g_o adds and (folded by the compiler once the
per-lane edge gradient is constant zero) the butterfly reduction of the edge gradient.

    python nodsh_variant.py sources/wprime-h200-skip+hoist.cu sources/wprime-h200-skip+hoist-nodsh.cu
"""
import re
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
n = {}
# v_j accumulations inside the term blocks
src, n["v_acc"] = re.subn(r"\n\s*v_\d+ [+-]= scaled \* input_value;", "", src)
# FACTORED's dsh_j += w v_j
src, n["v_apply"] = re.subn(r"\n\s*edge_gradient\[\d+\] \+=\s*\n\s*scaled_weight \* v_\d+;", "", src)
# SCALAR_PATHS' per-term edge gradient
src, n["scalar_edge"] = re.subn(r"\n\s*edge_gradient\[\d+\]\s*\n\s*\+= weighted_input \* scaled;", "", src)
# the now-unused declarations
src, n["v_decl"] = re.subn(r"\n\s*acc_t v_\d+ = acc_t\(0\);", "", src)
src, n["wi_decl"] = re.subn(r"\n\s*const scalar_t weighted_input =\s*\n\s*radial_weight \* x\[[^\]]*\];", "", src)
left = [m for m in ("v_", "weighted_input") if re.search(rf"\b{m}\d*\b", src) and m == "weighted_input"]
assert not re.search(r"\bv_\d+\b", src), "a v_j is left"
assert "weighted_input" not in src, "weighted_input is left"
assert not re.search(r"edge_gradient\[\d+\]\s*\+?=\s*[^;]*(v_|weighted)", src)
Path(sys.argv[2]).write_text(src)
print(n)
