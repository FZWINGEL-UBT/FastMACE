"""Split-kernel question (12 Nov 2026): the H200 default w' source (skip+hoist) with dx and dr removed, keeping only
dsh: the second kernel of a dx/dr + dsh split. With nodsh_variant.py's output it gives the split's instruction sum.

    python dshonly_variant.py sources/wprime-h200-skip+hoist.cu sources/wprime-h200-skip+hoist-dshonly.cu
"""
import re
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
n = {}
src, n["u_acc"] = re.subn(r"\n\s*u_\d+ [+-]= scaled \* edge_value;", "", src)
src, n["wg_acc"] = re.subn(r"\n\s*weight_gradient \+=\s*\n\s*x\[[^\]]*\] \* u_\d+;", "", src)
src, n["dx_acc"] = re.subn(r"\n\s*input_gradient_\d+_\d+ \+= scaled_weight \* u_\d+;", "", src)
src, n["slope"] = re.subn(r"\n\s*if \(sloped\) \{\s*\n\s*acc_t slope = acc_t\(0\);\s*\n\s*slope = radial_dt \* sc;\s*\n"
                          r"\s*radial_gradient \+= weight_gradient \* slope;\s*\n\s*\}", "", src)
src, n["dx_atomic"] = re.subn(r"\n\s*atomicAdd\(\s*\n\s*dx_lane \+ \d+,\s*\n\s*scalar_t\(input_gradient_\d+_\d+\)\);", "",
                              src)
src, n["u_decl"] = re.subn(r"\n\s*acc_t u_\d+ = acc_t\(0\);", "", src)
src, n["wg_decl"] = re.subn(r"\n\s*acc_t weight_gradient = acc_t\(0\);", "", src)
src, n["ig_decl"] = re.subn(r"\n\s*acc_t input_gradient_\d+_\d+ = acc_t\(0\);", "", src)
assert not re.search(r"\bu_\d+\b", src), "a u_i is left"
assert "weight_gradient" not in src and "input_gradient_" not in src
Path(sys.argv[2]).write_text(src)
print(n)
