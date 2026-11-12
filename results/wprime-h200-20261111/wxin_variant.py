"""Review idea 1 (12 Nov 2026): SCALAR_PATHS' weighted input extended to vector-input paths. In a path block whose
v_j sums have more than MIN_J components, dy_j += s (w x_i) per term replaces v_j += s x_i and the final
dy_j += w v_j (net J - 3 fewer multiplies per path and channel). Works on the skip+hoist and skip+vec2 sources.

    python wxin_variant.py IN.cu OUT.cu [MIN_J]     (MIN_J 3: only J > 3, the proposal; 0: every vector path)
"""
import re
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
min_j = int(sys.argv[3]) if len(sys.argv) > 3 else 3
marker = "acc_t weight_gradient = acc_t(0);"
parts = src.split(marker)
changed = 0
for n in range(1, len(parts)):
    seg = parts[n]
    # the block ends at the next marker; its apply lines map v_k -> edge_gradient[J]
    apply = dict((k, j) for j, k in re.findall(r"edge_gradient\[(\d+)\] \+=\s*\n\s*scaled_weight \* v_(\d+);", seg))
    if not apply or len(apply) <= min_j:
        continue
    def term(m):
        sign, k = m.group(2), m.group(1)
        return f"\n{m.group(0)[1:m.group(0).index('v_')]}edge_gradient[{apply[k]}] {sign}= scaled * (radial_weight * input_value);"
    seg, a = re.subn(r"\n\s*v_(\d+) ([+-])= scaled \* input_value;", term, seg)
    seg, b = re.subn(r"\n\s*edge_gradient\[\d+\] \+=\s*\n\s*scaled_weight \* v_\d+;", "", seg)
    seg, c = re.subn(r"\n\s*acc_t v_\d+ = acc_t\(0\);", "", seg)
    assert b == c == len(apply), (a, b, c, apply)
    assert not re.search(r"\bv_\d+\b", seg.split("acc_t weight_gradient")[0] if False else seg[:seg.find("if (sloped)")])
    parts[n] = seg
    changed += 1
Path(sys.argv[2]).write_text(marker.join(parts))
print("path blocks changed:", changed)
