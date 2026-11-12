"""WEIGHTED_VECTOR (12 Nov 2026): the template switch against wxin_variant.py, and its sources for the census."""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import render_sources as rs  # noqa: E402
from fastmace.kernels.oeq_streaming import wprime_backward as wp, wprime_tuning as wt  # noqa: E402
from fastmace.kernels.oeq_streaming.runtime import NUM_THREADS, render_kernels  # noqa: E402

schedule, _ = render_kernels(rs.problem(), rs.np.float32, NUM_THREADS, "ir_mul")
grid = rs.grid_from(rs.CAPTURED.read_text())
common = dict(vector=4, table_w=True, blocks=rs.blocks_of(schedule), factored=True, scale=True, horner=True,
              scalar_paths=True, butterfly=True, stage_dout=True, fold_alpha=True, spline=grid,
              stage_sh=NUM_THREADS // 32)
src = HERE / "sources"


def code(text):
    return [line.strip() for line in re.sub(r"//[^\n]*", "", text).splitlines() if line.strip()]


for gpu, stage_edges in (("h200", False), ("a100", True)):
    base = wp.render(schedule, "float", 4, stage_edges=stage_edges, **common)
    weighted = wp.render(schedule, "float", 4, stage_edges=stage_edges, weighted_vector=True, **common)
    for name, text in ((f"wprime-{gpu}-skip+hoist.cu", wt.variant(base, "skip+hoist")),
                       (f"wprime-{gpu}-weighted.cu", weighted),
                       (f"wprime-{gpu}-skip+hoist+weighted.cu", wt.variant(weighted, "skip+hoist"))):
        (src / name).write_bytes(text.encode())
    subprocess.run([sys.executable, str(HERE / "wxin_variant.py"), str(src / f"wprime-{gpu}-skip+hoist.cu"),
                    str(HERE / "_wxin_ref.cu")], check=True)
    ref = (HERE / "_wxin_ref.cu").read_text()
    (HERE / "_wxin_ref.cu").unlink()
    mine = (src / f"wprime-{gpu}-skip+hoist+weighted.cu").read_text()
    a, b = code(ref), code(mine)
    print(gpu, "template == wxin_variant.py (code lines):", a == b, len(a), len(b))
    if a != b:
        import difflib
        print("\n".join(list(difflib.unified_diff(a, b, lineterm="", n=0))[:20]))
