"""The A100 plan's production order for the follow-up A/B (jobs/alex_lp_skip.sh): render with STAGE_EDGES and
skip_outside=True (as WPrime does with SKIP_OUTSIDE on), then LATE_PULLBACK, then HOIST_DX. Checks that the transform
takes the render, that the result compiles for sm_80 and matches census_followup's source (wprime_tuning.skip on the
A100 render) apart from the skip's spelling, and emulates it in float64 (before the hoist) -> a100_skip_render.json.
"""
import json
import sys

from common import H200, HERE, setup
from emulate2 import check

sys.path.insert(0, str(H200))
from render_sources import sources as rendered  # noqa: E402
from census_sass import census  # noqa: E402

s = setup()
from fastmace.kernels.oeq_streaming import wprime_backward as wp  # noqa: E402
from fastmace.kernels.oeq_streaming import wprime_pullback as wpb  # noqa: E402
from fastmace.kernels.oeq_streaming import wprime_tuning as wt  # noqa: E402
from fastmace.kernels.oeq_streaming.runtime import NUM_THREADS  # noqa: E402

schedule, grid, _, _ = rendered()
common = dict(vector=4, table_w=True, blocks=s.blocks, factored=True, scale=True, horner=True, scalar_paths=True,
              butterfly=True, stage_dout=True, fold_alpha=True, spline=grid, stage_sh=NUM_THREADS // 32)
src = wp.render(schedule, "float", 4, stage_edges=True, skip_outside=True, **common)
lp = wpb.apply(src, s.schedule, s.blocks, True)
(HERE / "sources" / "fu-a100-render-skip-lp-nohoist.cu").write_text(lp)
final = wt.hoist(lp)
(HERE / "sources" / "fu-a100-render-skip-lp.cu").write_text(final)
res = dict(paths=lp.count("// LATE_PULLBACK path"), hoisted="dx_lane" in final)
res["emulation"] = check(s, "sources/fu-a100-render-skip-lp-nohoist.cu")
r = census(str(HERE / "sources" / "fu-a100-render-skip-lp.cu"), 80, (), str(HERE / "sass"))
res["sm80"] = dict(registers=r["registers"], local=r["local"], edge_loop=r["per_edge"]["instructions"])
print(json.dumps(res))
(HERE / "a100_skip_render.json").write_text(json.dumps(res, indent=1) + "\n")
