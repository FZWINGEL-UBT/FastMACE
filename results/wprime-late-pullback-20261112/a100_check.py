"""The A100 render (STAGE_EDGES, no SKIP_OUTSIDE/HOIST_DX) through LATE_PULLBACK: written to sources/a100-lp.cu and
checked with emulate2.py's float64 emulation (sources/a100-lp-check.json)."""
import json
from common import H200, HERE, setup
from emulate2 import check

s = setup()
from fastmace.kernels.oeq_streaming import wprime_pullback as wpb
src = (H200 / "sources" / "wprime-a100.cu").read_text()
out = wpb.apply(src, s.schedule, s.blocks, True)
(HERE / "sources" / "a100-lp.cu").write_text(out)
res = check(s, "sources/a100-lp.cu")
print(json.dumps(res))
(HERE / "a100_check.json").write_text(json.dumps(res, indent=1) + "\n")
