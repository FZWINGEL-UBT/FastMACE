"""float64 emulation (emulate2.check) of census_followup's A100 lp + SKIP_OUTSIDE source (before HOIST_DX, which emulate2
reads as the unhoisted form) -> fu_check.json."""
import json
from common import HERE, setup
from emulate2 import check

res = check(setup(), "sources/fu-a100-lp-nohoist-skip.cu")
print(json.dumps(res))
(HERE / "fu_check.json").write_text(json.dumps(res, indent=1) + "\n")
