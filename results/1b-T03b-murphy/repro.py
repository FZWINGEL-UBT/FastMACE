"""1b-T03b: the FastMACE child of test_mh1_hybrid_float32_against_the_research_engine, shortened.

Runs eager then checked (with the capture warm-up check) in the hybrid environment, with the float64
energy sum on and off (argument), and prints whether the capture raised.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests" / "gpu"), str(ROOT / "tests")]
import test_engine_gpu as t  # noqa: E402

flag = sys.argv[1] == "on"
modes = [("eager", int(sys.argv[2])), ("checked", 100)] if len(sys.argv) > 2 else [("checked", 100)]
spec = t.spec("hybrid", modes=modes, warmups=True, float64_energy=flag)
try:
    out = t.fastmace_run(spec, python=t.HYBRID_PYTHON)
    print(json.dumps({"float64_energy": flag, "modes": [m for m, _ in modes], "ok": True,
                      "warmups": out["warmups"]}))
except AssertionError as error:
    text = str(error)
    print(json.dumps({"float64_energy": flag, "modes": [m for m, _ in modes], "ok": False,
                      "stale": "stale reference" in text, "tail": text[-1500:]}))
