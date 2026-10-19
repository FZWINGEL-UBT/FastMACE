"""1b-T03b: stage B's kernels of FastMACE plain cuEq float64 with and without the float64 energy sum."""

import json
import sys

sys.path[:0] = ["tests/gpu", "tests"]
import test_engine_gpu as t  # noqa: E402

runs = {flag: t.fastmace_run(t.spec("cueq", modes=[("checked", 100)], float64_energy=flag))["modes"]["checked"]
        for flag in (False, True)}
summary = {}
for key, off in runs[False]["kinds"].items():
    on = runs[True]["kinds"].get(key)
    a, b = t.kernel_sequence(off), t.kernel_sequence(on or [])
    extra = [name for name in b if name not in a]
    summary[key] = {"nodes_off": len(off), "nodes_on": len(on or []), "equal": a == b,
                    "only_with_float64_sum": sorted(set(extra)),
                    "off_is_subsequence": all(name in b for name in a)}
summary["positions_equal"] = runs[False]["positions"] == runs[True]["positions"]
print(json.dumps(summary, indent=1))
