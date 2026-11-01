"""The near-limit probe, offline on the rescue matrix: for each crystal, GPU and stage, its smallest cell of at least
15,000 atoms plays the probe's replica, and its measured peak corrects rule 12's estimate of every larger cell:

    corrected = F + (estimate - F) * (measured_replica - F_replica) / (estimate_replica - F_replica) * HEADROOM

where F is the estimate's size-independent part (weights, resident buffers, workspace). Prints corrected/measured per
stage for HEADROOM = 1.
    python probe_offline.py FM_SOURCE
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

FM = Path(sys.argv[1])
sys.path.insert(0, str(FM))
sys.path.insert(0, str(FM / "tests" / "cpu"))
import test_memory_fit as t  # noqa: E402  (predicted(case): today's estimate of a fit case)

FIXED = ("weights (pristine and execution copy)", "rewrites' buffers", "workspace")


def parts(case):
    est = t.predicted(case)
    return est.total, sum(est.term(name) for name in FIXED)


groups = defaultdict(list)
for case in t.RESCUE:
    groups[(case["structure"], case["gpu"], case["stage"])].append(case)
by_stage = defaultdict(list)
rows = []
for (structure, gpu, stage), cases in sorted(groups.items()):
    cases.sort(key=lambda c: c["natoms"])
    replica, rest = cases[0], cases[1:]
    est_r, fixed_r = parts(replica)
    scale = (replica["peak_bytes"] - fixed_r) / (est_r - fixed_r)
    for case in rest:
        est, fixed = parts(case)
        corrected = fixed + (est - fixed) * scale
        ratio = corrected / case["peak_bytes"]
        by_stage[stage].append(ratio)
        rows.append(dict(structure=structure, gpu=gpu, stage=stage, replica=replica["natoms"], atoms=case["natoms"],
                         ratio=ratio, rules=est / case["peak_bytes"]))
        print(f"{structure:7s} {gpu[-12:]:12s} {stage:13s} replica {replica['natoms']:6d} -> {case['natoms']:7d}: "
              f"probe-corrected/measured {ratio:.3f} (rules {est / case['peak_bytes']:.3f})")
print()
for stage, v in sorted(by_stage.items()):
    print(f"{stage:13s} n={len(v):2d} {min(v):.3f}-{max(v):.3f}")
allv = [r for v in by_stage.values() for r in v]
print(f"all n={len(allv)} {min(allv):.3f}-{max(allv):.3f}; headroom for >= 1.0: {1 / min(allv):.3f}, then "
      f"{1.0:.3f}-{max(allv) / min(allv):.3f}")
Path("probe_offline.json").write_text(json.dumps(rows, indent=1))
