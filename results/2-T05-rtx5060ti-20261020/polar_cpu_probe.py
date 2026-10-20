"""2-T05: peak memory of one CPU float64 POLAR-1-L reference evaluation, natural cells against a box.

Caps its data segment first (argv[1] GiB, default 9). Prints the peak resident growth per evaluation.
"""

import sys

import fastmace  # noqa: F401  (before torch)
from fastmace.config import PolarConfig
from fastmace.planner import selfcheck

sys.path.insert(0, "tests")
from _support import structures  # noqa: E402

cap = float(sys.argv[1]) if len(sys.argv) > 1 else 9.0
print("cap", selfcheck.cap_memory(cap * 2**30), flush=True)
base = structures.nbcl5(repeat=1)
job = {"source": "polar1-l", "head": None, "sha256": None, "charge": None, "spin_multiplicity": None,
       "family": None, "device": "cpu", "polar": {}, "backend": "cueq"}
job["polar"] = {k: v for k, v in PolarConfig().__dict__.items()}
reference = selfcheck._load_reference(job, "float64", base)
print("loaded RSS GiB %.2f" % (selfcheck._status_kib("VmRSS") / 2**30), flush=True)
cases = [("primitive 36", base), ("1x1x2 72", base.repeat((1, 1, 2))), ("2x2x2 288", base.repeat((2, 2, 2)))]
cases += [(f"box {len(b)}", b) for b in selfcheck.calibration_boxes(base, 6.0)]
for name, atoms in cases:
    reference.rebase(atoms)
    resident = selfcheck._status_kib("VmRSS")
    selfcheck._reset_peak()
    try:
        reference.evaluate(atoms)
        print(name, "edges", selfcheck.edge_count(atoms, 6.0), "peak growth GiB %.3f" % (
            (selfcheck._status_kib("VmHWM") - resident) / 2**30), "polar", reference.polar_info, flush=True)
    except (MemoryError, RuntimeError) as error:
        print(name, "out of memory under the cap:", str(error)[:100], "polar", reference.polar_info, flush=True)
