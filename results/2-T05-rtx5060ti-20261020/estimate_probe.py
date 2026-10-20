"""2-T05: why did the CPU step's estimate let the 1944-atom POLAR-1-L reference start? CPU only.

Calibrates the float64 CPU reference on the two boxes, prints the estimates for NbCl5 with 288, 648
and 1944 atoms, and measures the real peak of one float64 evaluation on 288 and 648 atoms. The
process caps its own data segment at 6 GiB first (selfcheck.cap_memory), so it cannot exhaust WSL.
"""

import json
import sys
import time
from types import SimpleNamespace

import fastmace  # noqa: F401  (before torch)
from fastmace.config import GateConfig, PolarConfig
from fastmace.model import loading
from fastmace.model.forcefield import ForceField
from fastmace.planner import selfcheck

sys.path.insert(0, "tests")
from _support import structures  # noqa: E402

label = sys.argv[1] if len(sys.argv) > 1 else "polar1-l"
CAP = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0


def phase(name):
    print("phase", name, "VmData GiB %.2f VmRSS GiB %.2f" % (selfcheck._status_kib("VmData") / 2**30,
                                                            selfcheck._status_kib("VmRSS") / 2**30), flush=True)


phase("start")
print("cap", selfcheck.cap_memory(CAP * 2**30), flush=True)
loaded = loading.load_pristine(label)
cells = {n: structures.nbcl5(repeat=1).repeat(r) for n, r in ((288, (2, 2, 2)), (648, (3, 3, 2)), (1944, (3, 3, 6)))}
for atoms in cells.values():
    atoms.rattle(0.03, seed=1)
ff = ForceField(loaded, cells[1944])
plan = SimpleNamespace(polar=PolarConfig() if selfcheck._polar_config({"polar": {}}) is not None else None,
                       engine=SimpleNamespace(device="cpu", backend="cueq"))
plan.polar = PolarConfig() if loaded.family == "polar" else None
job = selfcheck.cpu_job(plan, ff, [("production", None, a, selfcheck.make_copies(a, selfcheck._integrator_config(
    __import__("fastmace.config", fromlist=["IntegratorConfig"]).IntegratorConfig(temperature_K=300.0)), 1))
    for a in cells.values()][::-1], cells[1944], GateConfig(cpu_float64_time_limit_s=1e-9))
phase("job built")
reference = selfcheck._load_reference(job, "float64", selfcheck._atoms(job["calibration"][0]))
boxes = [selfcheck._atoms(entry) for entry in job["calibration"]]
phase("reference loaded")
points = selfcheck._calibrate(reference, boxes, job["cutoff"])
phase("calibrated")
print("calibration", json.dumps(points), flush=True)
for candidate in job["candidates"]:
    estimate = selfcheck.estimate_cost(points, candidate, polar=job["polar"] is not None)
    print("estimate", candidate["atoms"], candidate["edges"], json.dumps(estimate), flush=True)
for n in (288, 648):
    atoms = cells[n]
    if reference.is_polar:
        reference.rebase(atoms)
    resident = selfcheck._status_kib("VmRSS") if selfcheck._reset_peak() else None
    started = time.perf_counter()
    try:
        reference.evaluate(atoms)
        peak = selfcheck._status_kib("VmHWM") - resident
        print("measured", n, selfcheck.edge_count(atoms, job["cutoff"]), time.perf_counter() - started, peak, flush=True)
    except (MemoryError, RuntimeError) as error:
        print("measured", n, "out of memory under the cap:", str(error)[:120], flush=True)
        break
