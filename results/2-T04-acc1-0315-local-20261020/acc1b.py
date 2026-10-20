"""2-T04 acceptance 1, diagnosis: the plan on 0.3.15, then check="skip" (where does the capture failure land?)."""
import fastmace  # noqa: F401

import sys
import traceback

from ase.build import bulk

import fastmace as fm
from fastmace.config import IntegratorConfig
from fastmace.planner import minimal
from fastmace.planner.workload import Workload

atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 3))
atoms.rattle(0.02, seed=1)
ff = fm.ForceField(fm.load("mh-1", head="matpes_r2scan"), atoms)
plan = minimal.default_planner(ff).plan(ff, atoms, IntegratorConfig(temperature_K=300.0),
                                        Workload(max_temperature_K=300.0))
for key in ("mode", "capture", "EngineConfig.backend", "EngineConfig.compile_mode", "EngineConfig.cueq_conv_fusion",
            "force_precision"):
    k, c, r = plan.decision(key)
    print(f"PLAN {key} = {c}  [{getattr(r, 'code', None)}] {str(r)[:160]}")
print("PLAN tier", plan.tier, "revision", plan.revision)
sys.stdout.flush()
try:
    run = fm.md(atoms, model="mh-1", head="matpes_r2scan", temperature_K=300.0, steps=20, seed=7,
                output=sys.argv[1] + "/skip", check="skip")
    print("SKIP-RUN ok tier", run.plan.tier, "revision", run.plan.revision, "reason", run.plan.reason)
    for t in run.plan.fallbacks:
        print("FALLBACK", t)
except Exception as error:
    print("SKIP-RUN FAILED", type(error).__name__, str(error).splitlines()[0])
    tb = traceback.extract_tb(error.__traceback__)
    for frame in tb[-12:]:
        if "fastmace" in frame.filename:
            print("  at", frame.filename.split("fastmace/")[-1], frame.lineno, frame.name)
