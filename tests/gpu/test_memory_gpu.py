"""Rule 12's estimate against the real peak on this GPU (plan task 5-T03; SPEC §9.3 rule 12, decision 51).

In the compatibility tier (``-m compat``), so every production GPU type checks it: MH-1 ``matpes_r2scan`` with
the planner's default plan (hybrid float32 where the OpenEquivariance runtime is there), NaCl 3x3x3, Langevin
NVT on the graph tier, in a fresh process. The plan's ``predicted_memory_bytes`` must not be below
``torch.cuda.max_memory_allocated`` over the build, the capture and 100 steps; the plan's ``memory`` decision is
rule 12's, and on a free GPU it fits. The ratio is recorded (``FASTMACE_RESULTS``); the 19-label validation is
``docs/validation/5-T03.md`` (``tools/memory_peaks.py``).
"""

import importlib.util
from pathlib import Path

import pytest


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_memory_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")

pytestmark = [
    pytest.mark.gpu, pytest.mark.slow, pytest.mark.compat,
    pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1"),
]

PEAK_CHILD = """
import json
import fastmace
import fastmace as fm
import torch
torch.cuda.reset_peak_memory_stats()
atoms = structure()
sim = fm.Simulation(atoms, fm.ForceField(fm.load("mh1-r2scan"), atoms),
                    integrator=fm.integrators.Langevin(timestep_fs=1.0, temperature_K=300.0,
                                                       friction_per_fs=0.01),
                    seed=1, workload=fm.Workload(max_temperature_K=300.0))
sim.run(steps=100)
torch.cuda.synchronize()
_, choice, reason = sim.plan.decision("memory")
out = {"predicted": sim.plan.predicted_memory_bytes, "peak": int(torch.cuda.max_memory_allocated()),
       "choice": choice, "code": getattr(reason, "code", None), "rule": getattr(reason, "rule", None),
       "reason": str(reason), "tier": sim.plan.tier, "backend": sim.plan.engine.backend,
       "dtype": sim.plan.precision.dtype, "gpu": torch.cuda.get_device_name(),
       "capability": list(torch.cuda.get_device_capability())}
sim.close()
print(json.dumps(out, default=str))
"""


@pytest.mark.rule(12)
def test_the_prediction_covers_the_measured_peak():
    spec = {"temperature": 300.0, "steps": 0, "timed": 0, "every": 1}
    out = ENGINE.run_fresh(ENGINE.COMMON.format(spec=spec) + PEAK_CHILD)
    out["ratio"] = out["predicted"] / out["peak"]
    ENGINE.report("memory_rule_12", out)
    assert out["tier"] == "graph"
    assert (out["rule"], out["code"], out["choice"]) == (12, "rule-12-memory", "fits")
    assert out["predicted"] >= out["peak"], out
