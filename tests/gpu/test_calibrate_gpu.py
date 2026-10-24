"""The optional calibration on the GPU (plan task 5-T09; SPEC §9.1, decision 55).

A fresh process of the hybrid environment (``FASTMACE_HYBRID_PYTHON``) runs MH-1 through
``Simulation(calibration=CalibrationConfig(calibrate=True))`` on NaCl 3x3x3 at 600 K with the production defaults.
Hybrid (OEq builds) ties with plain cuEq, so four fresh children of the subprocess runner build the candidates
one after another (A B A B), each timing 50 steps after 50 warm-up steps; then the run plans in this process
and runs.

- Untimed (``slow``): the record has both candidates with two positive block times each, the setup time
  holds the calibration's cost, the report and provenance show the measured times, the hand-off directory is
  gone, and the run steps on the plan it chose.
- Timed only as a check (``acceptance``, 5-T11's quiet window, the RTX 5060 Ti): the calibration keeps hybrid
  (the rules' choice; 1a-T00 measured hybrid faster on NaCl). Off the reference GPU the choice is recorded,
  not judged (plans/phase-5.md, runbook item 9).
"""

import importlib.util
from pathlib import Path

import pytest


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_calibrate_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")

CHILD = """
import json
import os

import fastmace
import numpy as np
import torch
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from fastmace import integrators
from fastmace.config import CalibrationConfig
from fastmace.io.output import Output
from fastmace.model import loading
from fastmace.model.forcefield import ForceField
from fastmace.planner.workload import Workload
from fastmace.simulation import Simulation

SPEC = {spec!r}
loaded = loading.load_pristine("mh1-r2scan")
atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 3))
MaxwellBoltzmannDistribution(atoms, temperature_K=600.0, rng=np.random.default_rng(11))
directory = SPEC["directory"] + "/run"
sim = Simulation(atoms, ForceField(loaded, atoms, precision="float32"),
                 integrator=integrators.Langevin(timestep_fs=2.0, temperature_K=600.0, friction_per_fs=0.02),
                 seed=7, workload=Workload(max_temperature_K=600.0), calibration=CalibrationConfig(calibrate=True),
                 output=Output(directory, thermo_every=10, trajectory_every=None))
record = sim.plan.calibration
handoff_left = os.path.exists(directory + "/.fastmace-handoff")
sim.run(steps=SPEC["steps"])
sim.close()
metadata = json.load(open(directory + "/metadata.json"))
print(json.dumps({{
    "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
    "backend": sim.plan.engine.backend, "backend_code": sim.plan.decision("EngineConfig.backend")[2].code,
    "candidates": [[c.change, list(c.step_ms)] for c in record.candidates], "chosen": record.chosen,
    "changed": record.changed_plan, "record_setup_s": record.setup_s, "setup_s": sim.report.setup_s,
    "report": sim.report.text(), "plan_text": str(sim.plan), "metadata": metadata["plan"]["calibration"],
    "step": sim.committed().step, "handoff_left": handoff_left}}))
"""


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory):
    if ENGINE._mh1_path() is None:
        pytest.skip("needs FASTMACE_MODELS with MH-1")
    spec = {"steps": 100, "directory": str(tmp_path_factory.mktemp("calibrate"))}
    # The hybrid environment (OEq installed): the calibration's children use the same Python.
    result = ENGINE.run_fresh(CHILD.format(spec=spec), python=ENGINE.HYBRID_PYTHON)
    print({key: result[key] for key in ("gpu", "backend", "candidates", "chosen", "record_setup_s")})
    return result


@pytest.mark.slow
def test_the_calibration_measures_both_backends_in_children(calibrated):
    result = calibrated
    (rules, rules_ms), (other, other_ms) = result["candidates"]
    assert (rules, other) == ("", "backend cueq") and len(rules_ms) == len(other_ms) == 2
    assert all(ms > 0 for ms in rules_ms + other_ms)
    assert result["setup_s"]["calibration"] == result["record_setup_s"] > 0
    backend_line = next(line for line in result["report"].splitlines() if line.startswith("backend:"))
    assert "measured in this run" in backend_line and "speed-up" not in backend_line
    assert "calibration: " in result["plan_text"] and result["metadata"]["chosen"] == result["chosen"]
    assert result["backend"] == ("cueq" if result["changed"] else "hybrid")
    assert result["backend_code"] == ("backend-calibrated" if result["changed"] else "backend-hybrid")
    assert result["step"] == 100 and not result["handoff_left"]


@pytest.mark.acceptance
def test_the_calibration_keeps_hybrid_on_nacl(calibrated):
    """Timed: judged only on the reference GPU in a quiet window (plans/phase-5.md, runbook item 9)."""
    if "5060" not in calibrated["gpu"]:
        pytest.skip(f"the choice is judged on the RTX 5060 Ti only, not on {calibrated['gpu']}")
    assert calibrated["chosen"] == "" and calibrated["backend"] == "hybrid", calibrated["candidates"]
