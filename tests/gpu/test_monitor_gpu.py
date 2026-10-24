"""The runtime monitor on the GPU (plan task 5-T06; SPEC §9.6), untimed smoke.

Fresh processes (one per run) run MH-1 through ``Simulation`` with the production defaults (CUDA graphs, pipelined) on
NaCl 3x3x3 at 600 K, a restart every 20 steps. ``EngineConfig(graph_variant_cap=1)`` forces a re-capture
storm: each stage's replay evicts the other's graph, so every step re-captures. The counters must move:
re-captures with their host time (from ``GraphEngine._capture``'s own synchronisations, none added to
the step), a segment per restart write and one at the end of ``run()``, ``counters.segments`` in
``metadata.json`` and the report's ``runtime`` line. The control run (the default cap) re-captures
less, and has the line only if it re-captured or redid an overflow. The step time is not judged here: the alternating
A/B for the host timestamps is phase 5's acceptance item 5 (plan task 5-T11).
"""

import importlib.util
import sys
from pathlib import Path

import pytest


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_monitor_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")

CHILD = """
import json

import fastmace
import numpy as np
import torch
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from fastmace import integrators
from fastmace.config import EngineConfig
from fastmace.io.output import Output
from fastmace.model import loading
from fastmace.model.forcefield import ForceField
from fastmace.planner.workload import Workload
from fastmace.simulation import Simulation

SPEC = {spec!r}
loaded = loading.load_pristine(SPEC["label"])  # the label names the head (MH-1 has six)
atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 3))
MaxwellBoltzmannDistribution(atoms, temperature_K=600.0, rng=np.random.default_rng(11))
engine = {{}} if SPEC["cap"] is None else {{"engine": EngineConfig(graph_variant_cap=SPEC["cap"])}}
directory = SPEC["directory"]
# Nothing touches CUDA before the Simulation: the allocator is configured at its build (SPEC §8.7).
sim = Simulation(atoms, ForceField(loaded, atoms, precision="float32"),
                 integrator=integrators.Langevin(timestep_fs=2.0, temperature_K=600.0, friction_per_fs=0.02),
                 seed=7, workload=Workload(max_temperature_K=600.0),
                 output=Output(directory, thermo_every=20, trajectory_every=None, restart_every=20), **engine)
sim.run(steps=SPEC["steps"])
totals = sim._monitor.totals()
info = sim._engine.info()
sim.close()
metadata = json.load(open(directory + "/metadata.json"))
out = {{"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "plan": [sim.plan.tier, sim.plan.engine.backend], "segments": sim._monitor.records(),
        "totals": totals.record(), "engine": {{key: info[key] for key in (
            "captures", "capture_seconds", "recaptures", "recapture_seconds", "recapture_budget")}},
        "metadata_segments": metadata["counters"]["segments"],
        "neighbour_lists": metadata["counters"]["neighbour_lists"],
        "report": sim.report.text()}}
print(json.dumps(out))
"""


@pytest.mark.slow
@pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1")
def test_a_recapture_storm_moves_the_monitor_counters(tmp_path):
    runs = {}
    for tag, cap in (("storm", 1), ("control", None)):  # one fresh process each: one production engine
        spec = {"label": "mh1-r2scan", "steps": 60, "directory": str(tmp_path / tag), "cap": cap}
        runs[tag] = ENGINE.run_fresh(CHILD.format(spec=spec), python=sys.executable)
        print(tag, runs[tag]["gpu"], runs[tag]["plan"], runs[tag]["totals"], runs[tag]["engine"])

    storm, control = runs["storm"], runs["control"]
    assert storm["plan"][0] == "graph"
    reasons = [(entry["last_step"], entry["reason"], entry["canonical"]) for entry in storm["segments"]]
    assert reasons == [(20, "restart", True), (40, "restart", True), (60, "restart", True)]
    totals = storm["totals"]
    assert totals["recaptures"] > 0 and totals["recapture_s"] > 0.0
    assert totals["captures"] >= totals["recaptures"] and totals["capture_s"] >= totals["recapture_s"]
    assert totals["recapture_s"] < totals["wall_s"]  # host time inside the loop's wall time
    assert totals["recaptures"] <= storm["engine"]["recaptures"]  # the setup's captures are not in the segments
    assert totals["budget_exceeded_at"] is not None  # the storm passes the budget
    assert storm["metadata_segments"] == storm["segments"]
    assert storm["neighbour_lists"]["mace"]["build_time_s"] is None
    assert "runtime: re-captures" in storm["report"]

    quiet = control["totals"]
    assert quiet["recaptures"] < totals["recaptures"]
    assert ("runtime:" in control["report"]) == bool(quiet["recaptures"] or quiet["overflows_redone"])
