"""A mid-run fallback on the GPU (plan task 5-T08; SPEC §9.4 step 5, §18 decision 54), untimed.

A fresh process, in the hybrid environment when it is there, runs MH-1 (``mh1-r2scan``) on NaCl 3x3x3 at
600 K with ``EngineConfig(graph_variant_cap=1)`` (every new edge size re-captures, as in 5-T06's storm) and a
restart every 50 steps, adaptation off. After the first restart write a broken capture is armed: a host
synchronisation inside the stream capture (``torch.nonzero``, ``cudaErrorStreamCaptureUnsupported``), injected
in the child only (monkeypatched, never a hook in the library), as 2-T15's ``capture_error`` case does before
the first step. The next re-capture fails during ``run()``, and the run must:

- go back to the restart point at step 50 and end on a validated revision (no graph tier) at step 200;
- record it in the revision chain (the point's step, rule 14's decision), ``counters.fallbacks`` and the report;
- leave the thermo rows without duplicates or gaps (the chunks after the point deleted, as on resume);
- agree statistically with an unbroken compiled run of the same seed after the point.
"""

import importlib.util
from pathlib import Path

import pytest


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_fallback_mid_run_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1")]

CHILD = """
import fastmace
import torch
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from fastmace import integrators
from fastmace.config import AdaptationConfig, EngineConfig
from fastmace.engine import graph
from fastmace.io.output import Output, read_thermo
from fastmace.model import compile as compiling
from fastmace.model import loading
from fastmace.model.forcefield import ForceField
from fastmace.planner import fallback
from fastmace.planner.workload import Workload
from fastmace.simulation import Simulation

ARMED = {"on": False, "hits": []}
original_stage = graph.GraphEngine._stage


def _stage(self, key):
    run = original_stage(self, key)
    if not self.capture:
        return run

    def unsupported():  # a host sync inside the capture: cudaErrorStreamCaptureUnsupported
        run()
        if ARMED["on"] and torch.cuda.is_current_stream_capturing():
            ARMED["on"] = False
            ARMED["hits"].append(int(self.state.step) + 1)
            torch.nonzero(self.state.forces)

    return unsupported


graph.GraphEngine._stage = _stage
compiling.configure_allocator()  # before the load touches CUDA (5-T07's finding)
loaded = loading.load_pristine(SPEC["label"])


def new(tag, mode=None):
    from ase.build import bulk
    import numpy as np

    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 3))
    MaxwellBoltzmannDistribution(atoms, temperature_K=600.0, rng=np.random.default_rng(11))
    directory = SPEC["directory"] + "/" + tag
    sim = Simulation(atoms, ForceField(loaded, atoms, precision="float32"),
                     integrator=integrators.Langevin(timestep_fs=2.0, temperature_K=600.0, friction_per_fs=0.02),
                     seed=7, workload=Workload(max_temperature_K=600.0), mode=mode,
                     engine=EngineConfig(graph_variant_cap=1), adaptation=AdaptationConfig(adapt="off"),
                     output=Output(directory, thermo_every=5, trajectory_every=None, restart_every=50))
    return sim, directory


def thermo(directory):
    header, rows = read_thermo(directory)
    columns = [header.index(name) for name in ("step", "temperature_K", "potential_eV_atom")]
    return [[float(row[column]) for column in columns] for row in rows]


out = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "fastmace": fastmace.__version__}
sim, directory = new("broken")
out["initial"] = {"tier": sim.plan.tier, "backend": sim.plan.engine.backend, "dtype": sim.plan.precision.dtype}
sim.run(steps=50)
ARMED["on"] = True
started = time.perf_counter()
sim.run(steps=150)
out["second_run_s"] = time.perf_counter() - started
plan = sim.plan
committed = sim.committed()
decision = plan.decision("fallback.mid_run")
out["broken"] = {
    "hits": ARMED["hits"], "tier": plan.tier, "backend": plan.engine.backend, "revision": plan.revision,
    "plan_step": plan.step, "reason": str(plan.reason), "chain": [[p.revision, p.step, str(p.reason)]
                                                                  for p in plan.chain()],
    "decision": None if decision is None else [decision[1], decision[2].rule, decision[2].code],
    "step": committed.step, "finite": bool(np.isfinite(committed.forces).all()),
    "restart_points": [point["step"] for point in sim._restart_points],
    "fallbacks": fallback.mid_run_fallbacks(plan)}
report = sim.report.text()
sim.close()
out["broken"]["metadata_fallbacks"] = json.load(open(directory + "/metadata.json"))["counters"]["fallbacks"]
out["broken"]["thermo"] = thermo(directory)
out["broken"]["report_line"] = [line for line in report.splitlines() if fallback.MID_RUN_CODE in line]

sim, directory = new("compiled", mode="compiled")
sim.run(steps=200)
sim.close()
out["compiled"] = {"tier": sim.plan.tier, "thermo": thermo(directory)}
print(json.dumps(out, default=str))
"""


def _mean(rows, column, first_step):
    values = [row[column] for row in rows if row[0] > first_step]
    return sum(values) / len(values)


def test_a_broken_recapture_falls_back_from_the_last_restart_point(tmp_path):
    spec = {"temperature": 600.0, "label": "mh1-r2scan", "directory": str(tmp_path)}
    out = ENGINE.run_fresh(ENGINE.COMMON.format(spec=spec) + CHILD, python=ENGINE.HYBRID_PYTHON)
    broken, compiled = out["broken"], out["compiled"]
    ENGINE.report("5-T08-fallback-mid-run-mh1-nacl",
                  {**{key: value for key, value in out.items() if key not in ("broken", "compiled")},
                   "broken": {key: value for key, value in broken.items() if key != "thermo"}})
    print(out["initial"], broken["hits"], broken["tier"], broken["reason"])

    assert out["initial"]["tier"] == "graph"
    (failed,) = broken["hits"]
    assert failed > 50
    assert broken["step"] == 200 and broken["finite"] and broken["tier"] != "graph"
    assert broken["backend"] == out["initial"]["backend"] and broken["plan_step"] == 50
    assert broken["reason"].startswith("fallback: ") and "(the last restart point; capture error at step" in \
        broken["reason"]
    assert broken["decision"] == [f"graph -> {broken['tier']}", 14, "fallback-mid-run"]
    (entry,) = broken["fallbacks"]
    assert broken["metadata_fallbacks"] == broken["fallbacks"]
    assert (entry["step"], entry["from"], entry["to"]) == (50, "graph", broken["tier"])
    assert broken["report_line"] and f"step {failed:,}" in broken["report_line"][0]
    assert [int(row[0]) for row in broken["thermo"]] == list(range(0, 201, 5))
    assert broken["restart_points"] == [50, 100, 150, 200]

    # After the point, statistically the unbroken compiled run (step, temperature, potential per atom).
    for column, tolerance in ((1, 60.0), (2, 0.005)):
        assert abs(_mean(broken["thermo"], column, 70) - _mean(compiled["thermo"], column, 70)) < tolerance
