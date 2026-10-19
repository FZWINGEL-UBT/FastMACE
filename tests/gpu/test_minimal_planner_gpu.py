"""The minimal planner against the fixed-plan planner on the GPU (1b-T12's GPU check, run by 1b-T23).

MH-1 (``matpes_r2scan``) on NaCl 3x3x3, NVT 300 K, 500 steps, each engine in a fresh process: the engine
``MinimalPlanner`` plans and builds from an ``fm.ForceField`` (the public path of ``Simulation`` and
``fm.md``) against the one ``FixedPlanner`` plans and builds from main.py's defaults
(``tests/gpu/test_planner_gpu.py``), with the same seed. Plain cuEq float64 agrees within 1e-12 Å
(cuEq's run-time kernel choice); the hybrid within three times the scatter of two ``FixedPlanner`` runs.
The rewrite records, engine options, edge sizes and rebuilds are equal, and step times are reported.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLANNER = _module("_minimal_gpu_planner_tests", HERE / "test_planner_gpu.py")
ENGINE = PLANNER.ENGINE

MINIMAL_CHILD = """
import fastmace
from fastmace.config import EngineConfig, IntegratorConfig
from fastmace.model import loading
from fastmace.model.forcefield import ForceField
from fastmace.planner import fixed, minimal
from fastmace.planner.workload import Workload

planner = minimal.MinimalPlanner(engine=EngineConfig(backend=SPEC["planner_backend"]))
loaded = loading.load_pristine("mh1-r2scan")  # fm.load: prepares the OEq runtime before MACE is imported
hardware = planner.hardware()
import torch
atoms = structure()
force_field = ForceField(loaded, atoms, precision=SPEC["precision"])
integrator = IntegratorConfig(timestep_fs=0.5, friction=0.02, seed=SPEC["seed"], temperature_K=SPEC["temperature"])
plan = planner.plan(force_field, atoms, integrator, Workload(max_temperature_K=SPEC["temperature"]), hardware)
engine = planner.build(plan, force_field, atoms, temperature_K=SPEC["temperature"])
SPEC["steps"] = 500
out = protocol(torch, engine.step, lambda: engine.state.positions, lambda: engine._graphs, engine.flush)
out["info"] = {k: v for k, v in engine.info().items() if k != "capture_memory"}
out["mode"] = engine.mode
out["records"] = list(engine.force_field.resolution.enabled)
out["options"] = list(fixed.enabled_options(plan))
out["backend"], out["dtype"] = plan.engine.backend, plan.precision.dtype
out["plan"] = plan.describe()
out["hardware"] = [hardware.name, list(hardware.compute_capability), hardware.memory_bytes]
out["torch"] = torch.__version__
print(json.dumps(out))
"""


def minimal_run(backend, python):
    spec = ENGINE.spec("hybrid" if backend == "auto" else backend)
    spec["planner_backend"] = backend
    spec["precision"] = "float32" if backend == "auto" else "float64"
    return ENGINE.run_fresh(ENGINE.COMMON.format(spec=spec) + MINIMAL_CHILD, python=python)


def _summary(minimal, fixed, deviation, **extra):
    return {"minimal_vs_fixed_500": deviation,
            "ms_per_step": {"minimal": minimal["ms_per_step"], "fixed": fixed["ms_per_step"]},
            "edge_sizes": [minimal["info"]["edge_sizes_used"], fixed["info"]["edge_sizes_used"]],
            "rebuilds": [minimal["info"]["neighbour_rebuilds"], fixed["info"]["neighbour_rebuilds"]],
            "records": minimal["records"], "options": minimal["options"], "hardware": minimal["hardware"],
            "torch": minimal["torch"], "plan": minimal["plan"], **extra}


def _same_plan(minimal, fixed):
    assert minimal["records"] == fixed["records"]
    assert minimal["options"] == fixed["options"]
    assert (minimal["backend"], minimal["dtype"], minimal["mode"]) == (fixed["backend"], fixed["dtype"], fixed["mode"])
    assert minimal["info"]["edge_sizes_used"] == fixed["info"]["edge_sizes_used"]
    assert minimal["info"]["neighbour_rebuilds"] == fixed["info"]["neighbour_rebuilds"]


@pytest.mark.slow
@PLANNER.needs_mh1
def test_mh1_plain_cueq_float64_minimal_equals_fixed():
    minimal = minimal_run("cueq", sys.executable)
    fixed = PLANNER.planner_run("cueq", sys.executable)
    summary = _summary(minimal, fixed, ENGINE.deviation(minimal["positions"], fixed["positions"]))
    ENGINE.report("1b-T23-mh1-cueq-float64-minimal-vs-fixed", summary)
    _same_plan(minimal, fixed)
    assert summary["minimal_vs_fixed_500"] <= ENGINE.FLOAT64_LIMIT, summary["minimal_vs_fixed_500"]


@pytest.mark.slow
@ENGINE.needs_oeq
@PLANNER.needs_mh1
def test_mh1_hybrid_float32_minimal_equals_fixed():
    minimal = minimal_run("auto", PLANNER.HYBRID_PYTHON)
    first = PLANNER.planner_run("auto", PLANNER.HYBRID_PYTHON)
    second = PLANNER.planner_run("auto", PLANNER.HYBRID_PYTHON)
    scatter = ENGINE.deviation(second["positions"], first["positions"])
    summary = _summary(minimal, first, ENGINE.deviation(minimal["positions"], first["positions"]),
                       fixed_scatter_500=scatter)
    ENGINE.report("1b-T23-mh1-hybrid-float32-minimal-vs-fixed", summary)
    _same_plan(minimal, first)
    assert (minimal["backend"], minimal["dtype"]) == ("hybrid", "float32")
    limit = max(3 * scatter, 1e-6)
    assert summary["minimal_vs_fixed_500"] <= limit, (summary["minimal_vs_fixed_500"], limit)
