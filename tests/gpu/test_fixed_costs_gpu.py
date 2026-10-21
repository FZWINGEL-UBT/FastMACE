"""OPT2-T04's fixed-cost options on the GPU: exact, bitwise in the reproducible mode.

``EngineConfig.overlap_rebuild`` (an early neighbour rebuild on a side stream beside the force graph in
flight) and ``EngineConfig.step_graph`` (one ("ACPB", edges) CUDA graph per pipelined step). MH-1
(``mh1-r2scan``) on NaCl 3x3x3 in ``reproducible=True`` (plain cuEq, unfused convolution, the
deterministic process), pipelined, Langevin at 1000 K and 2 fs so that early rebuilds occur, each case
in a fresh process: off, each option alone, and both. Positions and momenta after the run are bitwise
equal to the default's; the overlapped runs did rebuild beside the force graph, and the step-graph runs
replayed ACPB graphs (plus B graphs only for redos).
"""

import importlib.util
from pathlib import Path

import pytest


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_fixed_costs_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")

pytestmark = [
    pytest.mark.gpu, pytest.mark.slow,
    pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1"),
]

DRIVER = """
import json
import fastmace  # noqa: F401  (before torch)
import fastmace as fm
import numpy as np
import torch
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from _support.structures import nacl
from fastmace.config import EngineConfig
SPEC = {spec!r}
atoms = nacl(repeat=3)
MaxwellBoltzmannDistribution(atoms, temperature_K=1000.0, rng=np.random.default_rng(11))
integrator = fm.integrators.Langevin(timestep_fs=2.0, temperature_K=1000.0, friction_per_fs=0.02 * units.fs)
sim = fm.Simulation(atoms, fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float64"), integrator=integrator,
                    seed=20261021, workload=fm.Workload(max_temperature_K=1000.0, reproducible=True),
                    engine=EngineConfig(pipelined="on", **SPEC["engine"]))
sim.run(steps=SPEC["steps"])
committed = sim.committed()
info = dict(sim._engine.info())
out = {{"step": committed.step, "positions": committed.positions.tobytes().hex(),
        "momenta": committed.momenta.tobytes().hex(), "energy": committed.energy,
        "pipelined": sim._engine.pipelined, "early_rebuilds": info["pipelined_early_rebuilds"],
        "rebuilds": info["neighbour_rebuilds"], "beside": info.get("rebuilds_beside"), "graphs": info["graphs"],
        "backend": sim.plan.engine.backend, "dtype": sim.plan.precision.dtype, "tier": sim.plan.tier,
        "gpu": torch.cuda.get_device_name()}}
sim.close()
print(json.dumps(out))
"""

CASES = {"default": {}, "overlap_rebuild": {"overlap_rebuild": True}, "step_graph": {"step_graph": True},
         "both": {"overlap_rebuild": True, "step_graph": True}}


def test_the_fixed_cost_options_are_bitwise_equal_to_the_default():
    runs = {name: ENGINE.run_fresh(DRIVER.format(spec={"engine": engine, "steps": 600}))
            for name, engine in CASES.items()}
    summary = {name: {k: v for k, v in run.items() if k not in ("positions", "momenta")} for name, run in runs.items()}
    ENGINE.report("opt2-T04-fixed-costs-mh1-nacl", summary)
    default = runs["default"]
    assert default["pipelined"] and default["tier"] == "graph" and default["early_rebuilds"] > 0, summary
    assert default["beside"] is None, summary
    for name in ("overlap_rebuild", "both"):
        assert runs[name]["beside"] == runs[name]["early_rebuilds"] > 0, summary
    for name in ("step_graph", "both"):
        graphs = runs[name]["graphs"]
        assert any("ACPB" in key for key in graphs) and "ACP" not in graphs, summary
    for name, run in runs.items():
        assert (run["step"], run["positions"], run["momenta"]) == (
            default["step"], default["positions"], default["momenta"]), (name, summary)
