"""A graph-tier MD start compiles once with ``EngineConfig.dynamic_edges`` (opt-4 OPT4-T05).

Without it the first compiled call specialises on the exact initial edge count, and capturing B at the
padded count makes dynamo compile the model again with a dynamic edge dimension (8-9 s warm, about 22 s
cold, locally). With it the edges are marked dynamic from the first call and the capture reuses that
graph; the edges stay padded and the graphs are still captured.

MH-1 (``mh1-r2scan``) hybrid float32, NaCl 3x3x3, NVT 300 K, 50 steps, one fresh process per side.
"""

import importlib.util
from pathlib import Path

import pytest


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_dynamic_edges_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")

CHILD = """
import fastmace as fm
import torch
from ase import units
from torch._dynamo.utils import counters

from fastmace.config import EngineConfig

counters.clear()
integrator = fm.integrators.Langevin(timestep_fs=2.0, temperature_K=SPEC["temperature"],
                                     friction_per_fs=0.02 * units.fs)
engine = EngineConfig(dynamic_edges=True) if SPEC["dynamic_edges"] else None
sim = fm.Simulation(structure(), fm.load(SPEC["label"]), integrator=integrator, seed=SPEC["seed"], engine=engine)
setup_graphs = counters["stats"]["unique_graphs"]
sim.run(steps=SPEC["steps"])
info, plan = sim._engine.info(), sim.plan
decisions = {entry[0]: entry[1] for entry in plan.decisions}
out = {"tier": plan.tier, "capture": info["capture"], "dynamic_edges": plan.engine.dynamic_edges,
       "decision": decisions.get("EngineConfig.dynamic_edges"), "setup_graphs": setup_graphs,
       "run_graphs": counters["stats"]["unique_graphs"], "captures": sim._engine.recapture_log,
       "energy": sim.committed().energy}
sim.close()
print(json.dumps(out, default=str))
"""


def _run(dynamic_edges):
    spec = {"temperature": 300.0, "seed": 20261024, "steps": 50, "label": "mh1-r2scan",
            "dynamic_edges": dynamic_edges}
    return ENGINE.run_fresh(ENGINE.COMMON.format(spec=spec) + CHILD, python=ENGINE.HYBRID_PYTHON)


@pytest.mark.slow
@pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1")
def test_dynamic_edges_md_start_compiles_once():
    if not ENGINE._has_oeq():
        pytest.skip("OpenEquivariance is not installed (hybrid environment needed)")
    static, dynamic = _run(False), _run(True)
    ENGINE.report("opt4-T05-dynamic-edges-md", {"default": static, "dynamic_edges": dynamic})
    for side in (static, dynamic):
        assert (side["tier"], side["capture"]) == ("graph", True), side
    assert static["dynamic_edges"] is False and static["decision"] is None
    assert dynamic["dynamic_edges"] is True and dynamic["decision"] == "on"
    # The default compiles the model twice (the static first call, then the padded capture); the option once.
    assert dynamic["setup_graphs"] < static["setup_graphs"], (static, dynamic)
    assert dynamic["run_graphs"] == dynamic["setup_graphs"], dynamic  # nothing compiled during the steps
    assert abs(dynamic["energy"] - static["energy"]) / 216 < 1e-3, (static["energy"], dynamic["energy"])


REPRODUCIBLE = """
import json
import fastmace  # noqa: F401  (before torch)
import fastmace as fm
import numpy as np
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from _support.structures import nacl
from fastmace.config import EngineConfig
SPEC = {spec!r}
atoms = nacl(repeat=3)
MaxwellBoltzmannDistribution(atoms, temperature_K=1000.0, rng=np.random.default_rng(11))
integrator = fm.integrators.Langevin(timestep_fs=2.0, temperature_K=1000.0, friction_per_fs=0.02 * units.fs)
sim = fm.Simulation(atoms, fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float64"), integrator=integrator,
                    seed=20261024, workload=fm.Workload(max_temperature_K=1000.0, reproducible=True),
                    engine=EngineConfig(dynamic_edges=SPEC["dynamic_edges"]))
sim.run(steps=SPEC["steps"])
committed = sim.committed()
out = {{"step": committed.step, "positions": committed.positions.tobytes().hex(),
        "momenta": committed.momenta.tobytes().hex(), "energy": committed.energy, "tier": sim.plan.tier,
        "dynamic_edges": sim.plan.engine.dynamic_edges, "rebuilds": sim._engine.info()["neighbour_rebuilds"]}}
sim.close()
print(json.dumps(out))
"""


@pytest.mark.slow
@pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1")
def test_dynamic_edges_reproducible_trajectories():
    """``reproducible=True`` (plain cuEq float64): with the option, two runs are bitwise equal, and the report
    records whether they equal the default's (the steps replay the same compiled code; the first forces
    come from the dynamic graph instead of the static one)."""
    runs = {name: ENGINE.run_fresh(REPRODUCIBLE.format(spec={"dynamic_edges": dynamic, "steps": 300}))
            for name, dynamic in (("default", False), ("dynamic", True), ("dynamic_again", True))}
    summary = {name: {k: v for k, v in run.items() if k not in ("positions", "momenta")} for name, run in runs.items()}
    summary["dynamic_equals_default"] = all(runs["dynamic"][k] == runs["default"][k] for k in ("positions", "momenta"))
    ENGINE.report("opt4-T05-dynamic-edges-reproducible", summary)
    assert runs["default"]["tier"] == "graph" and runs["dynamic"]["dynamic_edges"] is True, summary
    assert runs["default"]["rebuilds"] > 0, summary  # the edge count changed during the run
    for key in ("step", "positions", "momenta"):
        assert runs["dynamic_again"][key] == runs["dynamic"][key], (key, summary)
