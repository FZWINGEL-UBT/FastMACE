"""The planner inputs of ``fm.Simulation`` (plan task 2-T01; SPEC §7.6, §9.3, §9.4, §12.3).

``mode``, ``allow_reference``, ``fallback``, ``gate``, ``runner``, ``engine`` and ``neighbours`` reach
the planner, are recorded in ``simulation.json``, and ``Simulation.resume`` re-plans with them. The
defaults leave the plan and the trajectory as they were. Tiny MH-1-class model, NaCl, CPU float64.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace import integrators
from fastmace.config import EngineConfig, FallbackConfig, GateConfig, NeighbourConfig, RunnerConfig
from fastmace.engine import constraints
from fastmace.errors import PlanRefused
from fastmace.planner import minimal

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

NA_CL = (11, 17)
LANGEVIN = integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01)
CPU = EngineConfig(device="cpu", compile_mode="none")


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    pytest.importorskip("cuequivariance_torch")
    from fastmace.model import loading

    model = tiny_mace_model(list(NA_CL), **FAMILIES["mh1"])
    path = tmp_path_factory.mktemp("models") / "tiny-mh1-11-17.model"
    torch.save(model, path)
    loading._LICENCES_SHOWN.add(str(path))
    return loading.load_pristine(str(path), trusted=True)


@pytest.fixture(autouse=True)
def cpu_planner(monkeypatch):
    """The default planner: the CPU without compilation, with a fresh process record per test."""
    record = constraints.ProcessRecord()
    planner = minimal.MinimalPlanner(engine=CPU, environ={}, process=record)
    monkeypatch.setattr(minimal, "default_planner", lambda force_field: planner)
    return planner


def _simulation(loaded, **options):
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    ff = ForceField(loaded, atoms, precision="float64")
    return Simulation(atoms, ff, integrator=LANGEVIN, seed=1, **options)


def test_defaults_give_the_plan_and_trajectory_of_no_options(loaded):
    plain = _simulation(loaded)
    explicit = _simulation(loaded, mode="graph", allow_reference=False, fallback=FallbackConfig(),
                           gate=GateConfig(), runner=RunnerConfig(), engine=None, neighbours=None)
    for name in ("tier", "rewrites", "engine", "precision", "neighbours", "decisions", "runner"):
        assert getattr(explicit.plan, name) == getattr(plain.plan, name)
    plain.run(steps=3)
    explicit.run(steps=3)
    np.testing.assert_array_equal(explicit.committed().positions, plain.committed().positions)
    assert explicit.fallback == FallbackConfig() and explicit.gate_config == GateConfig()
    assert explicit.runner == RunnerConfig()


def test_engine_and_neighbours_reach_the_planner(loaded, cpu_planner):
    engine = dataclasses.replace(CPU, pipelined="off")
    sim = _simulation(loaded, engine=engine, neighbours=NeighbourConfig(edge_multiple=128))
    assert sim.plan.neighbours.edge_multiple == 128
    assert sim.plan.engine.pipelined == "off"
    assert cpu_planner.engine == CPU and cpu_planner.neighbours == NeighbourConfig()  # not changed in place


@pytest.mark.parametrize("mode", ["compiled"])
def test_modes_other_than_graph_reach_the_planner(loaded, mode):
    with pytest.raises(PlanRefused) as refused:  # the compiled MD tier: phase 3
        _simulation(loaded, mode=mode)
    assert f"mode-{mode}" in refused.value.codes


def test_eager_mode_reaches_the_planner(loaded):  # 2-T02
    assert _simulation(loaded, mode="eager").plan.tier == "eager"


def test_mode_reference_reaches_the_planner(loaded):
    assert _simulation(loaded, mode="reference").plan.tier == "reference"  # 2-T03


def test_allow_reference_reaches_the_planner(loaded):
    assert "allow_reference" in _simulation(loaded, allow_reference=True).plan.allowances  # 2-T03


def test_options_are_recorded_and_resume_replans_with_them(loaded, tmp_path):
    from fastmace.simulation import SIMULATION_NAME, Simulation

    engine = dataclasses.replace(CPU, pipelined="off")
    neighbours = NeighbourConfig(edge_multiple=128)
    fallback = FallbackConfig(fallback="refuse", diagnosis_budget=10)
    runner = RunnerConfig(environment={"A": "1"}, stall_timeout_s=60.0)
    sim = _simulation(loaded, engine=engine, neighbours=neighbours, fallback=fallback,
                      gate=GateConfig(reduced_cell="off"), runner=runner)
    sim.run(steps=2)
    sim.save_restart(tmp_path / "restart.npz")
    record = json.loads((tmp_path / SIMULATION_NAME).read_text(encoding="utf-8"))
    options = record["planner_options"]
    assert options["mode"] == "graph" and options["allow_reference"] is False
    assert options["fallback"]["fallback"] == "refuse" and options["gate"]["reduced_cell"] == "off"
    assert options["runner"]["environment"] == [["A", "1"]]
    assert options["engine"]["pipelined"] == "off" and options["neighbours"]["edge_multiple"] == 128

    resumed = Simulation.resume(tmp_path)  # same plan: no allow_plan_change needed
    assert resumed.plan.neighbours.edge_multiple == 128 and resumed.plan.engine.pipelined == "off"
    assert resumed.fallback == fallback and resumed.runner == runner
    assert resumed.gate_config == GateConfig(reduced_cell="off")
    sim.run(steps=2)
    resumed.run(steps=2)
    np.testing.assert_array_equal(resumed.committed().positions, sim.committed().positions)


def test_md_passes_the_options_to_the_simulation(loaded, tmp_path):
    import fastmace as fm

    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    run = fm.md(atoms, model=loaded, integrator=LANGEVIN, steps=2, seed=1, output=tmp_path / "md",
                neighbours=NeighbourConfig(edge_multiple=128), fallback=FallbackConfig(fallback="refuse"))
    assert run.plan.neighbours.edge_multiple == 128
    assert run.simulation.fallback == FallbackConfig(fallback="refuse")


def test_a_version_1_record_resumes_with_the_defaults(loaded, tmp_path):
    from fastmace.simulation import SIMULATION_NAME, Simulation

    sim = _simulation(loaded)
    sim.run(steps=2)
    sim.save_restart(tmp_path / "restart.npz")
    path = tmp_path / SIMULATION_NAME
    record = json.loads(path.read_text(encoding="utf-8"))
    del record["planner_options"]
    record["simulation_record_version"] = 1
    path.write_text(json.dumps(record), encoding="utf-8")
    resumed = Simulation.resume(tmp_path)
    assert resumed.plan.engine == sim.plan.engine and resumed.fallback == FallbackConfig()
