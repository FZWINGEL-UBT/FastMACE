"""The eager tier for MD (plan task 2-T02; SPEC §8.3, §9.3 rules 3-4 and the mode dimension, §9.4 step 2).

``mode="eager"`` plans the graph plan's backend, rewrite set, precision, neighbours and integrator,
without CUDA graphs and without compilation: tier ``"eager"``, ``capture`` off, ``compile_mode``
"none" (so no compile family is claimed, §8.7), no speculative graphs; ``pipelined`` keeps its
resolved value. Rule 4 (inexact padding) is lifted for the eager tier only, which uses exact edge
counts. PolarMACE on plain cuEq plans in eager mode (rule 3's "eager plain"). ``MinimalPlanner.build``
builds the ``GraphEngine`` uncaptured through ``FixedPlanner.build(capture=False)``. On the CPU
(tiny models, float64) eager steps equal graph steps bitwise over 50 steps, and
``Simulation(..., mode="eager", check="gate")`` passes the gate and records the eager plan.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import test_minimal_planner as tm
import torch

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace import integrators
from fastmace.config import EngineConfig, IntegratorConfig
from fastmace.engine import constraints
from fastmace.errors import PlanRefused
from fastmace.planner import minimal
from fastmace.planner.minimal import MinimalPlanner

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

EAGER_CAPTURE = "eager tier: no CUDA graphs"
SAME_PARTS = ("rewrites", "requested", "neighbours", "polar", "integrator", "precision", "superset", "workload",
              "software")


# --------------------------------------------------------------------------- plans (model traits, GPU hardware)


def test_eager_plan_is_the_graph_plan_without_graphs_or_compilation():
    graph = tm.plan_label("mh1-r2scan")
    eager = tm.plan_label("mh1-r2scan", mode="eager")
    assert (graph.tier, eager.tier) == ("graph", "eager")
    for part in SAME_PARTS:
        assert getattr(eager, part) == getattr(graph, part), part
    assert eager.engine.compile_mode == "none" and graph.engine.compile_mode == "default"
    assert eager.engine == dataclasses.replace(graph.engine, compile_mode="none", dynamic_edges=False)  # decision 59
    assert eager.engine.backend == "hybrid"
    assert eager.engine.pipelined == graph.engine.pipelined  # resolved value kept (conductor answer T02)
    mode = eager.decision("mode")
    assert mode[1] == "eager" and mode[2].code == "mode-eager-tier" and "§8.3" in mode[2]
    assert eager.decision("tier")[1] == "eager"
    assert eager.decision("capture")[1] == "off" and EAGER_CAPTURE in eager.decision("capture")[2]
    assert eager.decision("EngineConfig.compile_mode")[1] == "none"
    assert graph.decision("capture")[1] == "on"


def test_eager_plan_claims_no_compile_family():
    record = constraints.ProcessRecord()
    tm.plan_label("mh1-r2scan", mode="eager", planner=tm.minimal(process=record))
    assert record.family is None
    tm.plan_label("mh1-r2scan", planner=tm.minimal(process=record))
    assert record.family is not None


def test_eager_turns_speculative_graphs_off():
    planner = tm.minimal(engine=EngineConfig(speculative_graph=True, pipelined="off"))
    assert tm.plan_label("mh1-r2scan", planner=planner).engine.speculative_graph is True
    eager = tm.plan_label("mh1-r2scan", planner=planner, mode="eager")
    assert eager.engine.speculative_graph is False
    choice, reason = eager.decision("EngineConfig.speculative_graph")[1:]
    assert choice == "off" and "eager" in reason


def test_rule_4_is_lifted_for_the_eager_tier_only():
    record = dataclasses.replace(tm.tp.traits("mh1-r2scan"), padding_is_exact=False)

    def plan(mode):
        return tm.minimal().plan_traits(record, nacl(), IntegratorConfig(), tm.WORKLOAD, tm.GPU,
                                        precision="float32", mode=mode)

    with pytest.raises(PlanRefused) as caught:
        plan("graph")
    assert tm.codes(caught.value) == ["inexact-padding"]
    eager = plan("eager")
    assert eager.tier == "eager"
    choice, reason = eager.decision("padding")[1:]
    assert choice == "exact edge counts" and "rule 4" in reason


def test_polar_plain_cueq_plans_in_eager_mode():
    """Rule 3: PolarMACE on plain compiled cuEq gives zero forces; eager plain cuEq is allowed."""
    planner = tm.minimal(engine=EngineConfig(backend="cueq"))
    with pytest.raises(PlanRefused) as caught:
        tm.plan_label("polar1-s", planner=planner)
    assert "polar-plain-compiled-cueq" in tm.codes(caught.value)
    eager = tm.plan_label("polar1-s", planner=planner, mode="eager")
    assert (eager.tier, eager.engine.backend, eager.engine.compile_mode) == ("eager", "cueq", "none")
    assert eager.polar is not None


def test_eager_plan_round_trips_through_the_restart_record():
    from fastmace.planner.plan import Plan

    eager = tm.plan_label("mh1-r2scan", mode="eager")
    record = json.loads(json.dumps(eager.to_restart(edge_bucket=256, neighbour_capacity=1024)))
    back = Plan.from_restart(record)
    assert back.tier == "eager" and back.engine == eager.engine


# --------------------------------------------------------------------------- build and steps (CPU, tiny models)


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    pytest.importorskip("cuequivariance_torch")
    from fastmace.model import loading

    model = tiny_mace_model(list(tm.NA_CL), **FAMILIES["mh1"])
    path = tmp_path_factory.mktemp("models") / "tiny-mh1-11-17.model"
    torch.save(model, path)
    loading._LICENCES_SHOWN.add(str(path))
    return loading.load_pristine(str(path), trusted=True)


def _ff(loaded, atoms):
    from fastmace.model.forcefield import ForceField

    return ForceField(loaded, atoms, precision="float64")


def test_build_passes_capture_off_to_the_graph_engine(loaded, monkeypatch):
    from fastmace.engine import graph

    seen = []
    original = graph.GraphEngine.__init__

    def spy(self, *args, **kwargs):
        seen.append(kwargs.get("capture"))
        original(self, *args, **kwargs)

    monkeypatch.setattr(graph.GraphEngine, "__init__", spy)
    planner = tm.cpu_minimal()
    atoms = nacl(rattle=0.05, seed=1)
    for mode in ("graph", "eager"):
        plan = planner.plan(_ff(loaded, atoms), atoms, IntegratorConfig(temperature_K=300.0), tm.WORKLOAD, tm.CPU,
                            mode=mode)
        engine = planner.build(plan, _ff(loaded, atoms), atoms)
        assert engine.capture is False and engine.pad_edges is False
    assert seen == [None, False]  # graph: the engine's default (capture on CUDA); eager: never


def test_fixed_planner_build_refuses_other_tiers(loaded):
    from fastmace.planner import fixed

    plan = dataclasses.replace(tm.plan_label("mh1-r2scan"), tier="reference")
    with pytest.raises(ValueError, match="tier"):
        fixed.FixedPlanner(environ={}).build(plan, loaded, nacl())


@pytest.mark.parametrize("npt", [False, True], ids=["nvt", "npt"])
def test_eager_equals_graph_bitwise_on_the_cpu(loaded, npt):
    """50 steps, float64, the same seed: the eager engine's trajectory is the graph plan's, bit for bit."""
    atoms = nacl(rattle=0.05, seed=1)
    integrator = IntegratorConfig(ensemble="npt", pressure_GPa=0.5) if npt else IntegratorConfig()
    integrator = dataclasses.replace(integrator, temperature_K=300.0)
    engines = {}
    for mode in ("graph", "eager"):
        planner = tm.cpu_minimal()
        plan = planner.plan(_ff(loaded, atoms), atoms, integrator, tm.WORKLOAD, tm.CPU, mode=mode)
        assert plan.tier == mode
        engines[mode] = planner.build(plan, _ff(loaded, atoms), atoms)
    for engine in engines.values():
        engine.run(50)
        engine.flush()
    graph, eager = engines["graph"], engines["eager"]
    assert eager.info()["capture"] is False
    np.testing.assert_array_equal(eager.state.positions.numpy(), graph.state.positions.numpy())
    np.testing.assert_array_equal(eager.state.forces.numpy(), graph.state.forces.numpy())
    assert eager.potential_energy() == graph.potential_energy()
    if npt:
        np.testing.assert_array_equal(eager.state.cell.numpy(), graph.state.cell.numpy())


@pytest.fixture
def cpu_planner(monkeypatch):
    record = constraints.ProcessRecord()
    planner = MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ={}, process=record)
    monkeypatch.setattr(minimal, "default_planner", lambda force_field: planner)
    return planner


@pytest.fixture
def no_engines_yet():
    from fastmace.engine import base

    saved = (base._PRODUCTION_ENGINES.count, base._PRODUCTION_ENGINES.first)
    base._forget_production_engines()
    yield
    base._PRODUCTION_ENGINES.count, base._PRODUCTION_ENGINES.first = saved


def test_simulation_runs_gated_in_eager_mode(loaded, cpu_planner, no_engines_yet, tmp_path):
    from fastmace.io.output import Output
    from fastmace.simulation import Simulation

    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    langevin = integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01)
    sim = Simulation(atoms, _ff(loaded, atoms), integrator=langevin, seed=8, check="gate", mode="eager",
                     output=Output(tmp_path / "run", trajectory_every=None, restart_every=None))
    assert sim.plan.tier == "eager" and sim.plan.decision("mode")[1] == "eager"
    assert sim.gate.status == "passed"
    assert sim._engine.info()["capture"] is False
    text = sim.report.text()
    assert "eager, " in text and EAGER_CAPTURE in text
    sim.run(steps=3)
    sim.save_restart(tmp_path / "restart")
    sim.close()
    record = json.loads((tmp_path / "restart" / "simulation.json").read_text())
    assert record["planner_options"]["mode"] == "eager"
    resumed = Simulation.resume(tmp_path / "restart")
    assert resumed.plan.tier == "eager" and resumed._engine.info()["capture"] is False
    np.testing.assert_array_equal(resumed.committed().positions, sim.committed().positions)
