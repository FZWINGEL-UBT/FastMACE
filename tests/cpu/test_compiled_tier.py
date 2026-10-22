"""The compiled MD tier (plan task 3-T06; SPEC §8.2, §9.3 rules 3, 4 and 9 and the mode dimension, §9.4).

``mode="compiled"`` plans the graph plan's backend, rewrite set, precision, neighbours and integrator
with the planner's ``compile_mode``, without CUDA graphs: tier ``"compiled"``, ``capture`` off, no
speculative graphs, exact edge counts (``dynamic_edges`` on when something is compiled). Without a
mode the planner picks it for inexact padding (rule 4, which lifts the refusal) and for frequent host
callbacks (rule 9). ``MinimalPlanner.build`` builds a ``CompiledEngine``, the ``GraphEngine`` uncaptured
and unpadded. On the CPU (tiny models, float64) compiled steps equal graph steps bitwise over 50 steps
with ``compile_mode="none"`` on both sides; with Inductor on both sides (slow) they agree to float64
rounding. ``Simulation(..., mode="compiled", check="gate")`` passes the gate, records the plan, and
resumes compiled.
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
from fastmace.planner.workload import Workload

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

COMPILED_CAPTURE = "compiled tier: no CUDA graphs"
SAME_PARTS = ("rewrites", "requested", "neighbours", "polar", "integrator", "precision", "superset", "workload",
              "software")


# --------------------------------------------------------------------------- plans (model traits, GPU hardware)


def test_compiled_plan_is_the_graph_plan_without_graphs():
    graph = tm.plan_label("mh1-r2scan")
    compiled = tm.plan_label("mh1-r2scan", mode="compiled")
    assert (graph.tier, compiled.tier) == ("graph", "compiled")
    for part in SAME_PARTS:
        assert getattr(compiled, part) == getattr(graph, part), part
    assert compiled.engine.compile_mode == graph.engine.compile_mode == "default"
    assert compiled.engine == dataclasses.replace(graph.engine, dynamic_edges=True)
    assert compiled.engine.backend == "hybrid"
    mode = compiled.decision("mode")
    assert mode[1] == "compiled" and mode[2].code == "mode-compiled-tier" and "§8.2" in mode[2]
    assert compiled.decision("tier")[1] == "compiled"
    choice, reason = compiled.decision("capture")[1:]
    assert choice == "off" and COMPILED_CAPTURE in reason
    assert compiled.decision("EngineConfig.dynamic_edges")[1] == "on"
    assert graph.decision("capture")[1] == "on"


def test_compiled_plan_claims_the_compile_family():
    record = constraints.ProcessRecord()
    tm.plan_label("mh1-r2scan", mode="compiled", planner=tm.minimal(process=record))
    assert record.family is not None


def test_compiled_turns_speculative_graphs_off():
    planner = tm.minimal(engine=EngineConfig(speculative_graph=True, pipelined="off"))
    compiled = tm.plan_label("mh1-r2scan", planner=planner, mode="compiled")
    assert compiled.engine.speculative_graph is False
    choice, reason = compiled.decision("EngineConfig.speculative_graph")[1:]
    assert choice == "off" and "compiled" in reason


def test_an_uncompiled_planner_plans_static_edges():
    planner = tm.minimal(engine=EngineConfig(compile_mode="none"))
    compiled = tm.plan_label("mh1-r2scan", planner=planner, mode="compiled")
    assert compiled.tier == "compiled" and compiled.engine.dynamic_edges is False
    assert compiled.decision("EngineConfig.dynamic_edges")[1] == "off"


def test_rule_4_is_lifted_for_the_compiled_tier():
    record = dataclasses.replace(tm.tp.traits("mh1-r2scan"), padding_is_exact=False)

    def plan(mode):
        return tm.minimal().plan_traits(record, nacl(), IntegratorConfig(), tm.WORKLOAD, tm.GPU,
                                        precision="float32", mode=mode)

    for mode in (None, "compiled"):
        compiled = plan(mode)
        assert compiled.tier == "compiled"
        choice, reason = compiled.decision("padding")[1:]
        assert choice == "exact edge counts" and "rule 4" in reason


def test_rule_9_and_rule_4_name_their_reasons():
    frequent = Workload(max_temperature_K=300.0, host_callbacks_every=1)
    plan = tm.plan_label("mh1-r2scan", workload=frequent)
    assert plan.tier == "compiled" and plan.decision("mode")[2].rule == 9
    assert tm.plan_label("mh1-r2scan", workload=frequent, mode="compiled").decision("mode")[2].code == \
        "mode-compiled-tier"  # an explicit mode is the user's, not rule 9's


def test_polar_plain_cueq_stays_refused_on_the_compiled_tier():
    """Rule 3: PolarMACE on plain compiled cuEq gives zero forces; the compiled tier compiles."""
    planner = tm.minimal(engine=EngineConfig(backend="cueq"))
    with pytest.raises(PlanRefused) as caught:
        tm.plan_label("polar1-s", planner=planner, mode="compiled")
    assert "polar-plain-compiled-cueq" in tm.codes(caught.value)


def test_compiled_plan_round_trips_through_the_restart_record():
    from fastmace.planner.plan import Plan

    compiled = tm.plan_label("mh1-r2scan", mode="compiled")
    record = json.loads(json.dumps(compiled.to_restart(edge_bucket=256, neighbour_capacity=1024)))
    back = Plan.from_restart(record)
    assert back.tier == "compiled" and back.engine == compiled.engine


# --------------------------------------------------------------------------- build and steps (CPU, tiny models)


def _load(tmp_path_factory, name, **options):
    pytest.importorskip("cuequivariance_torch")
    from fastmace.model import loading

    model = tiny_mace_model(list(tm.NA_CL), **options)
    path = tmp_path_factory.mktemp("models") / name
    torch.save(model, path)
    loading._LICENCES_SHOWN.add(str(path))
    return loading.load_pristine(str(path), trusted=True)


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    return _load(tmp_path_factory, "tiny-mh1-11-17.model", **FAMILIES["mh1"])


@pytest.fixture(scope="module")
def inexact(tmp_path_factory):
    """MH-1's interaction class applying its own cutoff: edges beyond it get nonzero weights (rule 4)."""
    return _load(tmp_path_factory, "tiny-inexact-11-17.model", **dict(FAMILIES["mh1"], apply_cutoff=True))


def _ff(loaded, atoms):
    from fastmace.model.forcefield import ForceField

    return ForceField(loaded, atoms, precision="float64")


def _planner(compile_mode="none"):
    return MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode=compile_mode), environ={},
                          process=constraints.ProcessRecord())


def test_build_gives_a_compiled_engine_without_capture_or_padding(loaded):
    from fastmace.engine.compiled import CompiledEngine
    from fastmace.engine.graph import GraphEngine

    planner = tm.cpu_minimal()
    atoms = nacl(rattle=0.05, seed=1)
    plan = planner.plan(_ff(loaded, atoms), atoms, IntegratorConfig(temperature_K=300.0), tm.WORKLOAD, tm.CPU,
                        mode="compiled")
    engine = planner.build(plan, _ff(loaded, atoms), atoms)
    assert isinstance(engine, CompiledEngine) and isinstance(engine, GraphEngine)
    assert engine.capture is False and engine.pad_edges is False
    assert engine.info()["capture"] is False and engine.info()["tier"] == "compiled"
    graph = planner.build(planner.plan(_ff(loaded, atoms), atoms, IntegratorConfig(temperature_K=300.0),
                                       tm.WORKLOAD, tm.CPU, mode="graph"), _ff(loaded, atoms), atoms)
    assert not isinstance(graph, CompiledEngine)


def test_compiled_engine_refuses_captured_options(loaded):
    from fastmace.engine.compiled import CompiledEngine

    planner = tm.cpu_minimal()
    atoms = nacl(rattle=0.05, seed=1)
    plan = planner.plan(_ff(loaded, atoms), atoms, IntegratorConfig(temperature_K=300.0), tm.WORKLOAD, tm.CPU,
                        mode="compiled")
    engine = planner.build(plan, _ff(loaded, atoms), atoms)
    config = dataclasses.replace(engine.config, speculative_graph=True, pipelined="off")
    with pytest.raises(ValueError, match="speculative"):
        CompiledEngine(config, engine.force_field, engine.integrator, engine.state)


def test_an_inexact_padding_model_plans_and_runs_compiled(inexact):
    atoms = nacl(rattle=0.05, seed=1)
    ff = _ff(inexact, atoms)
    assert ff.traits.padding_is_exact is False
    planner = tm.cpu_minimal()
    with pytest.raises(PlanRefused) as caught:
        planner.plan(ff, atoms, IntegratorConfig(temperature_K=300.0), tm.WORKLOAD, tm.CPU, mode="graph")
    assert tm.codes(caught.value) == ["inexact-padding"] and 'mode="compiled"' in caught.value.reasons[0]
    plan = planner.plan(ff, atoms, IntegratorConfig(temperature_K=300.0), tm.WORKLOAD, tm.CPU)
    assert plan.tier == "compiled" and plan.decision("mode")[2].code == "mode-compiled-rule-4"
    engine = planner.build(plan, _ff(inexact, atoms), atoms)
    engine.run(5)
    engine.flush()
    assert engine.pad_edges is False and np.isfinite(engine.state.positions.numpy()).all()


def _trajectories(loaded, integrator, modes, planner_factory, steps=50):
    atoms = nacl(rattle=0.05, seed=1)
    engines = {}
    for mode in modes:
        planner = planner_factory()
        plan = planner.plan(_ff(loaded, atoms), atoms, integrator, tm.WORKLOAD, tm.CPU, mode=mode)
        assert plan.tier == mode
        engines[mode] = planner.build(plan, _ff(loaded, atoms), atoms)
    for engine in engines.values():
        engine.run(steps)
        engine.flush()
    return engines


def _integrator(npt):
    integrator = IntegratorConfig(ensemble="npt", pressure_GPa=0.5) if npt else IntegratorConfig()
    return dataclasses.replace(integrator, temperature_K=300.0)


@pytest.mark.parametrize("npt", [False, True], ids=["nvt", "npt"])
def test_compiled_equals_graph_bitwise_on_the_cpu(loaded, npt):
    """50 steps, float64, the same seed, ``compile_mode="none"`` on both sides: bit for bit."""
    engines = _trajectories(loaded, _integrator(npt), ("graph", "compiled"), _planner)
    graph, compiled = engines["graph"], engines["compiled"]
    assert compiled.info()["capture"] is False
    np.testing.assert_array_equal(compiled.state.positions.numpy(), graph.state.positions.numpy())
    np.testing.assert_array_equal(compiled.state.momenta.numpy(), graph.state.momenta.numpy())
    np.testing.assert_array_equal(compiled.state.forces.numpy(), graph.state.forces.numpy())
    assert compiled.potential_energy() == graph.potential_energy()
    if npt:
        np.testing.assert_array_equal(compiled.state.cell.numpy(), graph.state.cell.numpy())


@pytest.mark.slow
def test_compiled_equals_graph_to_rounding_with_inductor_on_the_cpu(loaded):
    """Inductor on both sides (3-T06 conductor answer 2): static edge shapes against dynamic ones.
    The differences are recorded in ``docs/validation/3-T06.md``; they must stay at float64 rounding."""
    pytest.importorskip("torch._inductor")
    torch._dynamo.reset()
    try:
        engines = _trajectories(loaded, _integrator(False), ("graph", "compiled"), lambda: _planner("default"))
    finally:
        torch._dynamo.reset()
    graph, compiled = engines["graph"], engines["compiled"]
    assert compiled.force_field.dynamic_edges and not graph.force_field.dynamic_edges
    positions = np.abs(compiled.state.positions.numpy() - graph.state.positions.numpy()).max()
    forces = np.abs(compiled.state.forces.numpy() - graph.state.forces.numpy()).max()
    energy = abs(compiled.potential_energy() - graph.potential_energy())
    print(f"3-T06 Inductor CPU float64, 50 steps: max |dx| {positions:.3e} A, max |dF| {forces:.3e} eV/A, "
          f"|dE| {energy:.3e} eV")
    assert positions <= 1e-10 and forces <= 1e-8 and energy <= 1e-9


@pytest.fixture
def cpu_planner(monkeypatch):
    planner = _planner()
    monkeypatch.setattr(minimal, "default_planner", lambda force_field: planner)
    return planner


@pytest.fixture
def no_engines_yet():
    from fastmace.engine import base

    saved = (base._PRODUCTION_ENGINES.count, base._PRODUCTION_ENGINES.first)
    base._forget_production_engines()
    yield
    base._PRODUCTION_ENGINES.count, base._PRODUCTION_ENGINES.first = saved


def test_simulation_runs_gated_in_compiled_mode(loaded, cpu_planner, no_engines_yet, tmp_path):
    from fastmace.io.output import Output
    from fastmace.simulation import Simulation

    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    langevin = integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01)
    sim = Simulation(atoms, _ff(loaded, atoms), integrator=langevin, seed=8, check="gate", mode="compiled",
                     output=Output(tmp_path / "run", trajectory_every=None, restart_every=None))
    assert sim.plan.tier == "compiled" and sim.plan.decision("mode")[1] == "compiled"
    assert sim.gate.status == "passed"
    assert sim._engine.info()["capture"] is False
    text = sim.report.text()
    assert "compiled, " in text and COMPILED_CAPTURE in text
    sim.run(steps=3)
    sim.save_restart(tmp_path / "restart")
    sim.close()
    record = json.loads((tmp_path / "restart" / "simulation.json").read_text())
    assert record["planner_options"]["mode"] == "compiled"
    resumed = Simulation.resume(tmp_path / "restart")
    assert resumed.plan.tier == "compiled" and resumed._engine.info()["capture"] is False
    np.testing.assert_array_equal(resumed.committed().positions, sim.committed().positions)
