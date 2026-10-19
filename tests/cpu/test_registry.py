"""The runtime-state registry and complete capture restore on the CPU (plan task 1b-T04, SPEC §8.1, §12.2).

Two checks, on eager engines (the ``_version`` counters of CPU tensors move on every in-place write;
captured graph replays do not move them, so the GPU tests compare state before and after instead):

- every tensor reachable from the engine whose version counter a stage moves (A, C, B, the pipelined
  ACP and its redo, the speculative ACB, a rebuild) is registered; an unregistered one is reported
  with its attribute path;
- the full registered state, sticky flags included, is bitwise identical before and after the
  warm-up runs of every stage.

The Lennard-Jones force field and the engine helper come from ``test_engine.py``; the tiny MACE
models of every family are built through the fixed planner, as ``main.py`` builds them.
"""

import dataclasses

import numpy as np
import pytest
import torch
from test_engine import build

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace.config import EngineConfig, IntegratorConfig
from fastmace.engine import base, constraints, registry
from fastmace.engine.graph import GraphEngine
from fastmace.engine.registry import Registry, RegistryError, policy_for, unregistered_mutations

CPU_ENGINE = EngineConfig(pipelined="on", device="cpu")


def stage_keys(engine):
    edges = engine.force_field.num_edges if hasattr(engine.force_field, "num_edges") else int(
        engine.force_field.edge_count)
    keys = ["AC", "C", "ACP", ("B", edges)]
    if not engine.variable_cell:
        keys.append(("ACB", edges, 0))
    return keys


def exercise(engine):
    """Every stage and host path once: checked and pipelined steps, a forced redo, a rebuild, ACB."""
    engine.integrator.set_temperature(300.0)
    engine.run(2)
    engine._pipe_edges = 1  # the next validation sees count > edges: restore mid and redo B
    engine.run(2)
    engine.flush()
    engine._rebuild()
    engine._run("C")
    for key in stage_keys(engine):
        engine.integrator.prepare_step(engine.state)
        engine._stage(key)()
    engine._step_checked()
    engine.flush()


# --------------------------------------------------------------------------- the policy table


def test_policy_table_records_are_valid_and_unique():
    patterns = [entry.name for entry in registry.POLICIES]
    assert len(patterns) == len(set(patterns))
    assert all(isinstance(entry, registry.RegisteredBuffer) for entry in registry.POLICIES)


@pytest.mark.parametrize(("name", "restart", "capture", "sticky"), [
    ("state.positions", "save", True, False),
    ("state.forces", "save", True, False),
    ("state.rng.batch", "save", True, False),
    ("state.p_eps", "save", True, False),
    ("integrator.x_old", "derived", True, False),
    ("force_field.neighbours.edge_index", "reset", True, False),
    ("force_field.volume", "derived", True, False),
    ("force_field.positions", "derived", True, False),
    ("force_field.node_e0", "derived", True, False),
    ("force_field.polar.overflow", "reset", True, True),
    ("force_field.polar.coefficients", "reset", True, False),
    ("engine.flags", "reset", True, True),
    ("engine.mid.0", "reset", True, False),
    ("engine.flags_host", "derived", False, False),
])
def test_policies_follow_the_contract(name, restart, capture, sticky):
    entry = policy_for(name)
    assert entry.name == name
    assert (entry.restart, entry.capture, entry.sticky) == (restart, capture, sticky)
    if restart == "reset":
        assert entry.recompute


def test_a_buffer_without_a_policy_is_refused():
    with pytest.raises(RegistryError, match="force_field.mystery"):
        policy_for("force_field.mystery")
    with pytest.raises(RegistryError, match="engine.mystery"):
        Registry(lambda: {"engine.mystery": torch.zeros(1)}).entries()


# --------------------------------------------------------------------------- the registry of an engine


def test_registry_lists_each_tensor_once_with_its_policy():
    engine = build(npt=True, engine=CPU_ENGINE)
    entries = engine.registry.entries()
    names = set(entries)
    assert {"state.positions", "state.velocities", "state.cell", "state.forces", "state.energy", "state.stress",
            "state.species", "state.masses", "state.rng.batch", "state.p_eps", "state.force_eps",
            "integrator.xi", "integrator.zeta", "integrator.sqrt_kT", "force_field.neighbours.edge_index",
            "force_field.neighbours.count", "force_field.neighbours.cell", "force_field.neighbours.build_cell",
            "engine.rebuild_needed", "engine.flags", "engine.pipe_flags", "engine.spec_flags", "engine.mid.0",
            "engine.mid.1", "engine.flags_host", "engine.pipe_flags_host", "engine.spec_flags_host"} <= names
    # state.positions is the integrator's "positions" too: one tensor, the state's name
    assert {"integrator.positions", "integrator.p_eps"}.isdisjoint(names)
    pointers = [tensor.data_ptr() for _, tensor in entries.values()]
    assert len(pointers) == len(set(pointers))
    assert all(policy.name == name for name, (policy, _) in entries.items())


def test_runtime_state_is_gone():
    assert not hasattr(GraphEngine, "runtime_state")


@pytest.mark.parametrize("npt", [False, True])
def test_stages_mutate_only_registered_tensors(npt):
    engine = build(npt=npt, engine=CPU_ENGINE)
    assert unregistered_mutations(engine, lambda: exercise(engine)) == []


def test_an_unregistered_mutated_tensor_is_reported_with_its_path():
    engine = build(engine=CPU_ENGINE)
    engine.integrator.probe = torch.zeros(3, dtype=torch.float64)
    second_half = engine.integrator.second_half

    def leaky(state):
        second_half(state)
        engine.integrator.probe.add_(1.0)

    engine.integrator.second_half = leaky
    assert unregistered_mutations(engine, lambda: exercise(engine)) == ["engine.integrator.probe"]


# --------------------------------------------------------------------------- warm-ups leave no trace


def set_sticky_flags(engine):
    """Flags as a host would find them mid-run: a warm-up must neither set nor clear them."""
    for name, (policy, tensor) in engine.registry.entries().items():
        if policy.sticky:
            tensor.fill_(1)


def assert_registered_state_unchanged(engine, before, key):
    after = engine.registry.entries()
    assert set(after) == set(before), key
    for name, (policy, tensor) in after.items():
        if policy.capture:
            assert torch.equal(tensor, before[name]), (key, name)


@pytest.mark.parametrize("npt", [False, True])
def test_warmups_restore_the_full_registered_state(npt):
    engine = build(npt=npt, engine=CPU_ENGINE)
    engine.integrator.set_temperature(100.0)
    engine.run(3)
    engine.flush()
    engine.integrator.prepare_step(engine.state)
    set_sticky_flags(engine)
    before = {name: tensor.clone() for name, (_, tensor) in engine.registry.entries().items()}
    for key in stage_keys(engine):
        engine._warm_up(key)
        assert_registered_state_unchanged(engine, before, key)


def test_snapshot_and_restore_put_every_captured_buffer_back():
    engine = build(npt=True, engine=CPU_ENGINE)
    engine.integrator.set_temperature(100.0)
    engine.run(2)
    engine.flush()
    saved = engine.registry.snapshot()
    before = {name: tensor.clone() for name, (_, tensor) in engine.registry.entries().items()}
    engine.run(3)
    engine.flush()
    engine.registry.restore(saved)
    assert_registered_state_unchanged(engine, before, "restore")


def test_restore_refuses_a_buffer_that_moved():
    holder = {"state.positions": torch.zeros(4, 3)}
    reg = Registry(lambda: dict(holder))
    saved = reg.snapshot()
    holder["state.positions"] = torch.zeros(8, 3)
    with pytest.raises(RegistryError, match="state.positions"):
        reg.restore(saved)


# --------------------------------------------------------------------------- production-engine record


def test_building_a_graph_engine_is_recorded_process_wide():
    base._forget_production_engines()
    assert not base.production_engine_built()
    build(engine=CPU_ENGINE)
    assert base.production_engine_built()
    base._forget_production_engines()
    assert not base.production_engine_built()


# --------------------------------------------------------------------------- tiny MACE models


@pytest.fixture
def mace_engine(tmp_path, monkeypatch):
    pytest.importorskip("cuequivariance_torch")
    from fastmace.model import loading
    from fastmace.planner.fixed import FixedPlanner
    from fastmace.planner.plan import Hardware

    monkeypatch.setattr(loading, "_LICENCES_SHOWN", set())

    def make(family, npt=False):
        model = tiny_mace_model([11, 17], **FAMILIES[family])
        path = tmp_path / f"tiny-{family}.model"
        torch.save(model, path)
        planner = FixedPlanner(engine=EngineConfig(device="cpu", backend="cueq", compile_mode="none"), environ={})
        loaded = planner.load(str(path), trusted=True)
        atoms = nacl(rattle=0.05, seed=1)
        integrator = IntegratorConfig(ensemble="npt", pressure_GPa=0.5) if npt else IntegratorConfig()
        plan = planner.plan(loaded, atoms, integrator, Hardware("cpu", (0, 0), 0))
        plan = dataclasses.replace(plan, engine=dataclasses.replace(plan.engine, pipelined="on"))
        return planner.build(plan, loaded, atoms, temperature_K=300.0, constraints=constraints.ProcessRecord())

    return make


MACE_CASES = [("mh1", False), ("mh1", True), ("residual", False), ("density", False), ("omol", False),
              ("off", False), ("polar", False), ("polar", True)]


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(("family", "npt"), MACE_CASES)
def test_mace_stages_mutate_only_registered_tensors(mace_engine, family, npt):
    if family == "polar":
        from _support.tiny_models import polar_available

        if not polar_available():
            pytest.skip("PolarMACE needs MACE's polar module")
    engine = mace_engine(family, npt)
    assert {"force_field.positions", "force_field.node_e0"} <= set(engine.registry.tensors())
    assert unregistered_mutations(engine, lambda: exercise(engine)) == []


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(("family", "npt"), [("mh1", False), ("mh1", True), ("polar", True)])
def test_mace_warmups_restore_the_full_registered_state(mace_engine, family, npt):
    if family == "polar":
        from _support.tiny_models import polar_available

        if not polar_available():
            pytest.skip("PolarMACE needs MACE's polar module")
    engine = mace_engine(family, npt)
    engine.run(3)
    engine.flush()
    engine.integrator.prepare_step(engine.state)
    set_sticky_flags(engine)
    if family == "polar":
        assert "force_field.polar.overflow" in engine.registry.entries()
    before = {name: tensor.clone() for name, (_, tensor) in engine.registry.entries().items()}
    for key in stage_keys(engine):
        engine._warm_up(key)
        assert_registered_state_unchanged(engine, before, key)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_trajectory_unchanged_by_warmups(mace_engine):
    """Warm-ups between steps leave the trajectory bitwise as it was (CPU float64, after a throwaway step)."""
    plain, warmed = mace_engine("mh1"), mace_engine("mh1")
    for engine in (plain, warmed):
        engine.run(1)
        engine.flush()
    for _ in range(4):
        for key in stage_keys(warmed):
            warmed._warm_up(key)
        for engine in (plain, warmed):
            engine.run(1)
            engine.flush()
    np.testing.assert_array_equal(plain.state.positions.numpy(), warmed.state.positions.numpy())
    np.testing.assert_array_equal(plain.state.forces.numpy(), warmed.state.forces.numpy())
