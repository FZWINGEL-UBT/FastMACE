"""Mutation events, the derived-state registry and the transaction contract (plan task 3-T03; SPEC §8.5, §8.1, §12.2).

On the CPU in float64, bit for bit:

- the CI registry check: after an accepted event the engine's derived registry, every other
  tensor and host constant its force field and integrator hold, and the neighbour list (as the set
  of compacted edges) equal a fresh construction; a buffer left out of the registry fails it;
- the first MD step after an accepted mass or friction event equals a fresh construction from the
  committed state (Lennard-Jones on the skin list in the checked, pipelined and speculative modes;
  BAOAB hydrostatic and full cell, so the cell force ``force_eps`` is the refreshed one);
- a rejected event, placed right after a canonical point, restores the state bit for bit, random
  numbers included, and the run continues as one without the event;
- a friction event drops the captured graphs (they bake ``c1``/``c2`` in) and, with the default
  barostat τ, moves ``P_tau`` and the barostat mass with it; an explicit barostat τ stays;
- a transaction writes only registered tensors;
- the ``Simulation`` mutation API (``update`` and the four setters) on tiny MACE models: one
  transaction with one force evaluation, a canonical point at the end, ``Committed``'s composition
  and the engine's state version; species events need 3-T02's superset checks;
- a ramp changes the target every step without an event.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest
import torch
from test_canonical_restart import (
    LJ,
    MODES,
    NEIGHBOURS,
    argon,
    assert_same_registered_state,
    assert_same_values,
    heat,
    registered,
)
from test_engine import ReplayEngine

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace import events, integrators
from fastmace.config import EngineConfig, IntegratorConfig
from fastmace.engine import constraints
from fastmace.engine.graph import GraphEngine
from fastmace.engine.registry import unregistered_mutations
from fastmace.events import DerivedStateRegistry, Event, Transaction
from fastmace.integrators.baoab import LangevinBAOAB
from fastmace.integrators.langevin import Langevin
from fastmace.planner import minimal
from fastmace.state import SimulationState

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

CASES = [("checked", False, "hydrostatic"), ("pipelined", False, "hydrostatic"), ("speculative", False, "hydrostatic"),
         ("pipelined", True, "hydrostatic"), ("checked", True, "full")]


def build(atoms=None, *, mode="pipelined", npt=False, cell_mode="hydrostatic", friction=0.02, barostat_tau_fs=None,
          seed=7):
    """``test_canonical_restart.lj_engine`` with the friction and barostat τ as parameters."""
    atoms = argon() if atoms is None else atoms
    options = MODES[mode]
    field = LJ(atoms, NEIGHBOURS, compute_stress=npt)
    state = SimulationState.from_atoms(atoms, "cpu", seed, compute_stress=npt)
    if npt:
        config = IntegratorConfig(ensemble="npt", timestep_fs=2.0, friction=friction, noise_batch=7, pressure_GPa=0.5,
                                  barostat_mass_factor=0.01, cell_mode=cell_mode, barostat_tau_fs=barostat_tau_fs)
        integrator = LangevinBAOAB(state, config)
    else:
        config = IntegratorConfig(timestep_fs=2.0, friction=friction, fixcm=True, noise_batch=7)
        integrator = Langevin(state, config)
    cls = ReplayEngine if options["replay"] else GraphEngine
    engine = cls(options["engine"], field, integrator, state, neighbours=NEIGHBOURS, capture=options["replay"],
                 constraints=constraints.ProcessRecord())
    if options["replay"]:
        engine.captures = []
    return engine


def committed_atoms(engine, masses=None):
    """The structure of the committed state: argon at the engine's positions and cell, with ``masses``."""
    atoms = argon()
    atoms.set_cell(engine.state.cell.numpy())
    atoms.set_positions(engine.state.positions.numpy())
    atoms.set_masses(engine.state.masses[:, 0].numpy() if masses is None else masses)
    return atoms


def take_dynamics(fresh, engine):
    """Give ``fresh`` the dynamical state of ``engine``: velocities (no round trip through momenta), the
    barostat momentum, the noise stream with its cursor, the step and the target; then its starting cell force."""
    state, source = fresh.state, engine.state
    state.velocities.copy_(source.velocities)
    if source.p_eps is not None:
        state.p_eps.copy_(source.p_eps)
    state.rng.generator = copy.deepcopy(source.rng.generator)
    state.rng.cursor = source.rng.cursor
    state.rng.batch.copy_(source.rng.batch)
    state.step = source.step
    fresh.integrator.set_temperature(engine.integrator.temperature_K)
    fresh.integrator.start(state)


def heavier(engine):
    masses = engine.state.masses[:, 0].numpy().copy()
    masses[::3] *= 1.7
    masses[1::5] *= 0.6
    return masses


# --------------------------------------------------------------------------- the registry


def test_refresh_runs_the_affected_entries_in_registration_order():
    registry = DerivedStateRegistry()
    calls = []
    registry.register("a", torch.zeros(1), lambda: calls.append("a"), depends_on=("masses",))
    registry.register("b", torch.zeros(1), lambda: calls.append("b"), depends_on=("friction",))
    registry.register("c", torch.zeros(1), lambda: calls.append("c"), depends_on=("masses", "forces"))
    assert registry.refresh(("masses",)) == ("a", "c")
    assert calls == ["a", "c"]
    assert registry.refresh(()) == ()


def test_check_against_names_every_difference():
    def registry(values, names=("x", "y")):
        out = DerivedStateRegistry()
        for name, value in zip(names, values):
            out.register(name, torch.tensor(value), lambda: None, depends_on=("masses",))
        return out

    assert registry([1.0, 2.0]).check_against(registry([1.0, 2.0])) == []
    assert registry([1.0, 2.0]).check_against(registry([1.0, 2.5])) == ["y"]
    assert registry([1.0, 2.0]).check_against(registry([1.0, 2.0], names=("x", "z"))) == ["y", "z"]
    assert registry([1.0, -0.0]).check_against(registry([1.0, 0.0])) == ["y"]  # bit for bit
    assert registry([1.0]).check_against(build(mode="checked")) != []  # an engine: its derived registry


@pytest.mark.parametrize("npt", [False, True])
def test_the_integrator_buffers_are_registered(npt):
    engine = build(mode="checked", npt=npt)
    names = set(engine.derived.names())
    expected = ({"integrator.inv_sqrt_masses", "integrator.force_eps"} if npt
                else {"integrator.c3_unit", "integrator.c4_unit", "integrator.c5_unit"})
    assert expected <= names
    for name in expected:  # in the runtime-state registry too, by identity (restart "derived")
        tensor = engine.derived[name].buffer
        assert any(other is tensor for other in engine.registry.tensors().values()), name


# --------------------------------------------------------------------------- accepted events


@pytest.mark.parametrize(("mode", "npt", "cell_mode"), CASES)
def test_first_step_after_an_accepted_mass_event_equals_a_fresh_construction(mode, npt, cell_mode):
    engine = build(mode=mode, npt=npt, cell_mode=cell_mode)
    heat(engine, 9)
    momenta = engine.state.momenta.clone()
    masses = heavier(engine)
    assert Transaction(engine, [Event("masses", values=masses)]).run() == "accepted"
    np.testing.assert_array_equal(engine.state.masses[:, 0].numpy(), masses)
    torch.testing.assert_close(engine.state.momenta, momenta, rtol=1e-15, atol=0)  # momenta kept (ASE)

    fresh = build(committed_atoms(engine, masses), mode=mode, npt=npt, cell_mode=cell_mode)
    take_dynamics(fresh, engine)
    assert events.check_against(engine, fresh) == []
    for name in ("forces", "energy", *(("stress", "force_eps") if npt else ())):
        torch.testing.assert_close(getattr(engine.state, name), getattr(fresh.state, name), rtol=0, atol=0, msg=name)
    heat(engine, 1, start=9)
    heat(fresh, 1, start=9)
    assert_same_values(engine.sync(), fresh.sync())


@pytest.mark.parametrize(("mode", "npt", "cell_mode"), CASES)
def test_first_step_after_an_accepted_friction_event_equals_a_fresh_construction(mode, npt, cell_mode):
    engine = build(mode=mode, npt=npt, cell_mode=cell_mode)
    heat(engine, 9)
    if engine.capture:
        assert engine._graphs
    assert Transaction(engine, [Event("friction", values=0.035)]).run() == "accepted"
    assert engine.integrator.config.friction == 0.035
    if engine.capture:
        assert not engine._graphs  # c1, c2 (and BAOAB's gamma) are baked into the captured stages
        captured = len(engine.captures)

    fresh = build(committed_atoms(engine), mode=mode, npt=npt, cell_mode=cell_mode, friction=0.035)
    take_dynamics(fresh, engine)
    assert events.check_against(engine, fresh) == []
    heat(engine, 2, start=9)
    heat(fresh, 2, start=9)
    assert_same_values(engine.sync(), fresh.sync())
    if engine.capture:
        assert len(engine.captures) > captured  # a re-capture


def test_friction_moves_the_default_barostat_and_keeps_an_explicit_one():
    default = build(mode="checked", npt=True)
    explicit = build(mode="checked", npt=True, barostat_tau_fs=150.0)
    p_tau, mass = default.integrator.P_tau, default.integrator.barostat_mass
    for engine in (default, explicit):
        Transaction(engine, [Event("friction", values=0.05)]).run()
    assert default.integrator.P_tau == pytest.approx(20.0 / 0.05) and default.integrator.P_tau != p_tau
    assert default.integrator.barostat_mass != mass
    assert explicit.integrator.P_tau == build(mode="checked", npt=True, barostat_tau_fs=150.0).integrator.P_tau
    assert events.check_against(explicit, build(committed_atoms(explicit), mode="checked", npt=True, friction=0.05,
                                                barostat_tau_fs=150.0)) == []


def test_several_events_are_one_transaction():
    engine = build(mode="checked")
    heat(engine, 4)
    calls = []
    original = engine._force_stage

    def counting():
        calls.append(1)
        original()

    engine._force_stage = counting
    masses = heavier(engine)
    Transaction(engine, [Event("masses", values=masses), Event("friction", values=0.03)]).run()
    assert calls == [1]  # one force evaluation, at the canonical point
    fresh = build(committed_atoms(engine, masses), mode="checked", friction=0.03)
    take_dynamics(fresh, engine)
    assert events.check_against(engine, fresh) == []


@pytest.mark.parametrize("npt", [False, True])
def test_a_transaction_writes_only_registered_tensors(npt):
    engine = build(mode="checked", npt=npt)
    heat(engine, 3)
    masses = heavier(engine)
    assert unregistered_mutations(engine, lambda: Transaction(
        engine, [Event("masses", values=masses), Event("friction", values=0.03)]).run()) == []


def test_a_buffer_left_out_of_the_registry_fails_the_check(monkeypatch):
    """The negative control of the CI check: Langevin without its c3 registration."""
    original = Langevin.register_derived

    def without_c3(self, registry, state):
        original(self, registry, state)
        registry._entries.pop("integrator.c3_unit")

    monkeypatch.setattr(Langevin, "register_derived", without_c3)
    engine = build(mode="checked")
    masses = heavier(engine)
    Transaction(engine, [Event("masses", values=masses)]).run()
    fresh = build(committed_atoms(engine, masses), mode="checked")
    take_dynamics(fresh, engine)
    differences = events.check_against(engine, fresh)
    assert any("c3_unit" in name for name in differences), differences


def test_a_forgotten_host_constant_fails_the_check(monkeypatch):
    """Friction constants are host floats baked into graphs: the check compares them too."""
    engine = build(mode="checked", npt=True)
    monkeypatch.setattr(LangevinBAOAB, "_friction_constants", lambda self: None)  # forgets to recompute
    Transaction(engine, [Event("friction", values=0.05)]).run()
    monkeypatch.undo()
    fresh = build(committed_atoms(engine), mode="checked", npt=True, friction=0.05)
    take_dynamics(fresh, engine)
    differences = events.check_against(engine, fresh)
    assert "integrator.gamma" in differences and "integrator.ou_unit" in differences, differences


# --------------------------------------------------------------------------- rejected events


@pytest.mark.parametrize(("mode", "npt", "cell_mode"), CASES)
@pytest.mark.parametrize("kind", ["masses", "friction", "both"])
def test_a_rejected_event_restores_the_state_exactly(mode, npt, cell_mode, kind):
    engine = build(mode=mode, npt=npt, cell_mode=cell_mode)
    twin = build(mode=mode, npt=npt, cell_mode=cell_mode)
    for one in (engine, twin):
        heat(one, 9)
        one.canonicalise()  # the event right after a canonical point
    before = registered(engine)
    cursor = engine.state.rng.cursor
    masses = heavier(engine)
    changes = {"masses": [Event("masses", values=masses)], "friction": [Event("friction", values=0.035)]}
    changes["both"] = changes["masses"] + changes["friction"]
    seen = {}

    def evaluate(eng):
        # The hook sees the applied and refreshed state, and may use random numbers.
        seen["masses"] = eng.state.masses[:, 0].clone()
        seen["friction"] = eng.integrator.config.friction
        eng.integrator.prepare_step(eng.state)
        return "rejected"

    assert Transaction(engine, changes[kind], evaluate=evaluate).run() == "rejected"
    if kind != "friction":
        np.testing.assert_array_equal(seen["masses"].numpy(), masses)
    if kind != "masses":
        assert seen["friction"] == 0.035
    assert_same_registered_state(registered(engine), before)
    assert engine.state.rng.cursor == cursor
    assert engine.integrator.config.friction == twin.integrator.config.friction
    assert events.check_against(engine, twin) == []
    assert engine.transactions_committed == 0
    heat(engine, 3, start=9)
    heat(twin, 3, start=9)
    assert_same_values(engine.sync(), twin.sync())


def test_a_redo_restores_and_reports_it():
    engine = build(mode="checked")
    heat(engine, 4)
    before = registered(engine)
    outcome = Transaction(engine, [Event("masses", values=heavier(engine))], evaluate=lambda eng: "redo").run()
    assert outcome == "redo"
    assert_same_registered_state(registered(engine), before)
    with pytest.raises(ValueError, match="outcome"):
        Transaction(engine, [Event("masses", values=heavier(engine))], evaluate=lambda eng: "maybe").run()
    assert_same_registered_state(registered(engine), before)


@pytest.mark.parametrize("values", [np.ones(5), np.zeros(108), np.full(108, np.nan), -np.ones(108)])
def test_invalid_masses_raise_before_anything_is_written(values):
    engine = build(mode="checked")
    heat(engine, 2)
    before = registered(engine)
    with pytest.raises(ValueError, match="masses"):
        Transaction(engine, [Event("masses", values=values)]).run()
    assert_same_registered_state(registered(engine), before)


@pytest.mark.parametrize("values", [0.0, -1.0, float("nan"), "fast"])
def test_invalid_friction_raises(values):
    engine = build(mode="checked")
    with pytest.raises((ValueError, TypeError), match="friction"):
        Transaction(engine, [Event("friction", values=values)]).run()


def test_kinds_of_later_tasks_are_refused():
    engine = build(mode="checked")
    for kind, task in (("swap", "3-T09"), ("transmute", "3-T09"), ("displace", "3-T10"), ("barostat", "barostat")):
        with pytest.raises(NotImplementedError, match=task):
            Transaction(engine, [Event(kind, values=0.0)]).run()


def test_the_state_version_counts_committed_transactions():
    engine = build(mode="checked")
    assert engine.state_version == (0, 0)
    heat(engine, 2)
    Transaction(engine, [Event("masses", values=heavier(engine))]).run()
    assert engine.state_version == (2, 1)
    Transaction(engine, [Event("friction", values=0.03)], evaluate=lambda eng: "rejected").run()
    assert engine.state_version == (2, 1)


def test_species_events_need_the_force_fields_superset_check():
    """LJ has no species: a species event is refused rather than leaving the forces unchanged (3-T02)."""
    engine = build(mode="checked")
    numbers = np.full(len(engine.state.species), 18)
    with pytest.raises(NotImplementedError, match="check_species"):
        Transaction(engine, [Event("species", values=numbers)]).run()


# --------------------------------------------------------------------------- Simulation (tiny MACE models)


NA_CL = (11, 17)
LANGEVIN = integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01)
BAOAB = integrators.LangevinBAOAB(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01, pressure_GPa=0.5,
                                  cell_mode="full")


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
    record = constraints.ProcessRecord()
    planner = minimal.MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ={},
                                     process=record)
    monkeypatch.setattr(minimal, "default_planner", lambda force_field: planner)
    return planner


def simulation(loaded, atoms=None, *, integrator=LANGEVIN, seed=1):
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    atoms = nacl(repeat=1, rattle=0.05, seed=3) if atoms is None else atoms
    return Simulation(atoms, ForceField(loaded, atoms, precision="float64"), integrator=integrator, seed=seed)


def fresh_simulation(loaded, sim, *, integrator=LANGEVIN, masses=None, numbers=None):
    """A plain construction from ``sim``'s committed state, with its velocities and noise stream."""
    committed = sim.committed()
    atoms = nacl(repeat=1)
    if numbers is not None:
        atoms.set_atomic_numbers(numbers)
    atoms.set_cell(committed.cell)
    atoms.set_positions(committed.positions)
    atoms.set_masses(committed.masses if masses is None else masses)
    fresh = simulation(loaded, atoms, integrator=integrator)
    take_dynamics(fresh._engine, sim._engine)
    return fresh


def _has_superset_check():
    from fastmace.model.forces import MACEForceField

    return hasattr(MACEForceField, "check_species")


@pytest.mark.parametrize("integrator", [LANGEVIN, BAOAB], ids=["langevin", "baoab-full"])
def test_set_masses_equals_a_fresh_simulation(loaded, integrator):
    sim = simulation(loaded, integrator=integrator)
    sim.run(steps=3)
    momenta = sim.committed().momenta
    masses = sim.committed().masses * np.where(np.arange(8) % 2 == 0, 1.5, 0.8)
    sim.set_masses(masses)
    committed = sim.committed()
    np.testing.assert_array_equal(committed.masses, masses)
    np.testing.assert_allclose(committed.momenta, momenta, rtol=1e-14, atol=0)
    assert committed.step == 3
    fresh = fresh_simulation(loaded, sim, integrator=integrator, masses=masses)
    assert events.check_against(sim._engine, fresh._engine) == []
    sim.run(steps=1)
    fresh._engine.step()
    for name in ("positions", "momenta", "forces", "energy"):
        np.testing.assert_array_equal(getattr(sim.committed(), name), getattr(fresh.committed(), name), err_msg=name)


def test_set_friction_updates_the_record_and_equals_a_fresh_simulation(loaded):
    sim = simulation(loaded)
    sim.run(steps=3)
    sim.set_friction(0.02)
    assert sim.integrator.friction_per_fs == 0.02
    assert sim._engine.integrator.config.friction == sim.integrator.to_config(seed=sim.seed).friction
    faster = dataclasses.replace(LANGEVIN, friction_per_fs=0.02)
    fresh = fresh_simulation(loaded, sim, integrator=faster)
    assert events.check_against(sim._engine, fresh._engine) == []
    sim.run(steps=2)
    for _ in range(2):
        fresh._engine.step()
    np.testing.assert_array_equal(sim.committed().positions, fresh.committed().positions)
    with pytest.raises(ValueError, match="friction"):
        sim.set_friction(0.0)


def test_update_is_one_transaction_with_one_force_evaluation(loaded):
    sim = simulation(loaded)
    sim.run(steps=2)
    engine = sim._engine
    calls = []
    original = engine._force_stage

    def counting():
        calls.append(1)
        original()

    engine._force_stage = counting
    committed = sim.committed()
    masses = committed.masses * 1.25
    momenta = np.full((8, 3), 0.02)
    positions = committed.positions + 0.01
    sim.update(masses=masses, momenta=momenta, positions=positions, friction_per_fs=0.015)
    assert calls == [1]
    after = sim.committed()
    np.testing.assert_array_equal(after.masses, masses)
    np.testing.assert_array_equal(after.positions, positions)
    np.testing.assert_allclose(after.momenta, momenta, rtol=1e-14, atol=0)  # the momenta of the new masses
    assert engine.state_version == (2, 1)
    assert sim.integrator.friction_per_fs == 0.015
    with pytest.raises(ValueError):  # validated before anything is written
        sim.update(masses=masses, positions=np.zeros((3, 3)))
    np.testing.assert_array_equal(sim.committed().positions, positions)
    assert engine.state_version == (2, 1)
    sim.update()  # nothing: no transaction
    assert engine.state_version == (2, 1)


def test_committed_carries_species_masses_and_charges(loaded):
    sim = simulation(loaded)
    committed = sim.committed()
    np.testing.assert_array_equal(committed.species, nacl(repeat=1).numbers)
    np.testing.assert_array_equal(committed.masses, nacl(repeat=1).get_masses())
    assert committed.charges is None
    assert not committed.species.flags.writeable and not committed.masses.flags.writeable


def test_charges_need_a_model_that_takes_them(loaded):
    sim = simulation(loaded)
    with pytest.raises(ValueError, match="charges"):
        sim.set_charges(np.zeros(8))
    with pytest.raises(ValueError, match="total charge"):
        sim.set_charges(None, total_charge=1.0)
    assert sim._engine.state_version == (0, 0)


@pytest.mark.skipif(_has_superset_check(), reason="3-T02 merged: the species tests below apply")
def test_set_species_is_refused_until_the_superset_lands(loaded):
    sim = simulation(loaded)
    with pytest.raises(NotImplementedError, match="check_species"):
        sim.set_species(nacl(repeat=1).numbers[::-1])


@pytest.mark.skipif(not _has_superset_check(), reason="needs 3-T02's MACEForceField.check_species")
@pytest.mark.parametrize("outcome", ["accepted", "rejected"])
def test_species_events_equal_a_fresh_construction(loaded, outcome):
    sim = simulation(loaded)
    sim.run(steps=3)
    numbers = nacl(repeat=1).numbers[::-1].copy()  # Na <-> Cl
    engine = sim._engine
    if outcome == "accepted":
        sim.set_species(numbers)
        committed = sim.committed()
        np.testing.assert_array_equal(committed.species, numbers)
        fresh = fresh_simulation(loaded, sim, numbers=numbers)
        assert events.check_against(engine, fresh._engine) == []
        sim.run(steps=1)
        fresh._engine.step()
        np.testing.assert_array_equal(sim.committed().forces, fresh.committed().forces)
    else:
        engine.canonicalise()
        before = registered(engine)
        Transaction(engine, [Event("species", values=numbers)], evaluate=lambda eng: "rejected").run()
        assert_same_registered_state(registered(engine), before)
    with pytest.raises(Exception, match="superset|SpeciesNotInSuperset"):
        sim.set_species(np.full(8, 8))


def test_set_species_validates_its_input(loaded):
    sim = simulation(loaded)
    with pytest.raises(ValueError, match="species"):
        sim.set_species([11, 17])
    with pytest.raises(ValueError, match="species"):
        sim.set_species(["Na"] * 7 + ["Xx"])


# --------------------------------------------------------------------------- ramps are not events


def test_a_ramp_changes_the_target_every_step_without_an_event(loaded, monkeypatch):
    from fastmace.protocols import ramp

    def refuse(self):
        raise AssertionError("a ramp step quiesced the pipeline")

    monkeypatch.setattr(Transaction, "quiesce", refuse)
    sim = simulation(loaded)
    integrator = sim._engine.integrator
    targets = []
    original = type(integrator).set_temperature

    def spy(self, temperature_K):
        targets.append(temperature_K)
        original(self, temperature_K)

    monkeypatch.setattr(type(integrator), "set_temperature", spy)
    sim.run(steps=4, schedule=ramp(300.0, 200.0, ps=0.004))
    assert len(set(targets)) >= 4
    assert sim._engine.transactions_committed == 0
