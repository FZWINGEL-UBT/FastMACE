"""Phase 3 contracts (plan task 3-T01; SPEC §7.2, §7.6, §8.1, §8.5, §9.2, §10.4, §10.6).

The new config fields and the ``MCConfig`` record, the public ``fm.mc`` names, the ``Simulation``
mutation API, the event and derived-state contract, the compiled tier, the energy-only and
``captured`` entry points of the graph engine, the MC driver and the restart version 3 schema.
Every stub raises ``NotImplementedError`` naming the task that fills it.
``docs/design/contracts-3.md`` is the page that explains it.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import subprocess
import sys
import typing

import pytest

from fastmace import config, errors


def _frozen(instance, name, value):
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, name, value)


def _parameters(function):
    return list(inspect.signature(function).parameters)


def _keyword_only(function):
    return [name for name, p in inspect.signature(function).parameters.items()
            if p.kind is inspect.Parameter.KEYWORD_ONLY]


def _raises_naming(task, call):
    with pytest.raises(NotImplementedError, match=task):
        call()


# --------------------------------------------------------------------------- config fields


def test_workload_max_displacement():  # rule 11's displacement skin (3-T05); T10 refuses a larger delta_A
    from fastmace.planner.workload import Workload

    assert Workload(max_temperature_K=300.0).max_displacement_A is None
    moves = dict(mc_moves=("displace",), min_mc_temperature_K=300.0)
    assert Workload(max_temperature_K=300.0, max_displacement_A=0.2, **moves).max_displacement_A == 0.2
    with pytest.raises(ValueError, match="max_displacement_A is required"):
        Workload(max_temperature_K=300.0, **moves)
    for value in (0.0, -0.1, float("inf"), True, "0.2"):
        with pytest.raises(ValueError):
            Workload(max_temperature_K=300.0, max_displacement_A=value, **moves)


def test_neighbour_config_admits_auto_skin_and_headroom():  # rule 11 (3-T05)
    neighbours = config.NeighbourConfig()
    assert neighbours.skin == "auto" and neighbours.headroom == "auto"
    assert config.NeighbourConfig(skin=0.4, headroom="fixed").skin == 0.4
    assert config.DEFAULT_SKIN_A == 0.3
    for kwargs in ({"skin": "big"}, {"skin": 0.0}, {"headroom": "on"}):
        with pytest.raises(ValueError):
            config.NeighbourConfig(**kwargs)



def test_engine_config_recapture_fields():  # §8.1 re-capture policy (3-T04)
    engine = config.EngineConfig()
    assert engine.recapture_budget == "auto"
    assert engine.graph_variant_cap == 64
    assert config.EngineConfig(recapture_budget=0).recapture_budget == 0
    assert config.EngineConfig(graph_variant_cap=1).graph_variant_cap == 1
    for kwargs in ({"recapture_budget": -1}, {"recapture_budget": 2.5}, {"recapture_budget": True},
                   {"recapture_budget": "on"}, {"graph_variant_cap": 0}, {"graph_variant_cap": 65},
                   {"graph_variant_cap": 8.0}, {"graph_variant_cap": 65, "dynamo_cache_size_limit": 128}):
        with pytest.raises(ValueError):
            config.EngineConfig(**kwargs)
    with pytest.raises(ValueError):  # the cap never exceeds dynamo's cache limit
        config.EngineConfig(graph_variant_cap=32, dynamo_cache_size_limit=16)


def test_compiled_mode_is_a_mode():  # §8.2: "compiled" admitted for MD (planner: 3-T06)
    assert "compiled" in config.MODES


def test_precision_config_mc_energy():  # §10.4 energy precision (3-T07, 3-T11)
    precision = config.PrecisionConfig()
    assert precision.mc_energy == "auto"
    assert typing.get_args(typing.get_type_hints(config.PrecisionConfig)["mc_energy"]) == (
        "auto", "float32", "float64")
    assert config.MC_ENERGY_PRECISIONS == ("auto", "float32", "float64")
    from fastmace.planner.workload import Workload

    assert typing.get_args(typing.get_type_hints(Workload)["mc_energy_precision"]) == config.MC_ENERGY_PRECISIONS
    with pytest.raises(ValueError):
        config.PrecisionConfig(mc_energy="float16")


def test_mc_config_defaults_and_validation():  # §10.4 driver options (3-T09)
    mc = config.MCConfig()
    assert mc.max_attempts_per_block == 10_000
    assert mc.max_move_sets == 8
    _frozen(mc, "max_move_sets", 2)
    for kwargs in ({"max_attempts_per_block": 0}, {"max_attempts_per_block": 1.5}, {"max_move_sets": 0},
                   {"max_move_sets": True}):
        with pytest.raises(ValueError):
            config.MCConfig(**kwargs)


def test_mc_stream_seeds_derive_from_the_seed():  # §8.5 random numbers per outcome
    assert config.MC_STREAMS == ("proposal", "acceptance", "velocity")
    seeds = config.mc_stream_seeds(1)
    assert tuple(seeds) == config.MC_STREAMS
    assert seeds == config.mc_stream_seeds(1)
    assert len(set(seeds.values())) == 3
    assert seeds != config.mc_stream_seeds(2)
    assert all(isinstance(value, int) and 0 <= value < 2**63 for value in seeds.values())
    # Independent of the process (no hash randomisation): a fixed value.
    assert config.mc_stream_seeds(0) == config.mc_stream_seeds(0)
    for bad in (-1, 1.0, True, "1"):
        with pytest.raises(ValueError):
            config.mc_stream_seeds(bad)


def test_mc_stream_seeds_are_stable_across_processes():
    code = "from fastmace import config; print(sorted(config.mc_stream_seeds(7).items()))"
    outputs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
               for _ in range(2)}
    assert len(outputs) == 1
    assert outputs.pop().strip() == str(sorted(config.mc_stream_seeds(7).items()))


# --------------------------------------------------------------------------- errors


def test_charge_parity_refusal_is_a_plan_refusal():  # §10.4 "Charge and spin"
    assert issubclass(errors.ChargeParityRefused, errors.PlanRefused)
    assert errors.CHARGE_PARITY_CODE == "transmute-charge-parity"
    reason = errors.Reason("Na <-> Mg: dZ - dq = 1 is odd", code=errors.CHARGE_PARITY_CODE)
    refused = errors.ChargeParityRefused.from_reasons([reason])
    assert type(refused) is errors.ChargeParityRefused
    assert refused.codes == ("transmute-charge-parity",)
    assert isinstance(refused, errors.FastMACEError)


# --------------------------------------------------------------------------- fm.mc


def test_fm_mc_is_a_lazy_public_name():
    import fastmace as fm

    assert "mc" in fm.__all__
    assert fm.mc.Swap is fm.mc.moves.Swap
    for name in ("Swap", "Transmute", "Displace", "configurational_mu", "MCDriver"):
        assert hasattr(fm.mc, name)


def test_fm_mc_imports_no_torch():
    code = ("import sys, fastmace as fm; fm.mc.Swap; fm.mc.configurational_mu; import fastmace.events; "
            "print('torch' in sys.modules, 'mace' in sys.modules, 'ase' in sys.modules)")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.split() == ["False", "False", "False"]


def test_swap_record():
    from fastmace.mc import Swap

    assert _parameters(Swap) == ["pairs", "every", "attempts", "velocities"]
    swap = Swap(pairs=[("Na", "Nb")], every=100, attempts=50)
    assert swap.move == "swap"
    assert swap.pairs == (("Na", "Nb"),)
    assert swap.velocities == "exchange"
    _frozen(swap, "every", 1)
    assert hash(swap) == hash(Swap(pairs=(("Na", "Nb"),), every=100, attempts=50))
    assert Swap([(11, 41)], 10, 5, velocities="rescale").pairs == ((11, 41),)
    for kwargs in ({"pairs": []}, {"pairs": [("Na", "Na")]}, {"pairs": [("Na",)]}, {"pairs": "NaCl"},
                   {"every": 0}, {"attempts": 0}, {"every": 1.5}, {"velocities": "keep"}):
        arguments = {"pairs": [("Na", "Cl")], "every": 10, "attempts": 5, **kwargs}
        with pytest.raises(ValueError):
            Swap(**arguments)
    assert config.SWAP_VELOCITIES == ("exchange", "rescale", "redraw", "redraw_all")


def test_transmute_record():
    from fastmace.mc import Transmute

    assert _parameters(Transmute) == ["species", "every", "attempts", "velocities", "mu", "formal_charges"]
    move = Transmute(species=["O", "Cl"], every=100, attempts=10, mu={"Cl": -0.2}, formal_charges={"O": -2, "Cl": -1})
    assert move.move == "transmute"
    assert move.species == ("O", "Cl")
    assert move.velocities is None  # the driver's default: rescale, or redraw_all under fixcm (§10.4)
    assert move.mu == (("Cl", -0.2),)
    assert move.formal_charges == (("Cl", -1), ("O", -2))
    assert hash(move) == hash(Transmute(("O", "Cl"), 100, 10, mu={"Cl": -0.2}, formal_charges={"O": -2, "Cl": -1}))
    assert Transmute(["Na", "K"], 1, 1, velocities="redraw_all").velocities == "redraw_all"
    assert config.TRANSMUTE_VELOCITIES == ("rescale", "redraw", "redraw_all")
    for kwargs in ({"species": ["Na"]}, {"species": ["Na", "Na"]}, {"velocities": "exchange"},
                   {"mu": {"Nb": 0.1}}, {"mu": {"Na": float("nan")}}, {"formal_charges": {"Nb": 1}},
                   {"formal_charges": {"Na": 1.5}}, {"attempts": 0}):
        arguments = {"species": ["Na", "K"], "every": 10, "attempts": 5, **kwargs}
        with pytest.raises(ValueError):
            Transmute(**arguments)


def test_displace_record():
    from fastmace.mc import Displace

    assert _parameters(Displace) == ["delta_A", "every", "attempts", "adaptive"]
    move = Displace(delta_A=0.1, every=10, attempts=20)
    assert move.move == "displace"
    assert move.adaptive is False
    for kwargs in ({"delta_A": 0.0}, {"delta_A": -0.1}, {"delta_A": float("inf")}, {"adaptive": "yes"},
                   {"every": 0}):
        arguments = {"delta_A": 0.1, "every": 10, "attempts": 5, **kwargs}
        with pytest.raises(ValueError):
            Displace(**arguments)


def test_moves_name_the_workload_move_types():
    from fastmace.mc import MOVE_TYPES, Displace, Swap, Transmute
    from fastmace.planner.workload import MC_MOVES

    assert MOVE_TYPES == ("swap", "transmute", "displace") == tuple(MC_MOVES)
    assert (Swap.move, Transmute.move, Displace.move) == MOVE_TYPES


def test_configurational_mu():  # §10.4: Δμ_conf = Δμ_full + (3/2) k_B T ln(m_s / m_ref)
    from fastmace.mc import configurational_mu

    assert _parameters(configurational_mu) == ["mu_full", "temperature_K", "reference", "masses"]
    k_b = 8.617333262e-5
    result = configurational_mu({"B": 0.5}, 300.0, reference="A", masses={"A": 1.0, "B": 4.0})
    assert result == {"B": pytest.approx(0.5 + 1.5 * k_b * 300.0 * math.log(4.0))}
    # Equal masses: unchanged; the reference itself is 0 by definition.
    assert configurational_mu({"B": 0.25, "A": 0.0}, 1000.0, reference="A", masses={"A": 2.0, "B": 2.0}) == {
        "B": pytest.approx(0.25), "A": 0.0}
    with pytest.raises(ValueError):
        configurational_mu({"A": 0.1}, 300.0, reference="A", masses={"A": 1.0})  # reference must be 0
    with pytest.raises(ValueError):
        configurational_mu({"B": 0.1}, -1.0, reference="A", masses={"A": 1.0, "B": 1.0})
    with pytest.raises(ValueError):
        configurational_mu({"B": 0.1}, 300.0, reference="A", masses={"A": 1.0})  # no mass for B


def test_configurational_mu_uses_ase_masses_by_default():
    pytest.importorskip("ase")
    from ase.data import atomic_masses, atomic_numbers

    from fastmace.mc import configurational_mu

    k_b = 8.617333262e-5
    expected = 1.5 * k_b * 500.0 * math.log(atomic_masses[atomic_numbers["Cl"]] / atomic_masses[8])
    assert configurational_mu({"Cl": 0.0}, 500.0, reference="O") == {"Cl": pytest.approx(expected)}
    assert configurational_mu({17: 0.0}, 500.0, reference=8) == {17: pytest.approx(expected)}


def test_mc_driver_contract():  # filled by 3-T09
    from fastmace.mc import MCDriver, Swap

    assert _parameters(MCDriver) == ["moves", "seed", "config"]
    assert _keyword_only(MCDriver) == ["seed", "config"]
    for name in ("attach", "run_block", "state_dict", "load_state_dict", "due"):
        assert callable(getattr(MCDriver, name))
    assert _parameters(MCDriver.attach) == ["self", "engine", "evaluator"]
    assert _keyword_only(MCDriver.attach) == ["evaluator"]
    assert _parameters(MCDriver.run_block) == ["self", "engine", "step"]
    driver = MCDriver((Swap([("Na", "Cl")], 10, 5),), seed=1)
    assert driver.last_block is None and driver.offsets == dict.fromkeys(config.MC_STREAMS, 0)


def test_attempt_contract():  # filled by 3-T09; displacements stay 3-T10's
    from fastmace.mc import Displace, moves

    assert _parameters(moves.build_attempt) == ["move", "engine", "evaluator", "streams", "fixcm"]
    assert _keyword_only(moves.build_attempt) == ["fixcm"]
    _raises_naming("3-T10", lambda: moves.build_attempt(Displace(0.1, 10, 5), None, None, None))


# --------------------------------------------------------------------------- events


def test_event_kinds_and_lifecycle():  # §8.5 transaction contract
    from fastmace import events

    assert events.EVENT_KINDS == ("species", "masses", "charges", "friction", "barostat", "swap", "transmute",
                                  "displace")
    assert events.LIFECYCLE == ("quiesce", "snapshot", "apply", "refresh", "evaluate", "decide", "resume")
    assert events.OUTCOMES == ("accepted", "rejected", "impossible", "redo")
    assert events.DEPENDENCIES == ("species", "masses", "charges", "friction", "barostat", "positions", "cell",
                                   "momenta", "forces", "stress")
    assert events.RESUME_ORDER == (("species", "masses", "charges", "momenta"), ("forces", "stress"),
                                   ("integrator caches",))


def test_event_record():
    from fastmace.events import Event

    event = Event("species", atoms=[3, 1], values=(17, 11))
    assert event.atoms == (3, 1)
    assert event.values == (17, 11)
    assert Event("friction", values=0.01).atoms is None  # all atoms / no atoms
    _frozen(event, "kind", "masses")
    for kwargs in ({"kind": "insert"}, {"kind": "species", "atoms": [-1]}, {"kind": "species", "atoms": [1.5]},
                   {"kind": "species", "atoms": [1, 1]}):
        with pytest.raises(ValueError):
            Event(**kwargs)


def test_derived_state_registry_register():
    from fastmace.events import DerivedBuffer, DerivedStateRegistry

    assert _parameters(DerivedStateRegistry.register) == ["self", "name", "buffer", "refresh", "depends_on"]
    registry = DerivedStateRegistry()
    buffer = object()
    entry = registry.register("integrator.c3_unit", buffer, lambda: None, depends_on=("masses", "friction"))
    assert isinstance(entry, DerivedBuffer)
    assert entry.depends_on == ("masses", "friction")
    assert registry.names() == ("integrator.c3_unit",)
    assert "integrator.c3_unit" in registry and len(registry) == 1
    assert registry["integrator.c3_unit"].buffer is buffer
    assert registry.affected(("friction",)) == ("integrator.c3_unit",)
    assert registry.affected(("species",)) == ()
    with pytest.raises(ValueError):  # one entry per name
        registry.register("integrator.c3_unit", buffer, lambda: None, depends_on=("masses",))
    with pytest.raises(ValueError):
        registry.register("x", buffer, lambda: None, depends_on=("temperature",))  # targets are not events
    with pytest.raises(ValueError):
        registry.register("y", buffer, lambda: None, depends_on=())
    with pytest.raises(TypeError):
        registry.register("z", buffer, "not callable", depends_on=("masses",))
    # 3-T03: refresh and the CI registry check.
    assert _parameters(DerivedStateRegistry.refresh) == ["self", "changed"]
    assert _parameters(DerivedStateRegistry.check_against) == ["self", "fresh"]
    assert registry.refresh(("masses",)) == ("integrator.c3_unit",)
    assert registry.check_against(registry) == []


def test_transaction_signature():
    from fastmace.events import Transaction

    assert _parameters(Transaction) == ["engine", "events", "evaluate", "registry"]
    assert _keyword_only(Transaction) == ["evaluate", "registry"]
    for name in ("run", "quiesce", "snapshot", "apply", "refresh", "accept", "reject", "resume"):
        assert callable(getattr(Transaction, name))


def test_force_field_superset_contract():  # §8.5 species superset (3-T02)
    from fastmace.model.forces import MACEForceField
    from fastmace.rewrites import registry as reg

    assert "species" in _keyword_only(MACEForceField)
    assert _parameters(MACEForceField.check_species) == ["self", "numbers"]
    assert _parameters(MACEForceField.register_derived) == ["self", "registry", "state", "prefix"]
    assert _parameters(MACEForceField.use_sources) == ["self", "species", "charges"]
    assert _parameters(MACEForceField.set_total_charge) == ["self", "total_charge"]
    assert _parameters(MACEForceField.derived_buffers) == ["self"]
    assert _parameters(reg.structure_inputs) == ["batch", "species_columns"]
    assert reg.JOINT_FEATURES == ("total_charge", "total_spin")


# --------------------------------------------------------------------------- Simulation mutation API


@pytest.mark.parametrize("name", ["set_species", "set_masses", "set_charges", "set_friction"])
def test_simulation_mutation_api(name):
    from fastmace.simulation import Simulation

    method = getattr(Simulation, name)
    assert _parameters(method) == ["self", {"set_species": "species", "set_masses": "masses",
                                            "set_charges": "charges", "set_friction": "friction_per_fs"}[name],
                                   *(["total_charge"] if name == "set_charges" else [])]


def test_set_charges_takes_the_total():
    from fastmace.simulation import Simulation

    assert _keyword_only(Simulation.set_charges) == ["total_charge"]


def test_simulation_update_is_one_transaction():  # 3-T03 (conductor answer; T13 applies an observer's change)
    from fastmace.simulation import Simulation

    fields = ["species", "masses", "charges", "total_charge", "friction_per_fs", "positions", "momenta"]
    assert _parameters(Simulation.update) == ["self", *fields]
    assert _keyword_only(Simulation.update) == fields
    for name in fields:
        assert inspect.signature(Simulation.update).parameters[name].default is None


def test_committed_carries_the_composition():  # 3-T03
    from fastmace.simulation import Committed

    assert [field.name for field in dataclasses.fields(Committed)] == [
        "step", "positions", "momenta", "cell", "energy", "forces", "stress", "species", "masses", "charges"]


def test_graph_engine_state_version():  # 3-T03, for 3-T07's energy cache; not restart state
    pytest.importorskip("torch")
    from fastmace.engine.graph import GraphEngine

    stub = object.__new__(GraphEngine)
    stub.state = type("State", (), {"step": 4})()
    stub.transactions_committed = 2
    assert stub.state_version == (4, 2)


# --------------------------------------------------------------------------- engines


def test_compiled_engine():  # §8.2 (3-T06)
    pytest.importorskip("torch")
    from fastmace.engine import compiled
    from fastmace.engine.graph import GraphEngine

    assert compiled.MODE == "compiled"
    assert _parameters(compiled.CompiledEngine) == ["config", "force_field", "integrator", "state", "neighbours",
                                                   "constraints", "dynamic_edges"]
    assert issubclass(compiled.CompiledEngine, GraphEngine)


def test_mode_defaults_to_none():  # §9.3: the planner picks graph, or compiled by rules 4 and 9 (3-T06)
    import inspect

    import fastmace as fm
    from fastmace.planner import minimal
    from fastmace.simulation import PlannerOptions, Simulation

    assert PlannerOptions().mode is None
    assert inspect.signature(Simulation).parameters["mode"].default is None
    assert inspect.signature(fm.md).parameters["mode"].default is None
    assert inspect.signature(minimal.MinimalPlanner.plan).parameters["mode"].default is None
    assert minimal.RULE_9_HOST_CALLBACKS == 10


def test_graph_engine_energy_and_captured_names():  # §8.1 energy-only graph (3-T07), captured (3-T04)
    pytest.importorskip("torch")
    from fastmace.engine.graph import ENERGY_KEY, ENGINE_KEYS, GraphEngine

    assert ENERGY_KEY == "E"
    assert {"ACP", "B", "C", "E"} <= set(ENGINE_KEYS)
    assert _parameters(GraphEngine.energy) == ["self", "precision"]
    assert _keyword_only(GraphEngine.energy) == ["precision"]
    assert _parameters(GraphEngine.captured) == ["self", "key", "fn"]
    assert _parameters(GraphEngine.add_energy_field) == ["self", "field"]  # the float64 MC copy (3-T07)
    assert isinstance(GraphEngine.energy_fields, property)
    assert _parameters(GraphEngine.energy_stage) == ["self", "precision"]
    assert _keyword_only(GraphEngine.energy_stage) == ["precision"]


def test_mc_evaluator_names():  # §10.4 evaluation, baseline energy, energy precision (3-T07)
    pytest.importorskip("torch")
    from fastmace.mc import evaluator
    from fastmace.model.forces import MACEForceField

    assert evaluator.PRECISIONS == ("float32", "float64")
    assert _parameters(evaluator.MCEvaluator.__init__) == ["self", "engine", "precision", "model", "copy", "polar"]
    assert _keyword_only(evaluator.MCEvaluator.__init__) == ["model", "copy", "polar"]
    for name in ("evaluate", "stage", "baseline", "current", "accept", "reject", "tag", "info"):
        assert callable(getattr(evaluator.MCEvaluator, name))
    assert _parameters(evaluator.float64_copy) == ["force_field", "model", "neighbours", "polar", "check"]
    assert _parameters(evaluator.float64_model) == ["force_field", "plan", "backend"]
    assert _parameters(MACEForceField.energy_only) == ["self", "positions"]
    assert _parameters(MACEForceField.follow) == ["self", "leader"]


def test_graph_engine_overflow_and_recapture_names():  # §8.1 overflow before commit, re-capture policy (3-T04)
    pytest.importorskip("torch")
    from fastmace.engine import graph
    from fastmace.engine.graph import GraphEngine

    assert (graph.EDGE_OVERFLOW, graph.KVECTOR_OVERFLOW) == (1, 2)
    assert issubclass(graph.CapacityOverflow, RuntimeError)  # internal (SPEC §7.6), not in errors.py
    assert issubclass(graph.RecaptureBudgetWarning, UserWarning)
    assert _parameters(GraphEngine.invalidate_graphs) == ["self", "reason", "keys"]
    assert _parameters(GraphEngine.uncommitted) == ["self"]
    assert _parameters(GraphEngine.overflowed) == ["self"]
    assert _parameters(GraphEngine.flag_edge_overflow) == ["self", "count", "edges"]
    assert _parameters(GraphEngine.clear_overflow) == ["self"]
    assert _parameters(GraphEngine.redo_on_overflow) == ["self", "attempt", "keep", "max_redos"]
    assert _keyword_only(GraphEngine.redo_on_overflow) == ["keep", "max_redos"]
    assert isinstance(GraphEngine.recapture_warning, property)


# --------------------------------------------------------------------------- restart v3


def test_restart_version_3_schema():  # §10.6 (3-T08)
    pytest.importorskip("torch")
    from fastmace.io import restart

    assert restart.FORMAT_VERSION == 3  # 3-T08 switched the writer to NEXT_FORMAT_VERSION
    assert restart.NEXT_FORMAT_VERSION == 3
    assert restart.V3_KEYS == ("mc",)
    assert restart.COMPOSITION_KEYS == ("species", "masses", "charges", "physical")
    assert restart.MC_RECORD_KEYS == ("moves", "cycle_position", "counters", "streams", "delta_A")
    assert restart.MC_COUNTER_KEYS == ("attempted", "impossible", "accepted")
    assert restart.MC_STREAM_KEYS == ("seed", "offset")
    assert restart.mc_record({}) is None  # 3-T08: no move set attached


# --------------------------------------------------------------------------- protocol runs


def test_protocol_run_names():  # §10.5 (3-T14)
    from fastmace.protocols import quench, runner

    assert _parameters(quench.relax) == ["atoms", "calc", "fmax", "max_steps", "logfile", "optimizer"]
    assert _parameters(quench.select_frames) == ["frames", "min_K", "bin_K", "per_bin", "peak_K"]
    assert _parameters(quench.picks) == ["resolved", "frame_every", "settings"]
    assert _parameters(quench.quench_run) == ["directory", "calc", "resolved", "frame_every", "timestep_fs",
                                              "settings", "log"]
    assert _keyword_only(runner.start) == ["relax", "quench", "steps", "source", "log"]
    assert _parameters(runner.resume) == ["directory", "steps", "log"]
    assert runner.RELAX_KEYS == ("fmax", "optimizer", "max_steps") and runner.PROTOCOL_RECORD == "protocol.json"
    assert "swap_anneal_NaNbOCl4" in runner.protocol_names()
