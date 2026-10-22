"""The move-energy gate and the MC energy precision (plan task 3-T11; SPEC §12.3 "move-energy
differences", §10.4 "Energy precision", "Accuracy prerequisite", §9.3, §9.4; decision 35).

- The sample moves: every swap pair of the superset (a pair with an absent species on a base with
  one atom transmuted when transmutations are declared, else "not sampleable"), a transmutation of
  each present species to every other superset species (absent ones included), and displacements
  of ``Workload.max_displacement_A``; drawn from the seed, the same in every process.
- The rule: every sample's ``|ΔE_prod − ΔE_ref| <= 0.01 k_B T_min``; a non-finite value fails; no
  ratio term.
- On the CPU (tiny model, eager stages): the gate passes in float64 and leaves the registered state
  unchanged; a species-dependent constant energy error in the float32 MC energies, invisible to
  forces, fails it: ``"auto"`` moves MC to float64, an explicit ``"float32"`` raises ``GateFailed``
  with the measured error; an absent superset species is gated; an unavailable reference makes
  ``"auto"`` choose float64; the record carries the new fields; ``check="skip"`` keeps the status
  ``skipped`` with the move-energy part in ``move_energy``.
- The planner resolves ``PrecisionConfig.mc_energy`` for MC plans; the fallback's MC energy
  precision dimension is ``"mc_energy"``.
"""

import dataclasses
import math

import numpy as np
import pytest
import torch

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace.config import EngineConfig, IntegratorConfig
from fastmace.engine import base, constraints
from fastmace.errors import GateFailed
from fastmace.planner import fallback, selfcheck
from fastmace.planner.workload import Workload

OXYGEN, NA, CL = 8, 11, 17
SUPERSET = (OXYGEN, NA, CL)
T_MIN = 250.0
DELTA = 0.1
NVT = IntegratorConfig(temperature_K=300.0)


def workload(moves=("swap", "transmute", "displace"), precision="auto", **extra):
    fields = dict(max_temperature_K=1000.0, min_mc_temperature_K=T_MIN, mc_moves=tuple(moves),
                  species_changes=any(move in ("swap", "transmute") for move in moves),
                  mc_energy_precision=precision)
    if "displace" in moves:
        fields["max_displacement_A"] = DELTA
    fields.update(extra)
    return Workload(**fields)


# --------------------------------------------------------------------------- the threshold and the rule


def test_threshold_is_a_hundredth_of_k_b_t_min():
    assert selfcheck.MOVE_FRACTION == 0.01
    assert selfcheck.move_threshold(T_MIN) == pytest.approx(0.01 * 8.617333262e-5 * T_MIN, rel=1e-12)


def rows(*errors):
    return [{"move": "transmute", "label": f"sample {i}", "production": 1.0 + e, "reference": 1.0, "error": abs(e)}
            for i, e in enumerate(errors)]


def test_move_rule_passes_at_the_threshold_and_fails_above():
    threshold = selfcheck.move_threshold(T_MIN)
    assert selfcheck.move_verdict(rows(0.0, threshold), threshold)["passed"]
    result = selfcheck.move_verdict(rows(0.0, 1.01 * threshold), threshold)
    assert not result["passed"]
    worst = result["moves"]["transmute"]
    assert worst["worst_eV"] == pytest.approx(1.01 * threshold)
    assert worst["worst_k_B_T_min"] == pytest.approx(1.01 * selfcheck.MOVE_FRACTION)
    assert worst["samples"] == 2 and not worst["passed"]


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_a_nonfinite_move_energy_fails_before_any_maximum(bad):
    threshold = selfcheck.move_threshold(T_MIN)
    case = rows(0.0, 0.0)
    case[0]["production"], case[0]["error"] = bad, bad
    result = selfcheck.move_verdict(case, threshold)
    assert not result["passed"] and result["nonfinite"] == ["sample 0"]


# --------------------------------------------------------------------------- the sample moves


def structure():
    return nacl(repeat=2, rattle=0.05, seed=1)  # 64 atoms, Na and Cl


def test_sample_moves_cover_every_pair_and_every_superset_species():
    atoms = structure()
    moves = selfcheck.sample_moves(atoms, SUPERSET, workload(), seed=3)
    assert moves.threshold_eV == selfcheck.move_threshold(T_MIN)
    assert moves.configurations[0] == selfcheck.MoveConfiguration()  # the base
    by_move = {}
    for sample in moves.samples:
        by_move.setdefault(sample.move, []).append(sample)
    # Swaps: each unordered pair of superset species, MOVE_SAMPLES["swap"] times.
    pairs = {tuple(sorted(sample.species)) for sample in by_move["swap"]}
    assert pairs == {(OXYGEN, NA), (OXYGEN, CL), (NA, CL)}
    assert len(by_move["swap"]) == 3 * selfcheck.MOVE_SAMPLES["swap"]
    # Transmutations: each present species to every other superset species, O (absent) included.
    targets = {sample.species for sample in by_move["transmute"]}
    assert targets == {(NA, OXYGEN), (NA, CL), (CL, OXYGEN), (CL, NA)}
    # Displacements of exactly max_displacement_A.
    assert len(by_move["displace"]) == selfcheck.MOVE_SAMPLES["displace"]
    for sample in by_move["displace"]:
        (atom, vector), = moves.configurations[sample.after].displacements
        assert np.linalg.norm(vector) == pytest.approx(DELTA, rel=1e-12)
        assert moves.configurations[sample.before] == selfcheck.MoveConfiguration()
    assert moves.not_sampleable == ()


def test_an_absent_species_swap_is_sampled_on_a_transmuted_base():
    atoms = structure()
    moves = selfcheck.sample_moves(atoms, SUPERSET, workload(), seed=3)
    for sample in moves.samples:
        if sample.move != "swap" or OXYGEN not in sample.species:
            continue
        before, after = moves.configurations[sample.before], moves.configurations[sample.after]
        assert OXYGEN in dict(before.species).values()  # one atom transmuted into the absent species first
        numbers = selfcheck.configuration_atoms(atoms, before).numbers
        swapped = selfcheck.configuration_atoms(atoms, after).numbers
        assert sorted(numbers) == sorted(swapped)  # a swap keeps the composition
        assert int((numbers != swapped).sum()) == 2


def test_an_absent_species_swap_without_transmutations_is_not_sampleable():
    moves = selfcheck.sample_moves(structure(), SUPERSET, workload(moves=("swap",)), seed=3)
    assert {tuple(sorted(sample.species)) for sample in moves.samples} == {(NA, CL)}
    assert len(moves.not_sampleable) == 2
    assert all("not sampleable" in text and "O" in text for text in moves.not_sampleable)


def test_sample_moves_come_from_the_seed():
    atoms = structure()
    first = selfcheck.sample_moves(atoms, SUPERSET, workload(), seed=3)
    assert selfcheck.sample_moves(atoms, SUPERSET, workload(), seed=3) == first
    assert selfcheck.sample_moves(atoms, SUPERSET, workload(), seed=4) != first


def test_no_mc_moves_no_samples():
    assert selfcheck.move_references(None, None, structure(), Workload(max_temperature_K=300.0), seed=0,
                                     check="skip") is None


# --------------------------------------------------------------------------- end to end on the CPU


@pytest.fixture
def no_engines_yet():
    saved = (base._PRODUCTION_ENGINES.count, base._PRODUCTION_ENGINES.first)
    base._forget_production_engines()
    yield
    base._PRODUCTION_ENGINES.count, base._PRODUCTION_ENGINES.first = saved


@pytest.fixture
def pristine(tmp_path, monkeypatch):
    pytest.importorskip("mace")
    pytest.importorskip("cuequivariance_torch")
    from fastmace.model import loading

    monkeypatch.setattr(loading, "_LICENCES_SHOWN", set())
    torch.serialization.add_safe_globals([slice])
    model = tiny_mace_model(list(SUPERSET), **FAMILIES["mh1"])
    path = tmp_path / "tiny-mh1-ona-cl.model"
    torch.save(model, path)
    return loading.load_pristine(str(path), trusted=True)


def cpu_planner():
    from fastmace.planner.minimal import MinimalPlanner

    return MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ={},
                          process=constraints.ProcessRecord())


def built(pristine, precision="float64", load=None, **options):
    from fastmace.model.forcefield import ForceField
    from fastmace.planner.plan import Hardware

    atoms = structure()
    load = workload(**options) if load is None else load
    ff = ForceField(pristine, atoms, precision=precision, species=SUPERSET)
    planner = cpu_planner()
    plan = planner.plan(ff, atoms, NVT, load, Hardware("cpu", (0, 0), 0))
    moves = selfcheck.move_references(plan, ff, atoms, load, seed=3, check="skip")
    engine = planner.build(plan, ff, atoms)
    return engine, moves, plan, ff


def registered(engine):
    return {name: tensor.detach().clone() for name, tensor in engine.registry.tensors().items()}


def inject_sodium_error(field, per_atom_eV=1e-3):
    """A species-dependent constant energy error in ``field``'s MC energies: invisible to forces."""
    original = field.energy_only

    def shifted(positions):
        return original(positions) + per_atom_eV * (field.atomic_numbers == NA).sum().to(torch.float64)

    field.energy_only = shifted


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_move_gate_passes_in_float64_and_leaves_the_state_unchanged(pristine, no_engines_yet):
    engine, moves, plan, ff = built(pristine)
    assert moves.ran == "in-process" and moves.unavailable is None
    assert len(moves.energies) == len(moves.moves.configurations)
    before = registered(engine)
    record = selfcheck.move_gate(engine, moves, plan=plan, force_field=ff)
    after = registered(engine)
    assert before.keys() == after.keys()
    for name in before:
        assert torch.equal(before[name], after[name]), name
    part = record.move_energy
    assert part["status"] == "passed", part
    assert record.mc_energy_precision == "float64" == part["precision"]
    assert {"swap", "transmute", "displace"} <= set(part["moves"])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_record_has_the_new_fields(pristine, no_engines_yet):
    engine, moves, plan, ff = built(pristine)
    record = selfcheck.move_gate(engine, moves, plan=plan, force_field=ff)
    assert record.status == "skipped"  # check="skip": the MD gate stays opt-in (decision 35)
    metadata = record.to_metadata()
    part = metadata["move_energy"]
    assert metadata["mc_energy_precision"] == "float64"
    for key in ("status", "requested", "precision", "threshold_eV", "min_mc_temperature_K", "fraction",
                "superset", "samples", "not_sampleable", "moves", "tried", "references", "seconds"):
        assert key in part, key
    assert part["requested"] == "auto" and part["min_mc_temperature_K"] == T_MIN
    assert part["superset"] == ["O", "Na", "Cl"]
    assert part["tried"] == [{"precision": "float64", "status": "passed",
                              "worst_k_B_T_min": part["tried"][0]["worst_k_B_T_min"]}]
    for name, summary in part["moves"].items():
        assert summary["worst_eV"] <= part["threshold_eV"], name
        assert summary["samples"] > 0
    assert part["references"]["ran"] == "in-process"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_an_absent_superset_species_is_gated(pristine, no_engines_yet):
    engine, moves, plan, ff = built(pristine)
    assert OXYGEN not in set(engine.state.species.tolist())
    record = selfcheck.move_gate(engine, moves, plan=plan, force_field=ff)
    labels = [row["label"] for row in record.move_energy["rows"]]
    assert any("-> O" in label for label in labels)
    assert any("swap" in label and "O" in label for label in labels)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_species_constant_error_moves_auto_to_float64(pristine, no_engines_yet):
    engine, moves, plan, ff = built(pristine, precision="float32")
    assert plan.precision.mc_energy == "float32"  # pending the gate
    inject_sodium_error(engine.force_field)
    record = selfcheck.move_gate(engine, moves, plan=plan, force_field=ff)
    part = record.move_energy
    assert [entry["precision"] for entry in part["tried"]] == ["float32", "float64"]
    assert [entry["status"] for entry in part["tried"]] == ["failed", "passed"]
    assert part["status"] == "passed" and record.mc_energy_precision == "float64"
    # The error shows in transmutations (one Na fewer or more: 1 meV), not in swaps (composition kept).
    failed = part["tried"][0]
    assert failed["moves"]["transmute"]["worst_eV"] == pytest.approx(1e-3, abs=2e-5)  # plus float32 noise
    assert failed["moves"]["swap"]["passed"]
    revised = fallback.mc_energy_revision(plan, record)
    assert revised.precision.mc_energy == "float64" and revised.precision.dtype == "float32"
    assert revised.fallbacks[-1].dimension == "mc_energy" and revised.fallbacks[-1].outcome == "activated"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_an_explicit_float32_that_fails_raises_with_the_measured_error(pristine, no_engines_yet):
    engine, moves, plan, ff = built(pristine, precision="float32",
                                    load=workload(precision="float32"))
    assert plan.precision.mc_energy == "float32"
    inject_sodium_error(engine.force_field)
    with pytest.raises(GateFailed) as raised:
        selfcheck.move_gate(engine, moves, plan=plan, force_field=ff)
    part = raised.value.gate_record.move_energy
    assert part["status"] == "failed" and part["precision"] == "float32"
    assert [entry["precision"] for entry in part["tried"]] == ["float32"]
    assert "move-energy" in str(raised.value) and "k_B T_min" in raised.value.gate_record.reason


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_an_unavailable_reference_makes_auto_choose_float64(pristine, no_engines_yet, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("no reference here")

    monkeypatch.setattr(selfcheck, "evaluate_references", broken)
    engine, moves, plan, ff = built(pristine, precision="float32")
    assert moves.unavailable is not None
    record = selfcheck.move_gate(engine, moves, plan=plan, force_field=ff)
    part = record.move_energy
    assert part["status"] == "unavailable" and record.mc_energy_precision == "float64"
    assert "decision 35" in part["reason"]


# --------------------------------------------------------------------------- the planner and the fallback


def test_mc_plans_resolve_the_mc_energy_precision(pristine):
    from fastmace.model.forcefield import ForceField
    from fastmace.planner.plan import Hardware

    atoms = structure()
    planner = cpu_planner()
    hardware = Hardware("cpu", (0, 0), 0)

    def plan(precision, load):
        return planner.plan(ForceField(pristine, atoms, precision=precision, species=SUPERSET), atoms, NVT, load,
                            hardware)

    def decision(p):
        return next(reason for item, _, reason in p.decisions if item == "PrecisionConfig.mc_energy")

    md = plan("float32", Workload(max_temperature_K=300.0))
    assert md.precision.mc_energy == "float32" and "no MC" in decision(md)
    auto = plan("float32", workload())
    assert auto.precision.mc_energy == "float32"
    assert decision(auto).code == "mc-energy-gate-pending"
    explicit = plan("float32", workload(precision="float64"))
    assert explicit.precision.mc_energy == "float64" and decision(explicit).code == "mc-energy-requested"
    from fastmace.errors import PlanRefused

    with pytest.raises(PlanRefused, match="below the float64 force precision"):
        plan("float64", workload(precision="float32"))


def test_the_fallback_has_an_mc_energy_dimension_first():
    assert fallback.DIMENSIONS[0] == "mc_energy"
    assert "mc_energy" in fallback.DIMENSIONS


def test_a_move_only_failure_is_recognised():
    record = selfcheck.GateRecord(status="skipped", check="skip",
                                  move_energy={"status": "failed", "precision": "float32"})
    failure = fallback.Failure(plan=None, stage="gate", error=GateFailed("x"), gate=record)
    assert fallback.move_energy_only(failure)
    md = dataclasses.replace(record, status="failed", move_energy={"status": "passed"})
    assert not fallback.move_energy_only(fallback.Failure(plan=None, stage="gate", error=GateFailed("x"), gate=md))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_fallback_tries_float64_mc_energies_first(pristine, no_engines_yet, monkeypatch):
    from fastmace.model.forces import MACEForceField

    original = MACEForceField.energy_only

    def float32_error(self, positions):  # every float32 force field: a constant error per Na atom
        energy = original(self, positions)
        if self.dtype == torch.float32:
            energy = energy + 1e-3 * (self.atomic_numbers == NA).sum().to(torch.float64)
        return energy

    monkeypatch.setattr(MACEForceField, "energy_only", float32_error)
    engine, moves, plan, ff = built(pristine, precision="float32")
    with pytest.raises(GateFailed) as raised:
        selfcheck.move_gate(engine, moves, plan=plan, force_field=ff, requested="float32")
    del engine
    failure = fallback.Failure(plan=plan, stage="gate", error=raised.value, gate=raised.value.gate_record)
    assert fallback.move_energy_only(failure)
    references = selfcheck.references(plan, ff, structure(), NVT, check="skip", seed=3)
    result = fallback.Fallback(cpu_planner(), ff, structure(), NVT, references, moves=moves).run(failure)
    assert result.plan.precision.mc_energy == "float64" and result.plan.precision.dtype == "float32"
    assert result.tried[0].dimension == "mc_energy" and result.tried[0].outcome == "activated"
    assert result.gate.move_energy["status"] == "passed"
    assert [entry["precision"] for entry in result.gate.move_energy["tried"]] == ["float64"]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_move_references_in_a_child_process_equal_in_process(pristine, no_engines_yet):
    engine, moves, plan, ff = built(pristine)  # in-process: no production engine existed yet
    assert moves.ran == "in-process"
    child = selfcheck.move_references(plan, ff, structure(), workload(), seed=3)
    assert child.ran == "child process" and child.unavailable is None
    np.testing.assert_allclose(child.energies, moves.energies, rtol=0, atol=1e-10)
    record = selfcheck.move_gate(engine, child, plan=plan, force_field=ff)
    assert record.move_energy["status"] == "passed"


# --------------------------------------------------------------------------- through Simulation (after 3-T15)


def simulation(pristine, precision, **options):
    from fastmace import integrators
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    atoms = structure()
    return Simulation(atoms, ForceField(pristine, atoms, precision=precision, species=SUPERSET),
                      integrator=integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01),
                      workload=workload(**options), seed=3)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_simulation_runs_the_move_energy_gate_under_skip(pristine, no_engines_yet):
    sim = simulation(pristine, "float64")
    assert sim.gate.status == "skipped"  # the energy, force and stress gate stays opt-in
    assert sim.gate.move_energy["status"] == "passed" and sim.gate.mc_energy_precision == "float64"
    assert sim.plan.precision.mc_energy == "float64"
    text = sim.report.text()
    assert "mc ΔE:" in text and "move-energy gate passed" in text


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_simulation_moves_auto_to_float64_and_the_driver_uses_it(pristine, no_engines_yet, monkeypatch):
    from fastmace import mc
    from fastmace.model.forces import MACEForceField

    original = MACEForceField.energy_only

    def float32_error(self, positions):
        energy = original(self, positions)
        if self.dtype == torch.float32:
            energy = energy + 1e-3 * (self.atomic_numbers == NA).sum().to(torch.float64)
        return energy

    monkeypatch.setattr(MACEForceField, "energy_only", float32_error)
    sim = simulation(pristine, "float32", moves=("swap", "transmute"))
    assert sim.plan.precision.dtype == "float32" and sim.plan.precision.mc_energy == "float64"
    assert sim.plan.fallbacks[-1].dimension == "mc_energy"
    assert [entry["status"] for entry in sim.gate.move_energy["tried"]] == ["failed", "passed"]
    sim.run(4, mc=mc.Transmute(["Na", "Cl"], every=2, attempts=1))
    assert sim._mc.evaluator.precision == "float64"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_charged_model_gates_against_references_at_its_charge(tmp_path, no_engines_yet):
    """The references carry the model's total charge and spin (found by 3-T11: they ran at charge 0)."""
    pytest.importorskip("cuequivariance_torch")
    from _support.mc_fixtures import tiny_loaded
    from fastmace.config import IntegratorConfig
    from fastmace.model.forcefield import ForceField
    from fastmace.planner.plan import Hardware

    torch.serialization.add_safe_globals([slice])
    superset = (11, 12, 17)
    loaded = tiny_loaded(tmp_path, superset, family="omol", charge=1, spin_multiplicity=1)
    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    numbers = atoms.numbers.copy()
    numbers[2] = 12  # one Mg: total charge 1 with even electrons
    atoms.set_atomic_numbers(numbers)
    ff = ForceField(loaded, atoms, precision="float64", species=list(superset))
    load = Workload(max_temperature_K=1000.0, species_changes=True, min_mc_temperature_K=1.0,
                    mc_moves=("transmute",))
    planner = cpu_planner()
    plan = planner.plan(ff, atoms, IntegratorConfig(temperature_K=300.0), load, Hardware("cpu", (0, 0), 0))
    references = selfcheck.move_references(plan, ff, atoms, load, seed=1)
    engine = planner.build(plan, ff, atoms)
    record = selfcheck.move_gate(engine, references, plan=plan, force_field=ff)
    assert record.move_energy["status"] == "passed", record.move_energy.get("reason")
    assert max(row["error"] for row in record.move_energy["rows"]) < 1e-9
