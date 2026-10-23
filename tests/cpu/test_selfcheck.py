"""The gate on the user's structure (plan task 1b-T16; SPEC §12.3, §12.2 gate row, §9.3 rule 14, §7.6).

- The ratio rule with the floors of §12.3 on synthetic numbers; NaN and inf anywhere fail before
  any maximum is taken (1a "second outside review", P1).
- The copies: 2 rattled (0.05 Å), and for a variable-cell integrator 2 strained (±2 %) as well,
  drawn from the seed.
- ``check="skip"`` (the default, owner decision D1) runs nothing and records ``skipped``; an
  unknown mode is a ``ValueError``.
- On the CPU (tiny models, float64 production, eager stages): NVT and NPT pass; NVT records
  "stress: not in plan"; the registered state is bitwise unchanged after the gate and the
  trajectory equals one without the gate; a species-dependent constant energy offset in the
  production path, invisible to forces and stress, fails on energy alone (``GateFailed``); a
  reference that cannot run is ``unavailable`` (``GateUnavailable`` under ``check="gate"``, a
  recorded record under ``check="allow_unavailable"``); after a production engine exists in the
  process the references run in a child process and give the same numbers.
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
from fastmace.errors import FastMACEError, GateFailed, GateUnavailable
from fastmace.planner import selfcheck
from fastmace.planner.workload import Workload
from fastmace.reference import records

NA_CL = (11, 17)
WORKLOAD = Workload(max_temperature_K=300.0)
NVT = IntegratorConfig(temperature_K=300.0)
NPT = IntegratorConfig(ensemble="npt", pressure_GPa=0.5, temperature_K=300.0)
NPT_FULL = dataclasses.replace(NPT, cell_mode="full")


# --------------------------------------------------------------------------- the rule (synthetic numbers)


def row(prod=(0.1, 1e-4, 1e-6), yard=(0.1, 1e-4, 1e-6)):
    """One copy's errors: (energy meV/atom, max force component eV/Å, max stress component eV/Å^3)."""
    def errors(values):
        energy, force, stress = values
        return {"energy_meV_per_atom": energy, "force_eV_per_A": force, "stress_eV_per_A3": stress}

    return {"production": errors(prod), "yardstick": errors(yard)}


def test_ratio_rule_passes_at_twice_the_yardstick_and_fails_above():
    assert selfcheck.verdict([row(prod=(0.2, 2e-4, 2e-6))])["passed"]
    result = selfcheck.verdict([row(prod=(0.21, 1e-4, 1e-6))])
    assert not result["passed"]
    assert [name for name, q in result["quantities"].items() if not q["passed"]] == ["energy_meV_per_atom"]
    assert not selfcheck.verdict([row(prod=(0.1, 2.1e-4, 1e-6))])["passed"]
    assert not selfcheck.verdict([row(prod=(0.1, 1e-4, 2.1e-6))])["passed"]


def test_ratio_rule_takes_the_worst_copy_of_each_path():
    rows = [row(prod=(0.3, 1e-4, 1e-6), yard=(0.1, 1e-4, 1e-6)), row(prod=(0.1, 1e-4, 1e-6), yard=(0.2, 1e-4, 1e-6))]
    energy = selfcheck.verdict(rows)["quantities"]["energy_meV_per_atom"]
    assert (energy["production"], energy["yardstick"], energy["limit"]) == (0.3, 0.2, 0.4)
    assert energy["passed"]


def test_floors_hold_where_the_yardstick_is_nearly_exact():
    assert selfcheck.FLOORS == {"energy_meV_per_atom": 0.01, "force_eV_per_A": 1e-5, "stress_eV_per_A3": 1e-7}
    tiny = (0.0, 0.0, 0.0)
    assert selfcheck.verdict([row(prod=(0.01, 1e-5, 1e-7), yard=tiny)])["passed"]
    for index, value in enumerate((0.011, 1.1e-5, 1.1e-7)):
        prod = [0.0, 0.0, 0.0]
        prod[index] = value
        assert not selfcheck.verdict([row(prod=tuple(prod), yard=tiny)])["passed"], index


POLAR_FLOORS = {"stress_eV_per_A3": 5e-7}  # SPEC decision 49, the PolarMACE record's entry (4-T12)


def test_a_record_floor_raises_only_its_quantity():
    floors = selfcheck.effective_floors(POLAR_FLOORS)
    assert floors == {**selfcheck.FLOORS, "stress_eV_per_A3": 5e-7}
    assert selfcheck.effective_floors({}) == selfcheck.FLOORS and selfcheck.effective_floors(None) == selfcheck.FLOORS
    tiny = (0.0, 0.0, 0.0)
    rows = [row(prod=(0.0, 0.0, 3e-7), yard=tiny)]
    assert not selfcheck.verdict(rows)["passed"]
    stress = selfcheck.verdict(rows, floors=floors)["quantities"]["stress_eV_per_A3"]
    assert stress["passed"] and stress["floor"] == 5e-7 and stress["limit"] == 5e-7
    assert stress["floor_source"] == "reference record"
    assert selfcheck.verdict(rows)["quantities"]["stress_eV_per_A3"]["floor_source"] == "SPEC §12.3"
    # A stress error above the PolarMACE floor is still caught, and the ratio rule still binds above it.
    assert not selfcheck.verdict([row(prod=(0.0, 0.0, 5.1e-7), yard=tiny)], floors=floors)["passed"]
    assert selfcheck.verdict([row(prod=(0.0, 0.0, 8e-7), yard=(0.0, 0.0, 4e-7))], floors=floors)["passed"]
    assert not selfcheck.verdict([row(prod=(0.0, 0.0, 8.1e-7), yard=(0.0, 0.0, 4e-7))], floors=floors)["passed"]
    # The energy and force floors stay the general ones.
    assert not selfcheck.verdict([row(prod=(0.011, 0.0, 0.0), yard=tiny)], floors=floors)["passed"]
    assert not selfcheck.verdict([row(prod=(0.0, 1.1e-5, 0.0), yard=tiny)], floors=floors)["passed"]
    assert selfcheck.FLOORS["stress_eV_per_A3"] == 1e-7  # the module's floors are unchanged


def test_the_1b_t23_nacl_rounding_case_passes_under_the_polar_floor():
    # 1b-T23: POLAR-1-S on NaCl 3x3x3, hybrid stress 1.2e-7 eV/Å^3 against a 3-4e-8 float32 yardstick.
    rows = [row(prod=(0.0, 0.0, 1.2e-7), yard=(0.0, 0.0, 4e-8))]
    assert not selfcheck.verdict(rows)["passed"]
    assert selfcheck.verdict(rows, floors=selfcheck.effective_floors(POLAR_FLOORS))["passed"]


def test_a_record_may_only_raise_a_known_floor():
    with pytest.raises(ValueError, match="unknown"):
        selfcheck.effective_floors({"pressure_GPa": 1.0})
    with pytest.raises(ValueError, match="below"):
        selfcheck.effective_floors({"stress_eV_per_A3": 1e-8})


@pytest.mark.parametrize("path", ["production", "yardstick"])
@pytest.mark.parametrize("quantity", ["energy_meV_per_atom", "force_eV_per_A", "stress_eV_per_A3"])
@pytest.mark.parametrize("bad", [math.nan, math.inf])
@pytest.mark.parametrize("first", [True, False])
def test_nonfinite_anywhere_fails_before_any_maximum(path, quantity, bad, first):
    rows = [row(), row()]
    rows[0 if first else 1][path][quantity] = bad
    result = selfcheck.verdict(rows)
    assert not result["passed"]
    assert result["nonfinite"] == [f"{0 if first else 1}/{path}/{quantity}"]


def test_nvt_rows_carry_no_stress_and_record_it():
    rows = [row(prod=(0.1, 1e-4, None), yard=(0.1, 1e-4, None))]
    result = selfcheck.verdict(rows)
    assert result["passed"]
    assert result["quantities"]["stress_eV_per_A3"] == {"passed": True, "note": "stress: not in plan"}


def test_errors_of_one_copy():
    reference = (-10.0, np.zeros((2, 3)), np.zeros((3, 3)))
    candidate = (-10.0 + 2e-3, np.array([[0.0, 3e-4, 0.0], [-5e-4, 0.0, 0.0]]), np.full((3, 3), -2e-6))
    errors = selfcheck.copy_errors(reference, candidate, natoms=2)
    assert errors == pytest.approx({"energy_meV_per_atom": 1.0, "force_eV_per_A": 5e-4, "stress_eV_per_A3": 2e-6})
    assert selfcheck.copy_errors(reference, (candidate[0], candidate[1], None), natoms=2)["stress_eV_per_A3"] is None


# --------------------------------------------------------------------------- the copies


def test_copies_rattle_and_strain_from_the_seed():
    atoms = nacl()
    nvt = selfcheck.make_copies(atoms, NVT, seed=7)
    assert [copy.kind for copy in nvt] == ["rattled", "rattled"]
    for copy in nvt:
        np.testing.assert_array_equal(copy.atoms.cell.array, atoms.cell.array)
        displacement = copy.atoms.positions - atoms.positions
        assert 0.03 < displacement.std() < 0.07
    npt = selfcheck.make_copies(atoms, NPT, seed=7)
    assert [copy.kind for copy in npt] == ["rattled", "rattled", "strained", "strained"]
    for copy in npt[2:]:
        strain = np.linalg.solve(atoms.cell.array, copy.atoms.cell.array)
        assert np.abs(strain - np.identity(3)).max() <= 0.02 + 1e-12
        np.testing.assert_allclose(strain, strain[0, 0] * np.identity(3), atol=1e-12)  # hydrostatic: isotropic
    full = selfcheck.make_copies(atoms, NPT_FULL, seed=7)
    strain = np.linalg.solve(atoms.cell.array, full[2].atoms.cell.array)
    assert np.abs(strain - np.diag(np.diag(strain))).max() > 0  # full cell: shear as well
    again = selfcheck.make_copies(atoms, NPT, seed=7)
    for one, two in zip(npt, again, strict=True):
        np.testing.assert_array_equal(one.atoms.positions, two.atoms.positions)
    assert not np.array_equal(selfcheck.make_copies(atoms, NVT, seed=8)[0].atoms.positions, nvt[0].atoms.positions)
    assert selfcheck.describe_copies(npt) == "2 rattled (0.05 Å), 2 strained (±2 %, hydrostatic)"
    assert selfcheck.describe_copies(nvt) == "2 rattled (0.05 Å)"


def test_public_integrator_records_are_accepted():
    from fastmace import integrators

    record = integrators.LangevinBAOAB(temperature_K=300.0, friction_per_fs=0.002, cell_mode="full")
    assert [copy.kind for copy in selfcheck.make_copies(nacl(), record, seed=1)].count("strained") == 2


# --------------------------------------------------------------------------- check modes


def test_unknown_check_mode_is_a_value_error():
    with pytest.raises(ValueError, match="allow_unavailable"):
        selfcheck.references(None, None, nacl(), NVT, check="on", seed=0)


def test_skip_records_skipped_and_runs_nothing():
    references = selfcheck.references(None, None, nacl(), NVT, check="skip", seed=0)
    record = selfcheck.production(None, references, check="skip")
    assert record.status == "skipped"
    metadata = record.to_metadata()
    for key in ("status", "reason", "copies", "energy_meV_per_atom", "force_eV_per_A", "stress_eV_per_A3"):
        assert key in metadata
    assert metadata["reason"] == 'opt-in: check="gate"'
    assert metadata["energy_meV_per_atom"] is None


def test_production_check_must_match_the_references():
    references = selfcheck.references(None, None, nacl(), NVT, check="skip", seed=0)
    with pytest.raises(ValueError, match="check"):
        selfcheck.production(None, references, check="gate")


# --------------------------------------------------------------------------- end to end on the CPU


@pytest.fixture
def no_engines_yet():
    """The process record of production engines, emptied for the test and put back afterwards."""
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

    def make(family="mh1"):
        model = tiny_mace_model(list(NA_CL), **FAMILIES[family])
        path = tmp_path / f"tiny-{family}.model"
        torch.save(model, path)
        return loading.load_pristine(str(path), trusted=True)

    return make


def cpu_planner():
    from fastmace.planner.minimal import MinimalPlanner

    return MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ={},
                          process=constraints.ProcessRecord())


def setup(pristine, integrator=NVT, family="mh1", atoms=None):
    from fastmace.model.forcefield import ForceField
    from fastmace.planner.plan import Hardware

    atoms = nacl(rattle=0.05, seed=1) if atoms is None else atoms
    ff = ForceField(pristine(family), atoms, precision="float64")
    planner = cpu_planner()
    plan = planner.plan(ff, atoms, integrator, WORKLOAD, Hardware("cpu", (0, 0), 0))
    return planner, plan, ff, atoms


def gated(pristine, integrator=NVT, check="gate", **kwargs):
    planner, plan, ff, atoms = setup(pristine, integrator, **kwargs)
    references = selfcheck.references(plan, ff, atoms, integrator, check=check, seed=3)
    engine = planner.build(plan, ff, atoms)
    return engine, references, (planner, plan, ff, atoms)


def registered(engine):
    return {name: tensor.detach().clone() for name, tensor in engine.registry.tensors().items()}


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("integrator", [NVT, NPT, NPT_FULL], ids=["nvt", "npt", "npt-full"])
def test_gate_passes_on_the_cpu_and_leaves_the_state_unchanged(pristine, no_engines_yet, integrator):
    engine, references, _ = gated(pristine, integrator)
    assert references.ran == "in-process"
    before = registered(engine)
    record = selfcheck.production(engine, references, check="gate")
    assert record.status == "passed", record.to_metadata()
    after = registered(engine)
    assert before.keys() == after.keys()
    for name in before:
        assert torch.equal(before[name], after[name]), name
    metadata = record.to_metadata()
    assert metadata["energy_meV_per_atom"] is not None and metadata["force_eV_per_A"] is not None
    assert metadata["references"]["record_id"].startswith(records.ID_PREFIX)
    assert metadata["references"]["ran"] == "in-process"
    assert metadata["mode"] == "eager stages C, B"
    if integrator.ensemble == "nvt":
        assert metadata["stress_eV_per_A3"] is None
        assert metadata["quantities"]["stress_eV_per_A3"]["note"] == "stress: not in plan"
        assert metadata["copies"] == "2 rattled (0.05 Å)"
    else:
        assert metadata["stress_eV_per_A3"] is not None
        assert "2 strained" in metadata["copies"]


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("integrator", [NVT, NPT], ids=["nvt", "npt"])
def test_trajectory_after_the_gate_equals_one_without(pristine, no_engines_yet, integrator):
    engine, references, (planner, plan, ff, atoms) = gated(pristine, integrator)
    selfcheck.production(engine, references, check="gate")
    plain = planner.build(plan, ff, atoms)
    for target in (engine, plain):
        target.run(8)
        target.flush()
    np.testing.assert_array_equal(engine.state.positions.numpy(), plain.state.positions.numpy())
    np.testing.assert_array_equal(engine.state.velocities.numpy(), plain.state.velocities.numpy())
    assert engine.potential_energy() == plain.potential_energy()
    if integrator.ensemble == "npt":
        np.testing.assert_array_equal(engine.state.cell.numpy(), plain.state.cell.numpy())


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_species_constant_energy_offset_fails_on_energy_alone(pristine, no_engines_yet):
    from fastmace.model import energy as energy64

    engine, references, _ = gated(pristine, NPT)
    e0 = engine.force_field.data[energy64.NODE_E0_KEY]
    sodium = torch.as_tensor(engine.force_field.numbers == 11)
    with torch.no_grad():
        e0[sodium] += 1e-3  # 1 meV per Na atom: 0.5 meV/atom, invisible to forces and stress
    with pytest.raises(GateFailed) as raised:
        selfcheck.production(engine, references, check="gate")
    record = raised.value.gate_record
    assert record.status == "failed"
    failed = [name for name, q in record.to_metadata()["quantities"].items() if not q["passed"]]
    assert failed == ["energy_meV_per_atom"]
    assert record.to_metadata()["energy_meV_per_atom"] == pytest.approx(0.5, rel=1e-6)
    assert "energy" in str(raised.value) and "check" not in record.reason
    assert isinstance(raised.value, FastMACEError)


def _stress_offset(monkeypatch, offset):
    """Production's stress off by ``offset`` eV/Å^3 in xx: a deliberate stress error, nothing else changed."""
    original = selfcheck._read_outputs

    def shifted(engine_, stress):
        energy, forces, values = original(engine_, stress)
        values = values.copy()
        values[0, 0] += offset
        return energy, forces, values

    monkeypatch.setattr(selfcheck, "_read_outputs", shifted)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("offset,floors,passes", [(3e-7, None, False), (3e-7, POLAR_FLOORS, True),
                                                  (6e-7, POLAR_FLOORS, False)],
                         ids=["general-3e-7", "polar-3e-7", "polar-6e-7"])
def test_the_gate_reads_the_stress_floor_from_the_reference_record(pristine, no_engines_yet, monkeypatch,
                                                                    offset, floors, passes):
    """SPEC decision 49: the record's floors reach the gate through the reference's info, and are recorded."""
    engine, references, _ = gated(pristine, NPT)
    if floors is not None:
        references.info["reference"]["gate_floors"] = dict(floors)
    _stress_offset(monkeypatch, offset)
    if passes:
        record = selfcheck.production(engine, references, check="gate")
    else:
        with pytest.raises(GateFailed) as raised:
            selfcheck.production(engine, references, check="gate")
        record = raised.value.gate_record
        assert f"floor {(floors or selfcheck.FLOORS)['stress_eV_per_A3']:g}" in record.reason
    metadata = record.to_metadata()
    assert metadata["status"] == ("passed" if passes else "failed")
    assert metadata["floors"] == {**selfcheck.FLOORS, **(floors or {})}
    stress = metadata["quantities"]["stress_eV_per_A3"]
    assert stress["floor"] == metadata["floors"]["stress_eV_per_A3"]
    assert stress["floor_source"] == ("reference record" if floors else "SPEC §12.3")
    assert stress["production"] == pytest.approx(offset, rel=0.05)
    assert metadata["quantities"]["force_eV_per_A"]["floor"] == selfcheck.FLOORS["force_eV_per_A"]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_nonfinite_production_value_fails(pristine, no_engines_yet, monkeypatch):
    engine, references, _ = gated(pristine, NVT)
    original = selfcheck._read_outputs

    def poisoned(engine_, stress):
        energy, forces, values = original(engine_, stress)
        forces = forces.copy()
        forces[0, 0] = np.nan
        return energy, forces, values

    monkeypatch.setattr(selfcheck, "_read_outputs", poisoned)
    with pytest.raises(GateFailed) as raised:
        selfcheck.production(engine, references, check="gate")
    assert raised.value.gate_record.to_metadata()["nonfinite"] == ["0/production/force_eV_per_A",
                                                                  "1/production/force_eV_per_A"]


def _no_reference(*args, **kwargs):
    raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 12.00 GiB")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_an_unavailable_reference_refuses_a_requested_gate(pristine, no_engines_yet, monkeypatch):
    planner, plan, ff, atoms = setup(pristine)
    monkeypatch.setattr(selfcheck, "_load_reference", _no_reference)
    with pytest.raises(GateUnavailable) as raised:
        selfcheck.references(plan, ff, atoms, NVT, check="gate", seed=0)
    assert 'check="allow_unavailable"' in str(raised.value)
    assert raised.value.gate_record.status == "unavailable"
    assert "out of memory" in raised.value.gate_record.reason
    assert not base.production_engine_built()  # refused before any engine was built


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_allow_unavailable_records_and_goes_on(pristine, no_engines_yet, monkeypatch):
    planner, plan, ff, atoms = setup(pristine)
    monkeypatch.setattr(selfcheck, "_load_reference", _no_reference)
    references = selfcheck.references(plan, ff, atoms, NVT, check="allow_unavailable", seed=0)
    engine = planner.build(plan, ff, atoms)
    record = selfcheck.production(engine, references, check="allow_unavailable")
    assert record.status == "unavailable"
    assert record.to_metadata()["check"] == "allow_unavailable"
    assert "out of memory" in record.reason


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_references_run_in_a_child_process_after_a_production_engine(pristine, no_engines_yet):
    planner, plan, ff, atoms = setup(pristine, NPT)
    inside = selfcheck.references(plan, ff, atoms, NPT, check="gate", seed=3)
    assert inside.ran == "in-process"
    planner.build(plan, ff, atoms)
    assert base.production_engine_built()
    child = selfcheck.references(plan, ff, atoms, NPT, check="gate", seed=3)
    assert child.ran == "child process"
    assert child.info["reference"]["record_id"] == inside.info["reference"]["record_id"]
    for one, two in zip(inside.values, child.values, strict=True):
        for role in ("reference", "yardstick"):
            energy_a, forces_a, stress_a = one[role]
            energy_b, forces_b, stress_b = two[role]
            assert energy_a == pytest.approx(energy_b, abs=1e-12, rel=0)
            np.testing.assert_allclose(forces_a, forces_b, rtol=0, atol=1e-12)
            np.testing.assert_allclose(stress_a, stress_b, rtol=0, atol=1e-12)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_references_are_freed_before_the_engine_is_built(pristine, no_engines_yet, monkeypatch):
    import gc
    import weakref

    planner, plan, ff, atoms = setup(pristine)
    built = []
    original = selfcheck._load_reference

    def tracked(*args, **kwargs):
        reference = original(*args, **kwargs)
        built.append(weakref.ref(reference.model))
        return reference

    monkeypatch.setattr(selfcheck, "_load_reference", tracked)
    selfcheck.references(plan, ff, atoms, NVT, check="gate", seed=0)
    gc.collect()
    assert len(built) == 2  # one calculator at a time: float64 reference, then float32 yardstick
    assert all(ref() is None for ref in built)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_grown_neighbour_capacity_is_rebuilt_at_the_committed_state(pristine, no_engines_yet, monkeypatch):
    planner, plan, ff, atoms = setup(pristine, NPT)
    original = selfcheck.make_copies

    def with_a_compressed_copy(atoms_, integrator, seed):
        copies = original(atoms_, integrator, seed)
        compressed = atoms_.copy()
        compressed.set_cell(compressed.cell.array * 0.85, scale_atoms=True)  # the third NaCl shell enters
        return copies[:1] + [selfcheck.GateCopy("strained", compressed)] + copies[1:]

    monkeypatch.setattr(selfcheck, "make_copies", with_a_compressed_copy)
    references = selfcheck.references(plan, ff, atoms, NPT, check="gate", seed=3)
    engine = planner.build(plan, ff, atoms)
    capacity = engine.force_field.edge_capacity
    record = selfcheck.production(engine, references, check="gate")
    assert engine.force_field.edge_capacity > capacity
    assert record.status == "passed" and "capacity grew" in record.notes[0]
    plain = planner.build(plan, ff, atoms)
    for target in (engine, plain):
        target.run(8)
        target.flush()
    np.testing.assert_array_equal(engine.state.positions.numpy(), plain.state.positions.numpy())
    np.testing.assert_array_equal(engine.state.cell.numpy(), plain.state.cell.numpy())
    assert engine.potential_energy() == plain.potential_energy()
