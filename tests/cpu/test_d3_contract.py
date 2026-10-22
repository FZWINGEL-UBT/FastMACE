"""The phase 3b D3 contract (plan task 3b-T01; SPEC §11.2, docs/design/d3.md).

- ``fm.D3`` carries the explicit Becke–Johnson numbers (s6, s8, a1, a2), filled from the
  functional's parameter set (torch-dftd 0.5.1's table) or given explicitly, and only float64
  reductions (conductor's choice 3);
- ``ForceField(dispersion=...)`` validates and expands the record, puts it in the force-field
  identity, refuses PolarMACE + D3 (decision 41) and plans with D3 until 3b-T04 integrates it;
- ``DispersionTerm`` is the interface the D3 list (3b-T02) and kernel (3b-T03) implement.
"""

from __future__ import annotations

import dataclasses
import json

import pytest
import torch

from fastmace.longrange import d3
from fastmace.longrange.d3 import D3

R2SCAN = dict(s6=1.0, s8=0.7898, a1=0.4948, a2=5.7308)
PBE = dict(s6=1.0, s8=0.7875, a1=0.4289, a2=4.4407)


# --------------------------------------------------------------------------- the record


@pytest.mark.parametrize("functional,numbers", [("r2scan", R2SCAN), ("pbe", PBE)])
def test_functional_fills_the_bj_numbers(functional, numbers):
    record = D3(functional=functional)
    assert {name: getattr(record, name) for name in numbers} == numbers


def test_explicit_numbers_equal_the_named_set():
    assert D3(functional="r2scan", **R2SCAN) == D3(functional="r2scan")


def test_explicit_numbers_name_any_set():
    record = D3(functional="my-functional", s6=1.0, s8=0.5, a1=0.4, a2=5.0)
    assert (record.s8, record.a1, record.a2) == (0.5, 0.4, 5.0)


def test_explicit_numbers_override_the_named_set():
    record = D3(functional="r2scan", **dict(R2SCAN, s8=0.8))
    assert record.s8 == 0.8 and record != D3(functional="r2scan")


def test_unknown_functional_needs_all_four_numbers():
    with pytest.raises(ValueError, match="b3lyp.*s6, s8, a1, a2"):
        D3(functional="b3lyp")
    with pytest.raises(ValueError, match="all four"):
        D3(functional="r2scan", s8=0.8)


@pytest.mark.parametrize("name,value", [("s6", -1.0), ("s8", float("nan")), ("a1", "0.4"), ("a2", True),
                                        ("s8", -0.1)])
def test_bad_numbers(name, value):
    with pytest.raises(ValueError, match=name):
        D3(functional="r2scan", **dict(R2SCAN, **{name: value}))


def test_zero_s8_is_allowed():
    # Some published sets switch the C8 term off (PBEh-3c: s8 = 0).
    assert D3(functional="x", s6=1.0, s8=0.0, a1=0.4, a2=4.5).s8 == 0.0


def test_precision_is_the_reduction_precision_float64_only():
    assert D3(functional="r2scan").precision == "float64"
    with pytest.raises(ValueError, match="precision"):
        D3(functional="r2scan", precision="float32")


def test_unit_constants_and_cutoffs_in_angstrom():
    record = D3(functional="r2scan")
    assert (record.autoang, record.autoev) == (0.52917726, 27.21138505)
    assert d3.pair_cutoff_A(record) == pytest.approx(40 * 0.52917726, rel=0, abs=1e-15)
    assert d3.cn_cutoff_A(record) == pytest.approx(40 * 0.52917726, rel=0, abs=1e-15)


def test_record_text_names_the_numbers():
    text = d3.describe(D3(functional="r2scan"))
    for field in ("s6=1.0", "s8=0.7898", "a1=0.4948", "a2=5.7308"):
        assert field in text


def test_identity_entry_has_every_field_and_the_tables():
    record = D3(functional="r2scan")
    entry = d3.identity_entry(record)
    for field in dataclasses.fields(D3):
        assert entry[field.name] == getattr(record, field.name)
    assert entry["tables"] == d3.REFERENCE_TABLES
    assert json.loads(json.dumps(entry)) == entry


# --------------------------------------------------------------------------- ForceField(dispersion=)


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    pytest.importorskip("cuequivariance_torch")
    from _support.mc_fixtures import tiny_loaded

    return tiny_loaded(tmp_path_factory.mktemp("models"), (11, 17))


def _with_functional(loaded, functional):
    return dataclasses.replace(loaded, functional=functional)


def test_forcefield_default_has_no_dispersion_and_an_unchanged_identity(loaded):
    from fastmace.model.forcefield import ForceField

    field = ForceField(loaded, species=[11, 17])
    assert field.dispersion is None and field.dispersion_warnings == ()
    assert json.loads(field.identity.physics) == {"charge": loaded.charge,
                                                  "spin_multiplicity": loaded.spin_multiplicity}


def test_forcefield_record_goes_into_the_identity(loaded):
    from fastmace.model.forcefield import ForceField

    record = D3(functional="r2scan")
    field = ForceField(loaded, species=[11, 17], dispersion=record)
    assert field.dispersion is record
    physics = json.loads(field.identity.physics)
    assert physics["dispersion"] == d3.identity_entry(record)
    plain = ForceField(loaded, species=[11, 17])
    assert field.identity.digest != plain.identity.digest
    other = ForceField(loaded, species=[11, 17], dispersion=D3(functional="r2scan", **dict(R2SCAN, a2=5.8)))
    assert other.identity.digest != field.identity.digest


def test_forcefield_d3bj_expands_from_the_training_functional(loaded):
    from fastmace.model.forcefield import ForceField

    field = ForceField(_with_functional(loaded, "r2scan"), species=[11, 17], dispersion="d3bj")
    assert field.dispersion == D3(functional="r2scan")
    with pytest.raises(ValueError, match="unknown"):
        ForceField(loaded, species=[11, 17], dispersion="d3bj")  # a path: functional unknown
    with pytest.raises(ValueError, match="twice"):
        ForceField(_with_functional(loaded, "wb97m-v"), species=[11, 17], dispersion="d3bj")


def test_forcefield_other_functional_warns(loaded):
    from fastmace.model.forcefield import ForceField

    field = ForceField(_with_functional(loaded, "r2scan"), species=[11, 17], dispersion=D3(functional="pbe"))
    assert len(field.dispersion_warnings) == 1 and "pbe" in field.dispersion_warnings[0]


@pytest.mark.parametrize("value,error", [(3, TypeError), ("d3", ValueError), ({"functional": "pbe"}, TypeError)])
def test_forcefield_bad_dispersion(loaded, value, error):
    from fastmace.model.forcefield import ForceField

    with pytest.raises(error, match="dispersion"):
        ForceField(loaded, species=[11, 17], dispersion=value)


def test_forcefield_three_body_is_refused(loaded):
    from fastmace.errors import PlanRefused
    from fastmace.model.forcefield import ForceField

    with pytest.raises(PlanRefused, match="ATM") as raised:
        ForceField(loaded, species=[11, 17], dispersion=D3(functional="r2scan", three_body=True))
    assert raised.value.reasons[0].code == "d3-three-body"


def test_forcefield_polar_with_d3_is_refused(loaded, monkeypatch):
    from fastmace.errors import PlanRefused
    from fastmace.longrange import polar
    from fastmace.model.forcefield import ForceField

    monkeypatch.setattr(polar, "is_polar", lambda model: True)
    with pytest.raises(PlanRefused, match="PolarMACE") as raised:
        ForceField(loaded, species=[11, 17], dispersion=D3(functional="r2scan"))
    assert raised.value.reasons[0].code == "d3-polar"
    ForceField(loaded, species=[11, 17])  # PolarMACE without D3 is unchanged


def test_plans_with_d3_are_refused_until_integration(loaded, monkeypatch):
    from _support.mc_fixtures import cpu_planner
    from _support.structures import nacl
    from fastmace import integrators
    from fastmace.errors import PlanRefused
    from fastmace.model.forcefield import ForceField
    from fastmace.planner.workload import Workload

    planner = cpu_planner(monkeypatch)
    field = ForceField(loaded, species=[11, 17], dispersion=D3(functional="r2scan"))
    atoms = nacl(1)
    with pytest.raises(PlanRefused, match="phase 3b") as raised:
        integrator = integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01)
        planner.plan(field, atoms, integrator.to_config(seed=1), Workload(max_temperature_K=300.0))
    assert raised.value.reasons[0].code == "d3-phase-3b"
    with pytest.raises(PlanRefused, match="phase 3b"):
        field.evaluate(atoms)


# --------------------------------------------------------------------------- DispersionTerm


class _Constant(d3.DispersionTerm):
    """A pair-free term for the interface tests: energy -1 eV, no forces, rebuilds past 0.5 Å."""

    def _build(self, positions, cell):
        self.reference = positions.detach().clone()
        return self.builds == 0

    def needs_rebuild(self, positions, cell):
        return ((positions - self.reference) ** 2).sum(dim=1).max() > 0.25

    def evaluate(self, positions, cell, *, virial=True):
        return d3.DispersionResult(energy=torch.tensor(-1.0, dtype=torch.float64),
                                   forces=torch.zeros_like(positions),
                                   virial=torch.zeros(3, 3, dtype=torch.float64) if virial else None)

    def evaluate_energy(self, positions, cell):
        return torch.tensor(-1.0, dtype=torch.float64)


def test_dispersion_term_is_abstract():
    with pytest.raises(TypeError):
        d3.DispersionTerm(D3(functional="r2scan"), torch.tensor([11, 17]))
    assert d3.DispersionTerm.__abstractmethods__ == {"_build", "needs_rebuild", "evaluate", "evaluate_energy"}


def test_dispersion_term_counts_builds():
    numbers = torch.tensor([11, 17])
    term = _Constant(D3(functional="r2scan"), numbers)
    assert term.record == D3(functional="r2scan") and term.numbers is numbers
    assert term.counters() == {"builds": 0, "rebuilds": 0, "regrowths": 0, "build_seconds": 0.0}
    positions, cell = torch.zeros(2, 3, dtype=torch.float64), torch.eye(3, dtype=torch.float64) * 5
    step = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    assert term.build(positions, cell) is True
    assert not bool(term.needs_rebuild(positions + 0.4 * step, cell))
    assert bool(term.needs_rebuild(positions + 0.6 * step, cell))
    assert term.build(positions + 0.6 * step, cell) is False
    counters = term.counters()
    assert (counters["builds"], counters["rebuilds"], counters["regrowths"]) == (2, 1, 1)
    assert counters["build_seconds"] >= 0.0
    result = term.evaluate(positions, cell)
    assert result.energy.dtype == torch.float64 and result.forces.shape == (2, 3) and result.virial.shape == (3, 3)
    assert term.evaluate(positions, cell, virial=False).virial is None
    assert term.evaluate_energy(positions, cell).dtype == torch.float64
    term.species_changed()  # the default has no per-species tables to refresh


def test_dispersion_result_stress_sign():
    # virial = -dE/d(strain); ASE stress = -virial / volume.
    result = d3.DispersionResult(energy=torch.tensor(0.0, dtype=torch.float64), forces=torch.zeros(1, 3),
                                 virial=torch.eye(3, dtype=torch.float64) * 2.0)
    assert torch.equal(result.stress(volume=4.0), -torch.eye(3, dtype=torch.float64) * 0.5)
