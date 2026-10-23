"""The reference policy: versioned records and corrected reference calculators (plan task 1b-T13; SPEC §12.1, §8.4).

- records: the four kinds of entries per (MACE version, model family), a stable id that changes
  when any entry changes; the polar stress fix (MACE issue 1642) required for PolarMACE on
  MACE 0.3.16, PR 1762 as optional physics, D3 refused until phase 3b, the BAOAB correction
  from 1b-T09's qualification, float64 accumulation in the float32 yardstick;
- calculators (CPU float64, tiny models): energy, forces and stress equal stock MACE where the
  record says stock; the polar stress fix is active and agrees with finite differences of the
  reference's own energy; a reference built in the wrong order raises; the yardstick's energy
  carries the float64 sum and its forces and stress are stock float32's bit for bit; a record
  without the fix is refused for a polar model.
"""

import copy
import dataclasses
import importlib.util

import numpy as np
import pytest
import torch

from _support.structures import nacl
from _support.tiny_models import FAMILIES, polar_available, tiny_mace_model
from fastmace.errors import FastMACEError
from fastmace.reference import calculators, records
from fastmace.reference.integrators import QUALIFICATION

pytest.importorskip("mace")
torch.serialization.add_safe_globals([slice])

needs_polar = pytest.mark.skipif(not (polar_available() and importlib.util.find_spec("cuequivariance_torch")),
                                 reason="PolarMACE needs graph_longrange and cuequivariance_torch")
# Synthetic all-electron E0 (eV), none of them a float32 number (as tests/cpu/test_energy.py).
E0 = {8: -2048.1234567891, 11: -4412.9876543219, 17: -12522.6512345678, 26: -34510.4321987654}
CASES = {
    "mh1": ([8, 11, 17, 26], {}),
    "residual": ([8, 11, 17, 26], {}),
    "density": ([8, 11, 17, 26], {}),
    "off": ([8, 11, 17, 26], {}),
    "omol": ([8, 11, 17, 26], {"use_embedding_readout": True}),
    "polar": ([11, 17], {"add_local_electron_energy": True}),
}
STOCK_FAMILIES = ["mh1", "residual", "density", "off", "omol"]
ALL_FAMILIES = STOCK_FAMILIES + [pytest.param("polar", marks=needs_polar)]


# --------------------------------------------------------------------------- helpers


def e0_table(z_values):
    return torch.tensor([E0[z] for z in sorted(z_values)], dtype=torch.float64)


def model64(family):
    z_values, extra = CASES[family]
    model = tiny_mace_model(z_values, **{**FAMILIES[family], **extra})
    with torch.no_grad():
        model.atomic_energies_fn.atomic_energies.copy_(e0_table(z_values).reshape(
            model.atomic_energies_fn.atomic_energies.shape))
    return model


def mace_calculator(model, dtype="float64"):
    """MACE's calculator of a copy of ``model``, converted to cuEquivariance (naive CPU kernels)."""
    from mace.calculators import MACECalculator
    from mace.cli.convert_e3nn_cueq import run

    from fastmace.model import compile as fc

    calc = MACECalculator(models=[copy.deepcopy(model)], device="cpu", default_dtype=dtype)
    with fc.naive_cueq():
        calc.models = [run(calc.models[0], device="cpu", return_model=True)]
    return calc


def structure(family, rattle=0.1, seed=8, sheared=False):
    atoms = nacl(repeat=2 if not sheared else 1, rattle=rattle, seed=seed)
    if CASES[family][0][0] == 8:
        atoms.numbers[3] = 8
    if sheared:
        e = np.random.default_rng(20261016).uniform(-0.02, 0.02, (3, 3))
        atoms.set_cell(atoms.cell.array @ (np.identity(3) + 0.5 * (e + e.T)), scale_atoms=True)
    return atoms


def stock(calc, atoms):
    current = atoms.copy()
    current.calc = calc
    return current.get_potential_energy(), current.get_forces(), current.get_stress(voigt=False)


def record_of(model, **options):
    return records.record_for(records.mace_version(), family="polar" if "Polar" in type(model).__name__ else "mh1",
                              **options)


# --------------------------------------------------------------------------- records


def test_stock_record_has_the_four_kinds_of_entries():
    record = records.record_for("0.3.16", "mh1")
    assert record.mace_version == "0.3.16" and record.family == "mh1"
    # 1a-T19's "stock", stated exactly (conductor answer, 19 October 2026).
    assert record.stock["calculator"] == "mace.calculators.MACECalculator"
    assert record.stock["backend"] == "cueq"
    assert record.stock["cueq_conv_fusion"] is False
    assert record.stock["single_graph_strain"] is True
    assert record.stock["dtypes"] == {"reference": "float64", "yardstick": "float32"}
    assert record.upstream_fixes == ()
    assert record.optional_physics == ()
    assert [entry["id"] for entry in record.integrator_corrections] == [QUALIFICATION["id"]]
    assert record.integrator_corrections[0]["qualification"] == QUALIFICATION
    assert [entry["name"] for entry in record.numerical] == ["float64_energy_accumulation"]
    assert record.numerical[0]["applies_to"] == "yardstick"
    assert record.is_stock


def test_polar_record_carries_the_issue_1642_fix_and_its_validation():
    record = records.record_for("0.3.16", "polar")
    (fix,) = record.upstream_fixes
    assert fix["name"] == "polar_kspace_stress" and fix["issue"] == "MACE issue 1642"
    assert fix["fixed_upstream"] == "MACE PR 1718, after 0.3.16"
    validation = fix["validation"]
    assert validation["method"] == "central finite differences of the reference energy"
    assert validation["structure"] == "NaCl 3x3x3" and validation["tolerance_eV_per_A3"] == 1e-6
    # The qualified record's text is fixed (its id): 3b-T08 resolved this finding, the entry still states it.
    assert "4.4e-4" in fix["open_findings"][0] and "phase 3b" in fix["open_findings"][0]
    # A fix is a correction of a defect, not non-stock physics.
    assert record.is_stock


def test_record_ids_are_stable_and_change_with_every_entry():
    record = records.record_for("0.3.16", "polar")
    assert record.id == records.record_for("0.3.16", "polar").id
    assert record.id.startswith("fastmace.reference/") and record.id == record.to_dict()["id"]
    assert records.ReferenceRecord.from_dict(record.to_dict()) == record
    changed = {
        "schema": 2, "mace_version": "0.3.17", "family": "omol",
        "stock": {**record.stock, "cueq_conv_fusion": True},
        "upstream_fixes": (), "optional_physics": ({"name": "x"},),
        "integrator_corrections": (), "numerical": (),
    }
    base = records.record_for("0.3.16", "mh1")
    changed["upstream_fixes"] = record.upstream_fixes
    ids = {base.id}
    for field in dataclasses.fields(base):
        other = dataclasses.replace(base, **{field.name: changed[field.name]})
        assert other.id not in ids, field.name
        ids.add(other.id)


def test_record_id_is_a_force_field_identity_value():
    from fastmace.state import ForceFieldIdentity

    record = records.record_for("0.3.16", "mh1")
    identity = ForceFieldIdentity(checkpoint_sha256="0" * 64, head=None, species=(11, 17),
                                  force_precision="float32", reference_record_id=record.id,
                                  mace_version="0.3.16", fastmace_version="0.1")
    assert identity.reference_record_id == record.id


def test_pr1762_is_optional_physics_and_labels_the_record_non_stock():
    from fastmace.config import PolarConfig

    options = PolarConfig(fukui_regularisation=0.1, dipole_cap=(1.0, 2.0))
    record = records.record_for("0.3.16", "polar", polar=options)
    (entry,) = record.optional_physics
    assert entry["name"] == "pr1762" and entry["upstream"] == "MACE PR 1762 (open)"
    assert entry["options"] == {"fukui_regularisation": 0.1, "dipole_cap": [1.0, 2.0]}
    assert not record.is_stock
    assert record.id != records.record_for("0.3.16", "polar").id
    # The released options are stock: no entry.
    assert records.record_for("0.3.16", "polar", polar=PolarConfig()).optional_physics == ()
    with pytest.raises(records.ReferencePolicyError, match="PolarMACE"):
        records.record_for("0.3.16", "mh1", polar=options)


def test_d3_is_accepted_optional_physics_and_refused_for_polar():
    """3b-T04: MACE plus torch-dftd on the record's numbers, a non-stock entry; PolarMACE + D3 refused (decision 41)."""
    from fastmace.longrange import d3
    from fastmace.longrange.d3 import D3

    record = records.record_for("0.3.16", "mh1", d3=D3(functional="r2scan"))
    assert [entry["name"] for entry in record.optional_physics] == ["d3"] and not record.is_stock
    assert record.optional_physics[0]["parameters"] == d3.identity_entry(D3(functional="r2scan"))
    assert records.OPTIONAL_PHYSICS["d3"]["status"] == "accepted"
    with pytest.raises(records.ReferencePolicyError, match="PolarMACE") as caught:
        records.record_for("0.3.16", "polar", d3=D3(functional="r2scan"))
    assert isinstance(caught.value, FastMACEError)


def test_unqualified_mace_version_polar_and_unknown_family_are_refused():
    # Another MACE gets the unqualified stock record (2-T04, test_versioned_registry.py); PolarMACE not.
    assert not records.record_for("0.3.17", "mh1").qualified
    with pytest.raises(records.ReferencePolicyError, match="0.3.17"):
        records.record_for("0.3.17", "polar")
    with pytest.raises(records.ReferencePolicyError, match="family"):
        records.record_for("0.3.16", "nonsense")


class _FakePolar:
    """Stands in for a PolarMACE instance in the record checks, which go by ``polar.is_polar``."""


@pytest.fixture
def fake_polar(monkeypatch):
    from fastmace.longrange import polar

    original = polar.is_polar
    monkeypatch.setattr(polar, "is_polar", lambda model: isinstance(model, _FakePolar) or original(model))


def test_a_polar_record_without_the_fix_is_refused(fake_polar):
    record = records.record_for("0.3.16", "polar")
    with pytest.raises(records.ReferencePolicyError, match="1642"):
        dataclasses.replace(record, upstream_fixes=())
    # A non-polar record cannot stand in for a polar model's (fallbacks keep the correction).
    with pytest.raises(records.ReferencePolicyError, match="1642"):
        records.require_record_for_model(records.record_for("0.3.16", "mh1"), _FakePolar())
    records.require_record_for_model(records.record_for("0.3.16", "mh1"), model64("mh1"))
    records.require_record_for_model(record, _FakePolar())


def test_family_of_a_model(fake_polar):
    assert records.family_of(model64("mh1"), "mh1") == "mh1"
    assert records.family_of(model64("mh1"), None) == "unlabelled"
    assert records.family_of(_FakePolar(), None) == "polar"
    with pytest.raises(records.ReferencePolicyError, match="PolarMACE"):
        records.family_of(_FakePolar(), "mh1")


# --------------------------------------------------------------------------- calculators


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_float64_reference_equals_stock_mace(family):
    model, atoms = model64(family), structure(family)
    calc = mace_calculator(model)
    record = record_of(model)
    reference = calculators.ReferenceCalculator.from_calculator(mace_calculator(model), record, atoms)
    energy, forces, stress = reference.evaluate(atoms)
    if family == "polar":
        # The stock energy and forces; the stock stress is MACE 0.3.16's defect (checked below).
        expected_energy, expected_forces, _ = stock(calc, atoms)
    else:
        expected_energy, expected_forces, expected_stress = stock(calc, atoms)
        np.testing.assert_allclose(stress, expected_stress, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(energy, expected_energy, rtol=1e-12, atol=0)
    # PolarMACE: the corrected path sums the k-space terms in another order (test_polar: atol 1e-12).
    np.testing.assert_allclose(forces, expected_forces, rtol=1e-12, atol=1e-12 if family == "polar" else 1e-13)
    assert isinstance(energy, float) and forces.dtype == np.float64 and stress.shape == (3, 3)
    info = reference.info()
    assert info["record_id"] == record.id and info["dtype"] == "float64" and info["role"] == "reference"
    assert info["corrections_active"] == (["polar_kspace_stress"] if family == "polar" else [])


@needs_polar
def test_polar_stress_fix_is_active_against_finite_differences():
    model, atoms = model64("polar"), structure("polar", rattle=0.05, seed=3, sheared=True)
    reference = calculators.ReferenceCalculator.from_calculator(mace_calculator(model), record_of(model), atoms)
    _, _, stress = reference.evaluate(atoms)
    fd = calculators.finite_difference_stress(reference, atoms)
    _, _, released = stock(mace_calculator(model), atoms)
    scale = np.abs(fd).max()
    # 1a-T19's tolerance is 1e-6 eV/A^3 on NaCl 3x3x3; the tiny model's stress is of another size.
    assert np.abs(stress - fd).max() < 1e-5 * scale
    assert np.abs(released - fd).max() > 1e-2 * scale
    check = reference.check_stress_fix(atoms, tolerance=1e-5 * scale)
    assert check["passed"] and check["max_abs"] < 1e-5 * scale
    with pytest.raises(records.ReferencePolicyError, match="finite differences"):
        reference.check_stress_fix(atoms, tolerance=1e-12 * scale)


def cell_at_the_kspace_cutoff(model, atoms, n=(3, 0, 0), ratio=1 + 5e-6):
    """``atoms`` scaled so that the k-vector of integer coefficients ``n`` lies at ``ratio`` x the cutoff."""
    k = np.asarray(n, float) @ (2 * np.pi * np.linalg.inv(atoms.cell.array.T))
    scale = np.linalg.norm(k) / (float(model.kspace_cutoff) * ratio)
    atoms = atoms.copy()
    atoms.set_cell(atoms.cell.array * scale, scale_atoms=True)
    return atoms


@needs_polar
def test_polar_stress_check_freezes_the_kset_across_the_cutoff():
    """3b-T08: a k-vector just outside the hard k-space cutoff enters under +h strain, so the live energy
    steps and its finite differences go as 1/h; the self-test holds the unstrained cell's included set."""
    from fastmace.longrange import polar

    model = model64("polar")
    atoms = cell_at_the_kspace_cutoff(model, structure("polar", rattle=0.05, seed=3, sheared=True))
    reference = calculators.ReferenceCalculator.from_calculator(mace_calculator(model), record_of(model), atoms)
    _, _, stress = reference.evaluate(atoms)
    live = calculators.finite_difference_stress(reference, atoms)
    scale = np.abs(stress).max()
    with polar.frozen_kset(reference.model, atoms.cell.array):
        frozen = calculators.finite_difference_stress(reference, atoms)
    assert polar.get_state(reference.model).frozen_included is None
    # The (3, 0, 0) pair enters at e_xx = +h: the live xx difference has the step (1.1e-5; frozen, every
    # component <= 2.7e-7, the O(h^2) truncation; 23 October 2026). The others are the same either way.
    assert np.abs(stress - frozen).max() < 1e-6 * scale
    assert np.abs(stress - live)[0, 0] > 20 * np.abs(stress - frozen).max()
    off_xx = np.ones((3, 3), bool)
    off_xx[0, 0] = False
    np.testing.assert_array_equal(live[off_xx], frozen[off_xx])
    check = reference.check_stress_fix(atoms, tolerance=1e-6 * scale)
    assert check["passed"] and check["kset"] == "frozen" and check["max_abs"] < 1e-6 * scale
    np.testing.assert_array_equal(check["fd_stress"], frozen)
    # The live set is back afterwards: the same strained energy as before the check.
    assert polar.get_state(reference.model).frozen_included is None
    np.testing.assert_array_equal(calculators.finite_difference_stress(reference, atoms), live)


@needs_polar
def test_polar_reference_built_after_a_stored_forward_raises():
    """1a-T19: configure_md_calculator stored the bound forward before the corrected one was in place."""
    model, atoms = model64("polar"), structure("polar")
    calc = mace_calculator(model)
    calc.models[0].forward = calc.models[0].forward  # what configure_md_calculator does
    with pytest.raises(calculators.ReferenceOrderError, match="forward"):
        calculators.ReferenceCalculator.from_calculator(calc, record_of(model), atoms)


@pytest.mark.parametrize("family", ["mh1", pytest.param("polar", marks=needs_polar)])
def test_a_forward_stored_after_construction_raises_at_evaluation(family):
    model, atoms = model64(family), structure(family)
    reference = calculators.ReferenceCalculator.from_calculator(mace_calculator(model, "float32"),
                                                                record_of(model), atoms)
    stored = reference.model.forward
    reference.model.forward = stored
    with pytest.raises(calculators.ReferenceOrderError):
        reference.evaluate(atoms)


@needs_polar
def test_a_polar_reference_without_its_kspace_state_raises():
    from fastmace.longrange import polar

    model, atoms = model64("polar"), structure("polar")
    reference = calculators.ReferenceCalculator.from_calculator(mace_calculator(model), record_of(model), atoms)
    polar.disable(reference.model)
    with pytest.raises(calculators.ReferenceOrderError, match="1642"):
        reference.evaluate(atoms)


@needs_polar
def test_a_polar_model_with_a_record_without_the_fix_is_refused():
    model, atoms = model64("polar"), structure("polar")
    with pytest.raises(records.ReferencePolicyError, match="1642"):
        calculators.ReferenceCalculator.from_calculator(mace_calculator(model), records.record_for("0.3.16", "mh1"),
                                                        atoms)


@pytest.mark.parametrize("family", ["mh1", "omol", pytest.param("polar", marks=needs_polar)])
def test_yardstick_energy_carries_the_float64_sum_and_forces_are_stock(family):
    model, atoms = model64(family), structure(family)
    exact = e0_table(CASES[family][0]).reshape(model.atomic_energies_fn.atomic_energies.shape)
    record = record_of(model)
    reference = calculators.ReferenceCalculator.from_calculator(mace_calculator(model), record, atoms)
    yardstick = calculators.ReferenceCalculator.from_calculator(mace_calculator(model, "float32"), record, atoms,
                                                                atomic_energies=exact)
    assert yardstick.info()["role"] == "yardstick" and yardstick.info()["float64_energy"] is True
    e64 = reference.evaluate(atoms)[0]
    energy, forces, stress = yardstick.evaluate(atoms)
    calc32 = mace_calculator(model, "float32")
    if family == "polar":
        # The stock float32 forces with the corrected long-range path; stress is MACE's defect there.
        from fastmace.longrange import polar

        polar.enable(calc32.models[0], atoms.cell.array, variable_cell=True)
    stock_energy, stock_forces, stock_stress = stock(calc32, atoms)
    np.testing.assert_array_equal(forces, stock_forces)
    np.testing.assert_array_equal(stress, stock_stress)
    assert abs(stock_energy - e64) > 1e-4  # the float32 E0 sum
    assert abs(energy - e64) < abs(stock_energy - e64) / 2
    if family == "polar":
        return  # the float32 charge solve differs with the edge order more than this check allows
    # The float64 sum of float32 terms: the same as the force field's float64 accumulation (1b-T03).
    from fastmace.config import EngineConfig, NeighbourConfig
    from fastmace.model.forces import MACEForceField

    ff = MACEForceField(mace_calculator(model, "float32"), atoms, atomic_energies=exact,
                        neighbours=NeighbourConfig(skin=0.5, sort_receivers="off", gpu_neighbour_list="off"),
                        engine=EngineConfig(compile_mode="none"))
    np.testing.assert_allclose(energy, ff.evaluate(atoms)[0], rtol=0, atol=1e-6 * len(atoms))


def test_yardstick_needs_the_record_entry():
    model, atoms = model64("mh1"), structure("mh1")
    record = record_of(model)
    without = dataclasses.replace(record, numerical=())
    with pytest.raises(records.ReferencePolicyError, match="float64"):
        calculators.ReferenceCalculator.from_calculator(mace_calculator(model, "float32"), without, atoms)


def test_reference_turns_on_the_one_graph_strain():
    from fastmace import _patches

    model, atoms = model64("mh1"), structure("mh1")
    reference = calculators.ReferenceCalculator.from_calculator(mace_calculator(model), record_of(model), atoms)
    assert _patches.record_of("single_graph_strain") is not None
    assert "single_graph_strain" in reference.info()["patches"]


def test_load_builds_the_reference_from_a_checkpoint(tmp_path):
    path = tmp_path / "tiny-mh1.model"
    torch.save(model64("mh1"), path)
    atoms = structure("mh1")
    reference = calculators.load(str(path), atoms, dtype="float64", device="cpu", trusted=True, family="mh1")
    yardstick = calculators.load(str(path), atoms, dtype="float32", device="cpu", trusted=True, family="mh1")
    info = reference.info()
    assert info["backend"] == "cueq" and info["cueq_conv_fusion"] is False
    assert info["record_id"] == records.record_for(records.mace_version(), "mh1").id
    e64 = reference.evaluate(atoms)[0]
    e32 = yardstick.evaluate(atoms)[0]
    # The checkpoint's exact E0 table reaches the yardstick: only the forward pass differs.
    assert abs(e32 - e64) < 1e-3
