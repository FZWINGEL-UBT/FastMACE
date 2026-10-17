"""The shared test helpers: tiny models of every §4.3 family on the small cells, in float64."""

import numpy as np
import pytest
import torch
from _support.structures import STRUCTURES, nacl, nanbo3, nbcl5
from _support.tiny_models import FAMILIES, polar_available, tiny_family_calculator

# One species table for all structures, so each model sees every cell.
Z_VALUES = [8, 11, 17, 41]


def _family(name):
    marks = []
    if FAMILIES[name].get("polar"):
        marks.append(pytest.mark.skipif(not polar_available(), reason="PolarMACE needs graph_longrange"))
    return pytest.param(name, marks=marks)


ALL_FAMILIES = [_family(name) for name in FAMILIES]


def evaluate(calc, atoms, charge=0, spin=1):
    atoms = atoms.copy()
    atoms.info.update(charge=charge, spin=spin)  # read by the OMOL and POLAR families only
    atoms.calc = calc
    return atoms.get_potential_energy(), atoms.get_forces()


def test_structures():
    expected = {"NaCl": (64, "Cl32Na32"), "NaNbO3": (40, "Na8Nb8O24"), "NbCl5": (36, "Cl30Nb6")}
    for name, build in STRUCTURES.items():
        atoms = build()
        assert (len(atoms), atoms.get_chemical_formula()) == expected[name]
        assert atoms.pbc.all()
        assert atoms.get_volume() > 0
        distances = atoms.get_all_distances(mic=True)[np.triu_indices(len(atoms), 1)]
        assert distances.min() > 1.8, name


def test_structure_options():
    assert len(nacl(repeat=1)) == 8
    assert len(nanbo3(repeat=1)) == 5
    assert len(nbcl5(repeat=2)) == 8 * 36
    plain, rattled = nacl(), nacl(rattle=0.05, seed=3)
    assert not np.allclose(plain.positions, rattled.positions)
    np.testing.assert_array_equal(rattled.positions, nacl(rattle=0.05, seed=3).positions)


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_tiny_model_builds_and_is_finite(family):
    calc = tiny_family_calculator(family, Z_VALUES)
    for name, build in STRUCTURES.items():
        atoms = build(rattle=0.05, seed=1)
        energy, forces = evaluate(calc, atoms)
        assert np.isfinite(energy), (family, name)
        assert forces.shape == (len(atoms), 3)
        assert np.isfinite(forces).all(), (family, name)
        assert np.abs(forces).max() > 0, (family, name)


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_tiny_model_is_deterministic_in_float64(family):
    first = tiny_family_calculator(family, Z_VALUES)
    second = tiny_family_calculator(family, Z_VALUES)
    assert all(p.dtype == torch.float64 for p in first.models[0].parameters())
    for name, build in STRUCTURES.items():
        atoms = build(rattle=0.05, seed=2)
        e1, f1 = evaluate(first, atoms)
        e1b, f1b = evaluate(first, atoms)
        e2, f2 = evaluate(second, atoms)
        assert e1 == e1b and e1 == e2, (family, name)
        np.testing.assert_array_equal(f1, f1b)
        np.testing.assert_array_equal(f1, f2)


def test_omol_embedding_reads_total_charge():
    calc = tiny_family_calculator("omol", Z_VALUES)
    atoms = nacl(rattle=0.05, seed=4)
    neutral, _ = evaluate(calc, atoms, charge=0)
    charged, _ = evaluate(calc, atoms, charge=1)
    assert neutral != charged


def test_families_use_distinct_interaction_classes():
    def classes(family):
        model = tiny_family_calculator(family, Z_VALUES).models[0]
        return type(model).__name__, tuple(type(i).__name__ for i in model.interactions)

    assert classes("residual") == ("ScaleShiftMACE", ("RealAgnosticResidualInteractionBlock",) * 2)
    assert classes("density") == ("ScaleShiftMACE", ("RealAgnosticDensityInteractionBlock",
                                                     "RealAgnosticDensityResidualInteractionBlock"))
    assert classes("off") == ("ScaleShiftMACE", ("RealAgnosticInteractionBlock",
                                                 "RealAgnosticResidualInteractionBlock"))
    assert classes("mh1") == ("ScaleShiftMACE", ("RealAgnosticResidualNonLinearInteractionBlock",) * 2)
    omol = tiny_family_calculator("omol", Z_VALUES).models[0]
    assert set(omol.embedding_specs) == {"total_charge", "total_spin"}
    if polar_available():
        assert classes("polar")[0] == "PolarMACE"
