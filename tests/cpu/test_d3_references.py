"""The D3 references against each other (plan task 3b-T01; SPEC §11.2 "Correctness criterion").

``torch-dftd`` 0.5.1 (float64) and ``simple-dftd3`` (PyPI ``dftd3`` 1.6.0), both on an ``fm.D3``
record's numbers, on rattled and strained NaCl, NbCl5 and NaNbCl6. Their agreement here is the
yardstick 3b-T03's kernel is held to. Measured on 22 October 2026 (largest over the cases):
forces 2.6e-9 eV/Å (pbe; r2scan 4.2e-10), stress 1.5e-10 eV/Å³, energies 1.5e-7 eV, a relative
2.2e-8 that is nearly the same in every case. The tolerances are about ten times those. Skipped
without the packages (``pip install 'fastmace[d3-reference]'``).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch_dftd")
pytest.importorskip("dftd3")

from _support.structures import nacl, nanbcl6, nbcl5  # noqa: E402
from fastmace.longrange import d3  # noqa: E402
from fastmace.longrange.d3 import D3  # noqa: E402
from fastmace.reference import d3 as reference  # noqa: E402

ENERGY_REL = 2e-7
FORCES_EV_A = 2e-8
STRESS_EV_A3 = 2e-9

# A symmetric strain, applied with the atoms (scale_atoms), on top of the rattle.
STRAIN = np.array([[0.03, 0.01, -0.005], [0.01, -0.02, 0.008], [-0.005, 0.008, 0.015]])


def _strained(atoms):
    atoms.set_cell(np.asarray(atoms.cell) @ (np.eye(3) + STRAIN), scale_atoms=True)
    return atoms


CASES = {
    "NaCl": lambda: nacl(2, rattle=0.05, seed=1),
    "NaCl-strained": lambda: _strained(nacl(2, rattle=0.05, seed=2)),
    "NbCl5": lambda: nbcl5(1, rattle=0.05, seed=1),
    "NbCl5-strained": lambda: _strained(nbcl5(1, rattle=0.05, seed=2)),
    "NaNbCl6": lambda: nanbcl6(1, rattle=0.05, seed=1),
    "NaNbCl6-strained": lambda: _strained(nanbcl6(1, rattle=0.05, seed=2)),
}


def test_installed_tables_are_the_contracts():
    assert d3.REFERENCE_TABLES.endswith("sha256=" + reference.torch_dftd3_tables_sha256())


@pytest.mark.parametrize("functional", ["r2scan", "pbe"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_torch_dftd3_and_simple_dftd3_agree(case, functional):
    atoms = CASES[case]()
    record = D3(functional=functional)
    e1, f1, s1 = reference.torch_dftd3(atoms, record)
    e2, f2, s2 = reference.simple_dftd3(atoms, record)
    assert e1 < 0 and np.abs(f1).max() > 1e-4 and np.abs(s1).max() > 1e-4  # a real signal
    assert abs(e1 - e2) <= ENERGY_REL * abs(e1)
    np.testing.assert_allclose(f1, f2, rtol=0, atol=FORCES_EV_A)
    np.testing.assert_allclose(s1, s2, rtol=0, atol=STRESS_EV_A3)
    np.testing.assert_allclose(s1, s1.T, rtol=0, atol=1e-12)


def test_explicit_numbers_reach_both_references():
    # A set neither package has under a name: both must use the record's numbers.
    atoms = CASES["NbCl5"]()
    record = D3(functional="custom", s6=0.9, s8=1.1, a1=0.3, a2=4.0)
    named = reference.torch_dftd3(atoms, D3(functional="r2scan"))[0]
    e1 = reference.torch_dftd3(atoms, record)[0]
    e2 = reference.simple_dftd3(atoms, record)[0]
    assert abs(e1 - named) > 1e-3
    assert abs(e1 - e2) <= ENERGY_REL * abs(e1)


def test_stress_is_the_strain_derivative():
    # Central differences of the energy in the strain: ASE's stress sign (positive under tension).
    atoms = CASES["NaNbCl6"]()
    record = D3(functional="r2scan")
    stress = reference.simple_dftd3(atoms, record)[2]
    h, volume = 1e-5, atoms.get_volume()
    for i, j in [(0, 0), (1, 2), (2, 2)]:
        strain = np.zeros((3, 3))
        strain[i, j] = strain[j, i] = h if i == j else h / 2  # dE = V sigma_ij h either way
        energies = []
        for sign in (1, -1):
            strained = atoms.copy()
            strained.set_cell(np.asarray(atoms.cell) @ (np.eye(3) + sign * strain), scale_atoms=True)
            energies.append(reference.simple_dftd3(strained, record)[0])
        assert (energies[0] - energies[1]) / (2 * h) / volume == pytest.approx(stress[i, j], abs=1e-8)


def test_cutoffs_are_the_records():
    # Both references must use the record's cutoffs, not their defaults (simple-dftd3's own pair
    # cutoff is 60 bohr): at 20 bohr they agree with each other and differ from 40 bohr.
    atoms = CASES["NaCl"]()
    short = D3(functional="r2scan", pair_cutoff_bohr=20.0, cn_cutoff_bohr=20.0)
    full = D3(functional="r2scan")
    e_short = (reference.torch_dftd3(atoms, short)[0], reference.simple_dftd3(atoms, short)[0])
    e_full = reference.torch_dftd3(atoms, full)[0]
    assert abs(e_short[0] - e_short[1]) <= ENERGY_REL * abs(e_short[0])
    assert abs(e_short[0] - e_full) > 1e-3


def test_three_body_and_non_periodic_are_refused():
    atoms = CASES["NaCl"]()
    with pytest.raises(ValueError, match="three_body"):
        reference.simple_dftd3(atoms, D3(functional="r2scan", three_body=True))
    atoms.pbc = (True, True, False)
    with pytest.raises(ValueError, match="periodic"):
        reference.simple_dftd3(atoms, D3(functional="r2scan"))
