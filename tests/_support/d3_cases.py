"""D3 kernel test cases (plan task 3b-T03): rattled and strained cells, a brute-force half list, tolerances.

- ``CASES``: rattled (0.05 Å) and strained (about ±2 %, with shear) NaCl, NbCl5 and NaNbCl6, every
  cell narrower than twice the 21.2 Å cutoff;
- ``half_list(atoms, cutoff)``: every pair within ``cutoff`` from ASE's neighbour list, as the half
  list the D3 kernel takes (``i < j``, plus each self-image pair once), with its image index and
  table. It stands in for 3b-T02's GPU list in these tests;
- ``within(deviation, agreement, floor)``: the runbook's item 1 rule (docs/validation/phase-3b-runbook.md):
  at most twice the two references' deviation from each other, plus a rounding floor.
"""

from __future__ import annotations

import numpy as np
from ase.neighborlist import primitive_neighbor_list

from _support.structures import nacl, nanbcl6, nbcl5

# Rounding floors of the runbook, item 1: energy per atom eV, max force component eV/Å, max stress eV/Å³
FLOORS = (1e-10, 1e-8, 1e-9)
# The gate's absolute floors (SPEC §12.3), for float32 against float64: the same three quantities
GATE = (1e-5, 1e-5, 1e-7)


def strained(atoms, sign):
    strain = np.eye(3) + sign * np.array([[0.02, 0.005, 0.0], [0.005, -0.01, 0.004], [0.0, 0.004, 0.015]])
    atoms = atoms.copy()
    atoms.set_cell(atoms.cell.array @ strain, scale_atoms=True)
    return atoms


def _rattled(atoms, seed):
    atoms = atoms.copy()
    atoms.rattle(0.05, seed=seed)
    return atoms


CASES = {
    "NaCl-rattled": lambda: _rattled(nacl(repeat=1), 1),
    "NaCl-strained": lambda: _rattled(strained(nacl(repeat=1), +1), 2),
    "NbCl5-rattled": lambda: _rattled(nbcl5(), 3),
    "NbCl5-strained": lambda: _rattled(strained(nbcl5(), -1), 4),
    "NaNbCl6-rattled": lambda: _rattled(nanbcl6(), 5),
    "NaNbCl6-strained": lambda: _rattled(strained(nanbcl6(), +1), 6),
}


def half_list(atoms, cutoff):
    """``(i, j, image_index, image_table)``: every pair within ``cutoff`` Å once; the table float64, as 3b-T02's."""
    i, j, shifts = primitive_neighbor_list("ijS", atoms.pbc, atoms.cell.array, atoms.positions, cutoff,
                                           self_interaction=False)
    first = shifts[np.arange(len(shifts)), np.argmax(shifts != 0, axis=1)]
    keep = (i < j) | ((i == j) & (first > 0))
    table, index = np.unique(shifts[keep], axis=0, return_inverse=True)
    return (i[keep].astype(np.int64), j[keep].astype(np.int64), index.reshape(-1).astype(np.int32),
            table.astype(np.float64))


def deviations(ours, reference, n_atoms):
    """``(energy per atom, max force component, max stress component)`` deviations of two results."""
    return (abs(ours[0] - reference[0]) / n_atoms, float(np.abs(ours[1] - reference[1]).max()),
            float(np.abs(ours[2] - reference[2]).max()))


def within(deviation, agreement, floors=FLOORS):
    return all(d <= 2.0 * a + f for d, a, f in zip(deviation, agreement, floors))
