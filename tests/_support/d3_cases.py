"""D3 test cases (plan task 3b-T03): rattled and strained cells, a brute-force half list, references.

- ``CASES``: rattled (0.05 Å) and strained (±2 %, with shear) NaCl, NbCl5 and NaNbCl6;
- ``half_list(atoms, cutoff)``: every pair within ``cutoff`` from ASE's neighbour list, as the half
  list the D3 kernel takes (``i < j``, plus each self-image pair once). It stands in for 3b-T02's
  GPU list in these tests;
- ``torch_dftd(atoms, record)``: energy, forces and stress from ``torch-dftd`` 0.5.1 with the
  record's parameters (Becke–Johnson explicit), in float64 on the CPU.

NaNbCl6 is a small cubic test cell (Na at the origin, a NbCl6 octahedron at the centre), not a
real structure. It stands in until 3b-T01's fixtures land.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.neighborlist import primitive_neighbor_list

from _support.structures import nacl, nbcl5


def nanbcl6():
    a, d = 7.4, 2.35 / 7.4
    cl = [(0.5 + s * d * (k == 0), 0.5 + s * d * (k == 1), 0.5 + s * d * (k == 2))
          for k in range(3) for s in (1, -1)]
    return Atoms(["Na", "Nb"] + ["Cl"] * 6, cell=np.eye(3) * a, pbc=True,
                 scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)] + cl)


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
    """``(i, j, images)`` int64 arrays of every pair within ``cutoff`` Å, each undirected pair once."""
    i, j, shifts = primitive_neighbor_list("ijS", atoms.pbc, atoms.cell.array, atoms.positions, cutoff,
                                           self_interaction=False)
    first_nonzero = np.where(shifts != 0, shifts, 0)
    sign = np.sign(first_nonzero[np.arange(len(shifts)), np.argmax(shifts != 0, axis=1)])
    keep = (i < j) | ((i == j) & (sign > 0))
    return i[keep].astype(np.int64), j[keep].astype(np.int64), shifts[keep].astype(np.int64)


def torch_dftd(atoms, record):
    """``(energy eV, forces (N, 3) eV/Å, stress (3, 3) eV/Å³)`` from torch-dftd, float64, CPU."""
    import torch
    from ase.units import Bohr
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

    calc = TorchDFTD3Calculator(damping="bj", xc=record.functional, cutoff=record.pair_cutoff_bohr * Bohr,
                                cnthr=record.cn_cutoff_bohr * Bohr, abc=False, dtype=torch.float64, device="cpu")
    atoms = atoms.copy()
    atoms.calc = calc
    return (atoms.get_potential_energy(), atoms.get_forces(),
            atoms.get_stress(voigt=False))
