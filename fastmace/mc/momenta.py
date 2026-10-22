"""Momenta of atoms whose mass changes in an MC move (SPEC §10.4 "Momenta of atoms whose mass changes").

The engines store velocities v (p = m v), in ASE units (masses in amu, energies in eV). Each per-move
treatment maps the velocities of the changed atoms before the move to their velocities after an
accepted move:

    treatment(velocities [k, 3], masses_old [k, 1], masses_new [k, 1], normals [k, 3], kT [1]) -> [k, 3]

``normals`` are the attempt's velocity-stream numbers (used only by ``redraw``) and ``kT`` the block's
k_B T in eV. The attempt writes the result only where the move is accepted. ``redraw_all`` acts on all
atoms once per block after an acceptance. The driver looks the functions up by name in this module
when it builds an attempt, so a test can monkeypatch one (3-T12's negative control).
"""

from __future__ import annotations

import torch


def exchange(velocities, masses_old, masses_new, normals, kT):
    """Swaps only: the two atoms exchange velocities with their species, v_i' = v_j, v_j' = v_i, so
    p_i' = p_j. Kinetic energy and total momentum are unchanged."""
    return velocities.flip(0)


def rescale(velocities, masses_old, masses_new, normals, kT):
    """Keep each atom's kinetic energy: p' = p sqrt(m_new / m_old), that is v' = v sqrt(m_old / m_new)."""
    return velocities * torch.sqrt(masses_old / masses_new)


def redraw(velocities, masses_old, masses_new, normals, kT):
    """Maxwell–Boltzmann at kT with the new masses: v' = sqrt(kT / m_new) z."""
    return torch.sqrt(kT / masses_new) * normals


def redraw_all(masses, normals, kT):
    """All velocities from Maxwell–Boltzmann at kT, projected to zero total momentum:
    v_i = sqrt(kT / m_i) z_i, then v_i <- v_i - (sum_j m_j v_j) / M. ``masses`` ``[N, 1]``, ``normals`` ``[N, 3]``."""
    velocities = torch.sqrt(kT / masses) * normals
    return velocities - (masses * velocities).sum(dim=0, keepdim=True) / masses.sum()


# The per-move treatments by name (config.SWAP_VELOCITIES, TRANSMUTE_VELOCITIES); redraw_all is per block.
PER_MOVE = ("exchange", "rescale", "redraw")

__all__ = ["PER_MOVE", "exchange", "redraw", "redraw_all", "rescale"]
