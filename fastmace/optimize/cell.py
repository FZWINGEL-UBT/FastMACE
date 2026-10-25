# SPDX-License-Identifier: LGPL-2.1-or-later
# Provenance: port of ASE 3.27's ase/filters.py (FrechetCellFilter), LGPL-2.1-or-later; see NOTICE.md.
"""ASE 3.27's ``FrechetCellFilter`` as a device ``DOFView`` (SPEC §10.3; plan task 4-T07).

The degrees of freedom are ASE's (N + 3) x 3 rows: the positions in the original cell, ``u = r F^-T``, and the
cell rows ``ecf * logm(cf * F)``, where ``F = solve(orig_cell, cell).T`` is the deformation gradient, ``ecf``
the ``exp_cell_factor`` (default N) and ``cf = N`` the ``cell_factor`` ASE's ``UnitCellFilter`` folds in, so
the cell rows carry a constant ``ecf * ln(N) * I``. ``set_x`` applies ``F = expm(X / ecf) / cf``, masked as
ASE masks it; the gradient is ASE's ``-get_forces()``: ``f @ F`` on the atoms and the Fréchet derivative of
``expm`` against the virial on the cell, divided by ``ecf``. LBFGS and FIRE run on it unchanged.

Everything runs on the device without host syncs or LAPACK calls, so a step stays capturable:
- ``expm`` is ``integrators.baoab.expm_3x3``, applied to ``X / ecf - ln(cf) I``: the same matrix as ASE's
  ``expm(X / ecf) / cf``, but of norm ~|log F| instead of ~ln N, inside the Taylor range;
- without a mask, ``log F`` is that argument itself (ASE recomputes it with ``logm``, equal to rounding); with
  a mask, ``F`` is no longer an exponential of ``X``, and ``logm_3x3`` (inverse scaling and squaring) gives it;
- ASE's nine ``expm_frechet`` calls are one: the adjoint of ``L_exp(A, .)`` is ``L_exp(A^T, .)``, the upper
  right block of ``expm([[A^T, G], [0, A^T]])``;
- 3 x 3 inverses and determinants by cofactors (``neighbors.skin``).

The fixed iteration counts are exact in float64 for ``|log F| <= LOG_LIMIT`` per element (strains up to about
65 %). A move outside it is not applied: ``cell_ok`` drops, ``max_force`` becomes NaN, and the driver stops
with ``failure_reason``. ``FixAtoms`` with a cell filter is refused (owner decision D1).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch
from ase import units
from ase.stress import voigt_6_to_full_3x3_stress

from fastmace.integrators.baoab import expm_3x3
from fastmace.neighbors.skin import det_3x3, inverse_3x3, times_3x3
from fastmace.optimize.base import DOFView

LOG_LIMIT = 0.5          # largest |log F| element the matrix functions below are sized for
LOGM_ROOTS = 4           # square roots before the series: |F^(1/16) - I| <= 0.1 inside LOG_LIMIT
SQRTM_ITERATIONS = 8     # product-form Denman-Beavers steps per root (quadratic; 5 reach rounding from 0.65)
ATANH_TERMS = 9          # log Y = 2 atanh(T): T^17 / 17 < 1e-22 for |T| <= 0.05
LOGM_RESIDUAL = 1e-10    # largest |expm(logm F) - F| element accepted (masked filters only)
FRECHET_SCALE = 2.0 ** -10  # the direction block's size in the 6 x 6 exponential (exact power of two)


def _sqrtm_3x3(a: torch.Tensor, eye: torch.Tensor) -> torch.Tensor:
    """Principal square root (product-form Denman-Beavers, Higham 2008 eq. 6.17)."""
    m, y = a, a
    for _ in range(SQRTM_ITERATIONS):
        m_inverse = inverse_3x3(m)
        y = y @ (eye + m_inverse) / 2
        m = (eye + (m + m_inverse) / 2) / 2
    return y


def logm_3x3(f: torch.Tensor) -> torch.Tensor:
    """Principal matrix logarithm of a 3 x 3 tensor near I (inverse scaling and squaring); CUDA-graph safe."""
    eye = torch.eye(3, dtype=f.dtype, device=f.device)
    y = f
    for _ in range(LOGM_ROOTS):
        y = _sqrtm_3x3(y, eye)
    t = (y - eye) @ inverse_3x3(y + eye)  # Y - I and (Y + I)^-1 commute: both are functions of Y
    t2 = t @ t
    term, total = t, t
    for j in range(1, ATANH_TERMS):
        term = term @ t2
        total = total + term / (2 * j + 1)
    return total * 2.0 ** (LOGM_ROOTS + 1)


def expm_frechet_adjoint(a: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """``D[mu, nu] = sum(expm_frechet(a, E_mu_nu) * g)``, ASE's nine directions, as one 6 x 6 exponential.

    ``D = L_exp(a^T, g)``. ``g`` is scaled to a small direction (the block is linear in it), so the block's norm
    is ``a``'s and ``expm_3x3`` stays in its range; the scale is undone after.
    """
    scale = g.abs().max()
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    at = a.T
    block = torch.cat((torch.cat((at, g * (FRECHET_SCALE / scale)), 1), torch.cat((torch.zeros_like(at), at), 1)))
    return expm_3x3(block)[:3, 3:] * (scale / FRECHET_SCALE)


class FrechetCellFilter(DOFView):
    """ASE's ``FrechetCellFilter(atoms, mask, exp_cell_factor, hydrostatic_strain, constant_volume,
    scalar_pressure)`` over a ``SimulationState`` with stress; ``Class(state, config)`` (``registry.build_dof``).

    The reference cell is the state's cell at construction, as ASE's ``orig_cell``. ``refresh`` (after each force
    call and before ``start``) fills ``x``'s atom rows, the gradient and the energy ``E + p V``; ``set_x`` moves
    positions and cell and fills ``x``'s cell rows.
    """

    # Every tensor ``set_x`` and ``refresh`` write besides the state's positions and cell (``dof.<name>``).
    BUFFERS = ("x", "log_deform", "gradient", "energy", "cell_ok")

    def __init__(self, state, config):
        if state.free_mask is not None:
            raise ValueError("FixAtoms with a cell filter is refused: fixed atoms in a moving cell have no ASE "
                             "reference in FastMACE 1.0 (owner decision D1, SPEC §4.4)")
        if not all(state.pbc):
            raise ValueError("a cell filter needs periodic boundaries in all three directions (SPEC §4.4)")
        if state.stress is None:
            raise ValueError("a cell filter needs a state with stress (planner rule 10)")
        self.state = state
        natoms = state.natoms
        self.cell_factor = float(natoms)
        factor = config.cell_filter_parameter("exp_cell_factor")
        self.exp_cell_factor = float(natoms) if factor is None else float(factor)
        self.scalar_pressure = float(config.cell_filter_parameter("scalar_pressure_GPa", 0.0)) * units.GPa
        self.hydrostatic_strain = bool(config.cell_filter_parameter("hydrostatic_strain", False))
        self.constant_volume = bool(config.cell_filter_parameter("constant_volume", False))
        mask = config.cell_filter_parameter("mask")
        mask = np.ones(6) if mask is None else np.asarray(mask, dtype=float)
        if mask.shape == (6,):
            mask = voigt_6_to_full_3x3_stress(mask)
        elif mask.shape != (3, 3):
            raise ValueError("shape of mask should be (3,3) or (6,)")
        self.masked = bool((mask != 1.0).any())
        f64 = dict(dtype=torch.float64, device=state.positions.device)
        self._mask = torch.tensor(mask, **f64)
        self._eye = torch.eye(3, **f64)
        self._log_cf_eye = math.log(self.cell_factor) * self._eye
        self.orig_cell = state.cell.detach().clone()
        self._orig_inverse = torch.tensor(np.linalg.inv(self.orig_cell.cpu().numpy()), **f64)
        rows = natoms + 3
        self._x = torch.zeros(3 * rows, **f64)
        self._gradient = torch.zeros(3 * rows, **f64)
        self._energy = torch.zeros(1, **f64)
        self.log_deform = torch.zeros((3, 3), **f64)  # log F at the current cell
        self.cell_ok = torch.ones((), dtype=torch.bool, device=f64["device"])
        with torch.no_grad():
            self._x.view(rows, 3)[natoms:] = self.exp_cell_factor * (self.log_deform + self._log_cf_eye)
            self._atom_rows()

    # ------------------------------------------------------------------ DOFView

    @property
    def rows(self) -> int:
        return self.state.natoms + 3

    @property
    def variable_cell(self) -> bool:
        return True

    @property
    def x(self) -> torch.Tensor:
        return self._x

    def gradient(self) -> torch.Tensor:
        return self._gradient

    def energy(self) -> torch.Tensor:
        return self._energy

    def max_force(self) -> torch.Tensor:
        value = torch.linalg.vector_norm(self._gradient.view(self.rows, 3), dim=1).max()
        return torch.where(self.cell_ok, value, torch.full_like(value, float("nan")))

    def set_x(self, x: torch.Tensor) -> None:
        """ASE's ``set_positions``: ``F = expm(X / ecf) / cf``, masked, the cell ``orig_cell @ F^T`` and the
        positions ``u @ F^T``. A move to the current ``x`` (a converged step) or out of range changes nothing."""
        state, eye = self.state, self._eye
        natoms = state.natoms
        with torch.no_grad():
            new = x.view(-1, 3)
            current = self._x.view(-1, 3)
            log_new = new[natoms:] / self.exp_cell_factor - self._log_cf_eye
            transform = eye + (expm_3x3(log_new) - eye).T * self._mask  # I + deform
            if self.masked:
                log_deform = logm_3x3(transform.T)
                residual = (expm_3x3(log_deform) - transform.T).abs().max()
                ok = torch.isfinite(log_deform).all() & (residual <= LOGM_RESIDUAL)
                cell_rows = self.exp_cell_factor * (log_deform + self._log_cf_eye)
            else:
                log_deform = log_new
                ok = torch.isfinite(new).all()
                cell_rows = new[natoms:]
            ok = ok & (log_deform.abs().max() <= LOG_LIMIT)
            same = (x == self._x).all()
            apply = ok & ~same
            state.cell.copy_(torch.where(apply, self.orig_cell @ transform, state.cell))
            state.positions.copy_(torch.where(apply, times_3x3(new[:natoms], transform), state.positions))
            self.log_deform.copy_(torch.where(apply, log_deform, self.log_deform))
            current[natoms:] = torch.where(apply, cell_rows, current[natoms:])
            current[:natoms] = torch.where(apply, new[:natoms], current[:natoms])
            self.cell_ok.copy_(ok | same)

    def refresh(self) -> None:
        """ASE's ``get_positions`` (atom rows) and ``get_forces`` at the committed forces, stress and cell."""
        state, eye = self.state, self._eye
        natoms = state.natoms
        with torch.no_grad():
            deform, deform_inverse = self._atom_rows()
            volume = det_3x3(state.cell).abs()
            virial = -volume * (state.stress + self.scalar_pressure * eye)
            if self.hydrostatic_strain:
                virial = eye * (torch.trace(virial) / 3.0)
            if self.masked:
                virial = virial * self._mask
            cell_gradient = expm_frechet_adjoint(self.log_deform, virial @ deform_inverse.T)
            if self.constant_volume:
                cell_gradient = cell_gradient - eye * (torch.trace(cell_gradient) / 3.0)
            gradient = self._gradient.view(-1, 3)
            gradient[:natoms] = -times_3x3(state.forces, deform)
            gradient[natoms:] = -cell_gradient / self.exp_cell_factor
            self._energy.copy_(state.energy + self.scalar_pressure * volume)

    def buffers(self) -> Mapping[str, torch.Tensor]:
        return {"x": self._x, "log_deform": self.log_deform, "gradient": self._gradient, "energy": self._energy,
                "cell_ok": self.cell_ok}

    # ------------------------------------------------------------------ helpers

    def failure_reason(self) -> str:
        """Why ``max_force`` is NaN (host text for the driver's error)."""
        return (f"the FrechetCellFilter's step left the range of its device matrix functions: |log F| <= {LOG_LIMIT} "
                "per element (strains up to about 65 %), finite values"
                + (f", and logm within {LOGM_RESIDUAL:g} (masked filter)" if self.masked else "")
                + "; the move was not applied. Relax in stages from a closer cell, or restart the filter from the "
                "current cell (4-T07)")

    def _atom_rows(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``x``'s atom rows ``solve(F, r.T).T`` at the state's positions and cell; returns ``F`` and ``F^-1``."""
        state = self.state
        deform = (self._orig_inverse @ state.cell).T
        deform_inverse = inverse_3x3(deform)
        self._x.view(-1, 3)[:state.natoms] = times_3x3(state.positions, deform_inverse.T)
        return deform, deform_inverse


__all__ = ["ATANH_TERMS", "FRECHET_SCALE", "FrechetCellFilter", "LOGM_ROOTS", "LOG_LIMIT", "expm_frechet_adjoint",
           "logm_3x3"]
