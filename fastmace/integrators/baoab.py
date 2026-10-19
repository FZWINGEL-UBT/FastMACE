"""ASE's ``LangevinBAOAB`` (NPT) as graph-capturable half steps on a SimulationState (SPEC §10.2).

ASE's LangevinBAOAB (ASE 3.27) integrates atoms with BAOAB Langevin dynamics (Leimkuhler and
Matthews 2013) and the cell with a Langevin-Hoover barostat (Quigley and Probert 2004), either
hydrostatic (cell shape fixed, volume free) or fully flexible. This is the research engine's
``GPULangevinBAOAB``: the same algorithm, coefficients and random-number draw order, with the
state on the device. ``prepare_step`` is its ``_load_noise``, ``first_half`` its ``_stage_a``
without ``set_cell`` and the neighbour check, ``start`` and the beginning of ``second_half`` its
``_update_force_eps``, and the rest of ``second_half`` the kicks at the end of its ``_stage_b``
(docs/design/interfaces.md).

Unlike ase.md.langevin.Langevin (the NVT integrator), LangevinBAOAB does not remove the
centre-of-mass motion; this port does not either.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from ase import units
from ase.stress import voigt_6_to_full_3x3_stress

from fastmace.config import IntegratorConfig
from fastmace.neighbors.skin import det_3x3, times_3x3

from .base import Integrator
from .langevin import allocate_noise, base_output_values, check_ensemble

if TYPE_CHECKING:
    from fastmace.state import SimulationState

EXPM_SCALING = 2   # exp(A) = exp(A / 2^s)^(2^s); with the Taylor degree below, exact in
EXPM_DEGREE = 14   # float64 for |A| <= 1, while a barostat half step moves |A| ~ 1e-4


def expm_3x3(a):
    """Matrix exponential of a 3 x 3 tensor (fixed scaling and squaring, Taylor); CUDA-graph safe."""
    a = a / 2 ** EXPM_SCALING
    eye = torch.eye(3, dtype=a.dtype, device=a.device)
    result, term = eye.clone(), eye
    for k in range(1, EXPM_DEGREE + 1):
        term = term @ a / k
        result = result + term
    for _ in range(EXPM_SCALING):
        result = result @ result
    return result


def barostat_mass(P_tau, natoms, P_mass=None, P_mass_factor=1.0):
    """LangevinBAOAB._set_barostat_mass: the given mass, or ASE's heuristic for a period of P_tau / 4."""
    if P_mass is not None:
        return float(P_mass)
    if not P_tau > 0:
        raise ValueError("Heuristic used for P_mass requires P_tau > 0")
    C = 1.66 * units.fs
    return ((P_tau / 4.0) / C * natoms ** (1.0 / 6.0)) ** 2 * P_mass_factor


def external_stress(externalstress, hydrostatic):
    """LangevinBAOAB._set_externalstress_hydrostatic: a scalar (hydrostatic) or a 3 x 3 matrix."""
    stress = np.asarray(externalstress, dtype=float)
    if stress.shape == ():
        stress = stress.reshape(-1)
    if hydrostatic:
        if stress.shape != (1,):
            raise ValueError(f"externalstress must be scalar when hydrostatic, got shape {stress.shape}")
        return np.asarray(stress[0])
    if stress.shape == (1,):
        return stress[0] * np.identity(3)
    if stress.shape == (3,):
        return np.diag(stress)
    if stress.shape == (6,):
        return voigt_6_to_full_3x3_stress(stress)
    if stress.shape == (3, 3):
        return stress
    raise ValueError(f"externalstress must be scalar, 3-vector, 6-vector (Voigt) or 3x3, got {stress.shape}")


class LangevinBAOAB(Integrator):
    """``ase.md.langevinbaoab.LangevinBAOAB`` (NPT, temperature set) with the state on the device.

    ``config`` maps as the research engine's ``workflow.baoab_settings``: ``T_tau = 1 / friction``,
    ``P_tau = barostat_tau_fs`` (default 20 ``T_tau``, ASE's), hydrostatic or full cell from
    ``cell_mode``, ASE's heuristic barostat mass times ``barostat_mass_factor``. The external
    stress is ``config.effective_external_stress_GPa()`` (``external_stress_GPa``, else
    ``-pressure_GPa``) in ASE units, and ``config.disable_cell_langevin`` turns the barostat's
    thermostat off, as ASE's options of the same names. The 1a keywords ``externalstress`` (ASE
    units: a scalar, 3-, 6- or 3 x 3 stress) and ``disable_cell_langevin`` replace the config's
    values when given; they stay for one release (``planner/fixed.py`` passes them).

    The barostat momentum and cell force are physical state (SPEC §10.6): they live in
    ``state.p_eps`` and ``state.force_eps``, which the constructor allocates unless the state
    already holds tensors of the right shape; ``self.p_eps`` and ``self.force_eps`` are the same
    tensors.

    Random numbers come from ``state.rng`` ``noise_batch`` steps at a time, in ASE's order: per step
    3N atom numbers, then 1 (hydrostatic) or 9 (full cell) barostat numbers (none when the cell
    Langevin is disabled). The state needs a stress tensor (``compute_stress``).

    Full cell: the drag of the barostat on the atom velocities is (p_eps + Tr(p_eps) / Ndof I) / W
    (Quigley and Probert 2004, Eq. 5b). ASE 3.27 writes ``p_eps + Tr(p_eps) / Ndof`` without the
    identity, so NumPy adds the trace term to all nine elements. ``ase_trace_drag_all_elements``
    reproduces ASE exactly (validation against ASE only); the difference is O(1 / Ndof) of the drag.
    """

    def __init__(self, state: SimulationState, config: IntegratorConfig | None = None, *,
                 temperature_K: float = 0.0, externalstress: Any = None, disable_cell_langevin: bool | None = None):
        config = IntegratorConfig(ensemble="npt") if config is None else config
        check_ensemble(config, "npt", "LangevinBAOAB")
        if state.stress is None:
            raise ValueError("NPT needs a state with stress (a force field with compute_stress=True)")
        self.config = config
        self.dt = config.timestep_fs * units.fs
        self.natoms = state.natoms
        self.hydrostatic = config.cell_mode == "hydrostatic"
        self.trace_drag_all_elements = config.ase_trace_drag_all_elements
        self.T_tau = 1.0 / config.friction
        # ASE's default with T_tau
        self.P_tau = 20.0 * self.T_tau if config.barostat_tau_fs is None else config.barostat_tau_fs * units.fs
        self.barostat_mass = barostat_mass(self.P_tau, self.natoms, None, config.barostat_mass_factor)
        self.Ndof = 3 * self.natoms  # Atoms.get_number_of_degrees_of_freedom() without constraints
        if disable_cell_langevin is None:
            disable_cell_langevin = config.disable_cell_langevin
        self.disable_cell_langevin = bool(disable_cell_langevin)
        if externalstress is None:
            externalstress = np.asarray(config.effective_external_stress_GPa()) * units.GPa
        f64 = dict(dtype=torch.float64, device=state.positions.device)

        shape = () if self.hydrostatic else (3, 3)
        for name in ("p_eps", "force_eps"):
            tensor = getattr(state, name)
            if (tensor is None or tuple(tensor.shape) != shape or tensor.dtype != torch.float64
                    or tensor.device != state.positions.device):
                setattr(state, name, torch.zeros(shape, **f64))
        self.p_eps = state.p_eps
        self.force_eps = state.force_eps
        self.externalstress = torch.tensor(external_stress(externalstress, self.hydrostatic), **f64)
        self.eye = torch.eye(3, **f64)

        # Constants of LangevinBAOAB.set_temperature, without sqrt(kT), which is a device scalar.
        dt = self.dt
        self.gamma = 1.0 / self.T_tau
        self.ou_unit = math.sqrt(1.0 - math.exp(-2.0 * self.gamma * dt))  # times sqrt(kT / m)
        self.barostat_gamma = 0.0 if self.disable_cell_langevin else 1.0 / self.P_tau
        self.barostat_ou_unit = (math.sqrt(1.0 - math.exp(-2.0 * self.barostat_gamma * dt))
                                 * math.sqrt(self.barostat_mass))  # times sqrt(kT)
        self.inv_sqrt_masses = 1.0 / torch.sqrt(state.masses)
        self.sqrt_kT = torch.zeros((), **f64)
        self.set_temperature(temperature_K)

        self.cell_numbers = 0 if self.barostat_gamma == 0 else (1 if self.hydrostatic else 9)
        self.noise_batch = config.noise_batch
        allocate_noise(state, (self.noise_batch, 3 * self.natoms + self.cell_numbers))
        self.xi = torch.zeros((self.natoms, 3), **f64)
        self.zeta = torch.zeros(max(self.cell_numbers, 1), **f64)

    @property
    def variable_cell(self) -> bool:
        return True

    def set_temperature(self, temperature_K: float) -> None:
        self.temperature_K = float(temperature_K)
        self.sqrt_kT.fill_(math.sqrt(units.kB * temperature_K))

    def prepare_step(self, state: SimulationState) -> None:
        rng = state.rng
        if rng.cursor == self.noise_batch:
            # rng.normal(size=(N, 3)) then rng.normal() or rng.normal(size=(3, 3)) per step, as ASE.
            draws = rng.generator.normal(size=(self.noise_batch, rng.batch.shape[1]))
            rng.batch.copy_(torch.from_numpy(draws))
            rng.cursor = 0
        row = rng.batch[rng.cursor]
        self.xi.copy_(row[:3 * self.natoms].view(self.natoms, 3))
        if self.cell_numbers:
            self.zeta.copy_(row[3 * self.natoms:])
        rng.cursor += 1

    def start(self, state: SimulationState) -> None:
        """LangevinBAOAB.__init__: the cell force of the starting forces and stress."""
        self._update_force_eps(state)

    def _kinetic_tensor(self, state):
        """sum_i m_i v_i v_i^T (LangevinBAOAB's kinetic stress contribution, times the volume)."""
        v = state.velocities
        mv = state.masses * v
        return torch.stack([(v[:, a:a + 1] * mv).sum(dim=0) for a in range(3)])

    def _update_force_eps(self, state):
        volume = det_3x3(state.cell)
        if self.hydrostatic:
            ke2 = (state.masses * state.velocities ** 2).sum()  # 2 KE
            tr_virial = -volume * torch.diagonal(state.stress).sum()
            X = (ke2 + tr_virial) / (3.0 * volume)
            self.force_eps.copy_(3.0 * volume * (X + self.externalstress) + (3.0 / self.Ndof) * ke2)
        else:
            kinetic = self._kinetic_tensor(state)
            X = (kinetic - volume * state.stress) / volume
            self.force_eps.copy_(volume * (X + self.externalstress)
                                 + torch.diagonal(kinetic).sum() / self.Ndof * self.eye)

    def _barostat_a(self, state):
        """Half step of the cell; positions scale with it (Atoms.set_cell(scale_atoms=True))."""
        cell, x = state.cell, state.positions
        half = 0.5 * self.dt / self.barostat_mass
        if self.hydrostatic:
            volume = det_3x3(cell)
            factor = (volume * torch.exp(half * 3.0 * self.p_eps) / volume) ** (1.0 / 3.0)
            cell.mul_(factor)
            x.mul_(factor)
        else:
            transform = expm_3x3(half * self.p_eps).T
            cell.copy_(cell @ transform)
            x.copy_(times_3x3(x, transform))

    def first_half(self, state: SimulationState) -> None:
        dt = self.dt
        x, v, masses = state.positions, state.velocities, state.masses
        self.p_eps.add_(0.5 * dt * self.force_eps)             # barostat B
        v.add_(0.5 * dt * state.forces / masses)               # B
        self._barostat_a(state)                                # barostat A
        x.add_(0.5 * dt * v)                                   # A
        # O: Langevin plus the Langevin-Hoover drag of the barostat velocity
        if self.hydrostatic:
            drag = (1.0 + 3.0 / self.Ndof) * self.p_eps / self.barostat_mass
            v.mul_(torch.exp(-(self.gamma + drag) * dt))
        else:
            trace_term = torch.diagonal(self.p_eps).sum() / self.Ndof
            if not self.trace_drag_all_elements:
                trace_term = trace_term * self.eye
            drag = (self.p_eps + trace_term) / self.barostat_mass
            v.copy_(times_3x3(v, expm_3x3(-(self.gamma * self.eye + drag) * dt).T))
        v.add_((self.ou_unit * self.sqrt_kT) * self.xi * self.inv_sqrt_masses)
        if self.barostat_gamma:                                # barostat O
            self.p_eps.mul_(math.exp(-self.barostat_gamma * dt))
            if self.hydrostatic:
                self.p_eps.add_((self.barostat_ou_unit * self.sqrt_kT) * self.zeta[0])
            else:
                random_force = self.zeta.view(3, 3)
                self.p_eps.add_((self.barostat_ou_unit * self.sqrt_kT) * ((random_force + random_force.T) / 2))
        x.add_(0.5 * dt * v)                                   # A
        self._barostat_a(state)                                # barostat A

    def second_half(self, state: SimulationState) -> None:
        self._update_force_eps(state)                          # with the velocities before the last B
        state.velocities.add_(0.5 * self.dt * state.forces / state.masses)   # B
        self.p_eps.add_(0.5 * self.dt * self.force_eps)        # barostat B

    def mid_step_tensors(self, state: SimulationState) -> tuple[torch.Tensor, ...]:
        return (state.velocities, self.p_eps)

    def buffers(self, state: SimulationState) -> Mapping[str, torch.Tensor]:
        return dict(positions=state.positions, velocities=state.velocities, cell=state.cell, p_eps=self.p_eps,
                    force_eps=self.force_eps, xi=self.xi, zeta=self.zeta)

    def output_values(self, state: SimulationState) -> Mapping[str, torch.Tensor]:
        values = base_output_values(state)
        values.update(cell=state.cell, stress=state.stress)
        return values

    def volume(self, state: SimulationState) -> torch.Tensor:
        return det_3x3(state.cell)

    def state_dict(self) -> dict[str, Any]:
        """Empty: the barostat momentum and cell force are in ``SimulationState`` since 1b (SPEC §10.6)."""
        return {}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Nothing, or a 1a ``state_dict`` (``p_eps``, ``force_eps``), copied into the state's tensors."""
        if not state_dict:
            return
        if set(state_dict) != {"p_eps", "force_eps"}:
            raise ValueError(f"LangevinBAOAB's 1a state is p_eps and force_eps, got {sorted(state_dict)}")
        for name in ("p_eps", "force_eps"):
            target = getattr(self, name)
            source = torch.from_numpy(np.array(state_dict[name], dtype=np.float64))  # 0-d stays 0-d
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError(f"{name} has shape {tuple(source.shape)}, this cell mode needs {tuple(target.shape)}")
            target.copy_(source)
