# SPDX-License-Identifier: LGPL-2.1-or-later
# Provenance: port of ASE 3.27's ase/md/langevinbaoab.py (LangevinBAOAB), LGPL-2.1-or-later; see NOTICE.md.
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

import dataclasses
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from ase import units
from ase.stress import voigt_6_to_full_3x3_stress

from fastmace.config import IntegratorConfig
from fastmace.config import external_stress as stress_components
from fastmace.neighbors.skin import det_3x3, matmul_small, times_3x3

from .base import Integrator
from .langevin import allocate_noise, base_output_values, check_ensemble

if TYPE_CHECKING:
    from fastmace.state import SimulationState

EXPM_SCALING = 2   # exp(A) = exp(A / 2^s)^(2^s); with the Taylor degree below, exact in
EXPM_DEGREE = 14   # float64 for |A| <= 1, while a barostat half step moves |A| ~ 1e-4


def expm_3x3(a):
    """Matrix exponential of a square tensor (fixed scaling and squaring, Taylor); CUDA-graph safe, no FP64 GEMM.

    Named for its 3 x 3 use here; ``optimize.cell`` also calls it on the 6 x 6 block of a Fréchet derivative.
    """
    a = a / 2 ** EXPM_SCALING
    eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    result, term = eye.clone(), eye
    for k in range(1, EXPM_DEGREE + 1):
        term = matmul_small(term, a) / k
        result = result + term
    for _ in range(EXPM_SCALING):
        result = matmul_small(result, result)
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
        self.externalstress_host = self.externalstress.detach().cpu().numpy().copy()
        # The target of the step in flight: stage A copies the filled target here and stage B reads it, so
        # a pipelined B(n), launched after the host filled step n+1's target, still uses step n's.
        self.externalstress_step = self.externalstress.clone()
        self.eye = torch.eye(3, **f64)

        self._friction_constants()
        self.inv_sqrt_masses = self._inv_sqrt_masses(state.masses)
        self.sqrt_kT = torch.zeros((), **f64)
        self.set_temperature(temperature_K)

        self.cell_numbers = 0 if self.barostat_gamma == 0 else (1 if self.hydrostatic else 9)
        self.noise_batch = config.noise_batch
        allocate_noise(state, (self.noise_batch, 3 * self.natoms + self.cell_numbers))
        self.xi = torch.zeros((self.natoms, 3), **f64)
        self.zeta = torch.zeros(max(self.cell_numbers, 1), **f64)

    def _friction_constants(self) -> None:
        """Every host float that follows the friction: ``T_tau``; with the default barostat τ, ``P_tau`` and
        the barostat mass (an explicit ``barostat_tau_fs`` stays); the constants of
        LangevinBAOAB.set_temperature without sqrt(kT), which is a device scalar. Baked into the stages."""
        config, dt = self.config, self.dt
        self.T_tau = 1.0 / config.friction
        # ASE's default with T_tau
        self.P_tau = 20.0 * self.T_tau if config.barostat_tau_fs is None else config.barostat_tau_fs * units.fs
        self.barostat_mass = barostat_mass(self.P_tau, self.natoms, None, config.barostat_mass_factor)
        self.gamma = 1.0 / self.T_tau
        self.ou_unit = math.sqrt(1.0 - math.exp(-2.0 * self.gamma * dt))  # times sqrt(kT / m)
        self.barostat_gamma = 0.0 if self.disable_cell_langevin else 1.0 / self.P_tau
        self.barostat_ou_unit = (math.sqrt(1.0 - math.exp(-2.0 * self.barostat_gamma * dt))
                                 * math.sqrt(self.barostat_mass))  # times sqrt(kT)

    @staticmethod
    def _inv_sqrt_masses(masses):
        return 1.0 / torch.sqrt(masses)

    def register_derived(self, registry: Any, state: SimulationState) -> None:
        """``inv_sqrt_masses`` (from the masses) and the cell force ``force_eps`` (from forces, stress, cell
        and momenta: refreshed in the resume, after the forces; SPEC §8.5 step 6)."""
        registry.register("integrator.inv_sqrt_masses", self.inv_sqrt_masses,
                          lambda: self.inv_sqrt_masses.copy_(self._inv_sqrt_masses(state.masses)),
                          depends_on=("masses",))
        registry.register("integrator.force_eps", self.force_eps, lambda: self._update_force_eps(state),
                          depends_on=("masses", "momenta", "forces", "stress", "cell"))

    def set_friction(self, friction: float) -> None:
        """A friction event's host part (inverse ASE time units): the config and every constant that follows it."""
        self.config = dataclasses.replace(self.config, friction=float(friction))
        self._friction_constants()

    @property
    def variable_cell(self) -> bool:
        return True

    def set_temperature(self, temperature_K: float) -> None:
        self.temperature_K = float(temperature_K)
        self.sqrt_kT.fill_(math.sqrt(units.kB * temperature_K))

    def set_pressure(self, pressure_GPa: float) -> None:
        """Fill a hydrostatic pressure target (GPa) for the next step: ``set_external_stress(-pressure_GPa)``.

        A device fill between steps (SPEC §10.1). The new target enters at the next step's cell-force
        update, so that step's first barostat half kick still uses the previous target: a half-step
        lag, as in ASE when ``externalstress`` is replaced between steps (``set_externalstress``).
        """
        if isinstance(pressure_GPa, bool) or not isinstance(pressure_GPa, (int, float)):
            raise ValueError(f"pressure_GPa must be a number of GPa, got {pressure_GPa!r}")
        self.set_external_stress(-pressure_GPa)

    def set_external_stress(self, stress_GPa: Any) -> None:
        """Fill the external stress target in GPa, ASE sign (``IntegratorConfig.external_stress_GPa``'s forms).

        A scalar, or for the full cell also six Voigt components (xx, yy, zz, yz, xz, xy). Same arithmetic
        as the config's target, so a run filled to P equals one configured with P bit for bit.
        """
        stress = stress_components(stress_GPa, "external_stress_GPa")
        if stress is None:
            raise ValueError("external_stress_GPa must be a number or six Voigt components, got None")
        if self.hydrostatic and not isinstance(stress, float):
            raise ValueError(f"a hydrostatic cell takes a scalar external stress, got {stress_GPa!r}")
        self.set_externalstress(np.asarray(stress) * units.GPa)

    def set_externalstress(self, externalstress: Any) -> None:
        """Fill the target in ASE units (the constructor's ``externalstress`` forms; restarts store this).

        A ``copy_`` into the tensor the half steps read (SPEC §10.1): captured graphs see the new target
        without a recapture. The next step's stage A copies it into ``externalstress_step``, which its
        cell-force update (stage B) reads: the end of the next step, as replacing ``externalstress``
        between ASE's steps does. That step's first barostat half kick still uses the cell force of the
        previous target. Outside a step (a restart, the engine's start) both take the value at once.
        """
        value = np.asarray(external_stress(externalstress, self.hydrostatic), dtype=np.float64)
        if not np.all(np.isfinite(value)):
            raise ValueError(f"externalstress must be finite, got {externalstress!r}")
        self.externalstress.copy_(torch.from_numpy(value))
        self.externalstress_host = value.copy()

    def restore_externalstress(self, externalstress: Any) -> None:
        """A restart's target: the filled target and the step's copy both (no step is in flight)."""
        self.set_externalstress(externalstress)
        self.externalstress_step.copy_(self.externalstress)

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
        self.externalstress_step.copy_(self.externalstress)
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
            self.force_eps.copy_(3.0 * volume * (X + self.externalstress_step) + (3.0 / self.Ndof) * ke2)
        else:
            kinetic = self._kinetic_tensor(state)
            X = (kinetic - volume * state.stress) / volume
            self.force_eps.copy_(volume * (X + self.externalstress_step)
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
            cell.copy_(matmul_small(cell, transform))
            x.copy_(times_3x3(x, transform))

    def first_half(self, state: SimulationState) -> None:
        dt = self.dt
        x, v, masses = state.positions, state.velocities, state.masses
        self.externalstress_step.copy_(self.externalstress)    # this step's pressure target
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
                    force_eps=self.force_eps, xi=self.xi, zeta=self.zeta, externalstress_step=self.externalstress_step)

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
