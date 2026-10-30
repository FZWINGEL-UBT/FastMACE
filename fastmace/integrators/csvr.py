# SPDX-License-Identifier: LGPL-2.1-or-later
# Provenance: port of ASE 3.27's ase/md/bussi.py (Bussi), LGPL-2.1-or-later; see NOTICE.md.
"""ASE 3.27's ``Bussi`` (CSVR, NVT) as graph-capturable half steps on a SimulationState (SPEC §10.2; 4-T02).

Bussi, Donadio and Parrinello, J. Chem. Phys. 126, 014101 (2007), Eq. (A7), as ASE writes it: at the start
of each step the velocities are scaled by

    alpha = sqrt(e + c (G + R^2) + 2 R sqrt(e c)),   e = exp(-dt / taut),
    c = (1 - e) K_target / (K ndof),                 K_target = kT ndof / 2,

with K the current kinetic energy, R a standard normal and G = 2 standard_gamma((ndof - 1) / 2); then
one velocity Verlet step (``nve.py``). ``first_half`` does the scaling, the kick and the drift on the
device, ``second_half`` the second kick. The energy the thermostat transferred, sum (alpha^2 - 1) K, is a
device scalar; the conserved quantity is E_kin + E_pot minus it, as ASE's ``transferred_energy``.

The noise (conductor's choice 3) is ASE's stream: per step ``standard_normal`` then ``standard_gamma``, drawn
on the host ``noise_batch`` steps at a time by a loop, so the generator's order is ASE's per-step order,
into ``state.rng.batch`` ``(noise_batch, 2)`` with the cursor (SPEC §10.6). ``ndof`` follows D2
(``Integrator.degrees_of_freedom``); with ``dof="ase"`` it is ASE's ``get_number_of_degrees_of_freedom()``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from ase import units

from fastmace.config import IntegratorConfig

from .base import CONSERVED_KEY, Integrator
from .langevin import (
    allocate_noise,
    base_output_values,
    check_ensemble,
    fixed_rows,
    kinetic_energy,
    noise_steps,
    refill_noise,
)
from .nve import constrained_drift, drift, half_kick, remove_momentum

if TYPE_CHECKING:
    from fastmace.state import SimulationState


def zero_kinetic_energy_error():
    """CSVR's refusal of a structure at rest (as ASE's ``Bussi``): an ``InputError``, which no fallback retries."""
    from fastmace.errors import InputError

    return InputError("Initial kinetic energy is zero. Set the initial velocities before running CSVR "
                      "(as ASE's Bussi).",
                      why="CSVR rescales the velocities, and a structure at rest has none to rescale.",
                      next_step="Draw velocities first, for example ase.md.velocitydistribution."
                                "MaxwellBoltzmannDistribution(atoms, temperature_K=..., rng=...), or start "
                                "with a Langevin segment.")


def check_initial_motion(config: IntegratorConfig, atoms) -> None:
    """Refuse a CSVR run whose structure is at rest before anything is planned or compiled."""
    if config.scheme == "csvr" and math.isclose(float(atoms.get_kinetic_energy()), 0.0, rel_tol=0, abs_tol=1e-12):
        raise zero_kinetic_energy_error()


class CSVR(Integrator):
    """``ase.md.bussi.Bussi`` with the state on the device.

    ``config``: the timestep (fs), ``parameter("tau_fs")`` (ASE's ``taut``), ``fixcm`` (as ``nve.py``:
    momentum removed at the start and after each kick; the scaling keeps it zero), ``dof`` and
    ``noise_batch``. ``temperature_K`` is the starting target (default ``config.temperature_K``). Raises,
    as ASE, when the starting kinetic energy is zero. ``FixAtoms`` as ``nve.py`` (ASE's ``Bussi`` steps with
    ``VelocityVerlet``), with ``ndof`` 3(N - n_fixed).
    """

    def __init__(self, state: SimulationState, config: IntegratorConfig | None = None, *,
                 temperature_K: float | None = None):
        config = IntegratorConfig(scheme="csvr") if config is None else config
        check_ensemble(config, "nvt", "CSVR")
        self.config = config
        self.dt = config.timestep_fs * units.fs
        tau_fs = config.parameter("tau_fs")
        if tau_fs is None or not tau_fs > 0:
            raise ValueError(f"CSVR needs a positive tau_fs parameter, got {tau_fs!r}")
        self.taut = tau_fs * units.fs
        self.fix_com = config.fixcm
        self.noise_batch = noise_steps(config.noise_batch, 2)
        self.fixed_rows = fixed_rows(state)
        self.x_old = None if self.fixed_rows is None else torch.zeros_like(state.positions)
        if math.isclose(float(kinetic_energy(state)), 0.0, rel_tol=0, abs_tol=1e-12):
            raise zero_kinetic_energy_error()
        f64 = dict(dtype=torch.float64, device=state.positions.device)
        self._exp_term = math.exp(-self.dt / self.taut)
        self.noise = torch.zeros(2, **f64)
        self.transferred_energy = torch.zeros((), **f64)
        self.target_kinetic = torch.zeros((), **f64)
        self.ndof = self.degrees_of_freedom(state)
        allocate_noise(state, (self.noise_batch, 2))
        if temperature_K is None:
            temperature_K = 0.0 if config.temperature_K is None else config.temperature_K
        self.set_temperature(temperature_K)

    @property
    def variable_cell(self) -> bool:
        return False

    def set_temperature(self, temperature_K: float) -> None:
        """Fill the target kinetic energy ``kT ndof / 2`` (a device scalar; ASE's ``target_kinetic_energy``)."""
        self.temperature_K = float(temperature_K)
        self.target_kinetic.fill_(0.5 * (self.temperature_K * units.kB) * self.ndof)

    def _draw(self, generator: np.random.Generator, rows: int) -> np.ndarray:
        """``rows`` steps of ASE's draws, in its order: ``standard_normal`` then ``2 standard_gamma``."""
        draws = np.empty((rows, 2))
        shape = 0.5 * (self.ndof - 1)
        for row in range(rows):
            draws[row, 0] = generator.standard_normal()
            draws[row, 1] = 2.0 * generator.standard_gamma(shape)
        return draws

    def start(self, state: SimulationState) -> None:
        if self.fix_com:
            remove_momentum(state)

    def prepare_step(self, state: SimulationState) -> None:
        rng = state.rng
        if rng.cursor == self.noise_batch:
            refill_noise(rng, lambda steps: self._draw(rng.generator, steps))
        self.noise.copy_(rng.batch[rng.cursor])
        rng.cursor += 1

    def first_half(self, state: SimulationState) -> None:
        # Bussi.scale_velocities: Eq. (A7) with the kinetic energy at the start of the step.
        kinetic = 0.5 * (state.masses * state.velocities ** 2).sum()
        normal, sum_of_noises = self.noise[0], self.noise[1]
        energy_scaling = (1 - self._exp_term) * self.target_kinetic / kinetic / self.ndof
        alpha = torch.sqrt(self._exp_term + energy_scaling * (sum_of_noises + normal ** 2)
                           + 2 * normal * torch.sqrt(self._exp_term * energy_scaling))
        state.velocities.mul_(alpha)
        self.transferred_energy.add_((alpha ** 2 - 1.0) * kinetic)
        half_kick(state, self.dt)
        if self.fix_com:
            remove_momentum(state)
        if self.fixed_rows is None:
            drift(state, self.dt)
        else:
            constrained_drift(state, self.dt, self.fixed_rows, self.x_old)

    def second_half(self, state: SimulationState) -> None:
        half_kick(state, self.dt)
        if self.fix_com:
            remove_momentum(state)
        if self.fixed_rows is not None:
            state.velocities.masked_fill_(self.fixed_rows, 0.0)

    def mid_step_tensors(self, state: SimulationState) -> tuple[torch.Tensor, ...]:
        return (state.velocities,)

    def buffers(self, state: SimulationState) -> Mapping[str, torch.Tensor]:
        buffers = dict(positions=state.positions, velocities=state.velocities, noise=self.noise,
                       transferred_energy=self.transferred_energy)
        if self.x_old is not None:
            buffers["x_old"] = self.x_old
        return buffers

    def output_values(self, state: SimulationState) -> Mapping[str, torch.Tensor]:
        values = base_output_values(state)
        values[CONSERVED_KEY] = values["kinetic"] + values["energy"] - self.transferred_energy
        return values

    def state_dict(self) -> dict[str, Any]:
        """The transferred energy (restart format 4); the noise rows and cursor are ``state.rng``'s."""
        return {"transferred_energy": self.transferred_energy.detach().cpu().numpy().copy()}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if set(state_dict) != {"transferred_energy"}:
            raise ValueError(f"CSVR's state is the transferred energy, got {sorted(state_dict)}")
        value = np.asarray(state_dict["transferred_energy"], dtype=np.float64)
        if value.shape != ():
            raise ValueError(f"transferred_energy is a scalar, got shape {value.shape}")
        self.transferred_energy.copy_(torch.from_numpy(value))
