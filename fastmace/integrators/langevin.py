# SPDX-License-Identifier: LGPL-2.1-or-later
# Provenance: port of ASE 3.27's ase/md/langevin.py (Langevin), LGPL-2.1-or-later; see NOTICE.md.
"""ASE's ``Langevin`` (NVT, fixed cell) as graph-capturable half steps on a SimulationState (SPEC §10.2).

The maths of the research engine's ``GPULangevin``, which reproduces ``ase.md.langevin.Langevin``
step for step with ASE's random numbers. ``prepare_step`` is its ``_load_noise``, ``first_half`` its
``_stage_a`` without the neighbour check, and ``second_half`` the kick at the end of its ``_stage_b``
(docs/design/interfaces.md). The operations and their order are the baseline's, so an eager step
is bitwise the baseline's and the captured graphs hold the same kernels.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from ase import units

from fastmace.config import IntegratorConfig

from .base import Integrator

if TYPE_CHECKING:
    from fastmace.state import RandomStream, SimulationState


# The drawn-ahead noise batch holds at most this many bytes of float64 (CAP-T06), so its device memory stops growing
# with the atom count: 256 Langevin steps up to 5,461 atoms, 27 steps at 50,000 atoms (0.57 GiB at 256). The numbers
# and their order do not depend on the batch size.
NOISE_BATCH_BYTES = 64 * 2**20


def noise_steps(limit: int, numbers: int) -> int:
    """Steps per drawn-ahead batch: at most ``limit`` (``IntegratorConfig.noise_batch``) and at most
    ``NOISE_BATCH_BYTES`` of float64 for ``numbers`` per step, but at least one."""
    return max(1, min(limit, NOISE_BATCH_BYTES // (8 * numbers)))


def allocate_noise(state: SimulationState, shape: tuple[int, ...]) -> None:
    """Give ``state.rng`` a drawn-ahead batch of ``shape`` (marked used up, so the first step refills it).

    A batch already there (a restart loaded before the integrator was built) is kept with its cursor
    if it has the integrator's shape. One of another length with the same layout (a restart written with
    another batch size) gives its unused rows to ``rng.queued``, which the next refills use first.
    """
    rng = state.rng
    if rng.batch is not None:
        if tuple(rng.batch.shape) == shape:
            return
        if tuple(rng.batch.shape[1:]) != shape[1:]:
            raise ValueError(f"the state's noise batch has shape {tuple(rng.batch.shape)}, the integrator "
                             f"draws {shape}")
        rng.queue(rng.batch[rng.cursor:].cpu().numpy())
    rng.batch = torch.zeros(shape, dtype=torch.float64, device=state.positions.device)
    rng.cursor = shape[0]


def refill_noise(rng: RandomStream, draw: Callable[[int], np.ndarray]) -> None:
    """Refill the used-up ``rng.batch``: the queued rows first (at its end, so the cursor runs into them), else
    ``draw(steps)`` new steps from the generator, in its order."""
    size = rng.batch.shape[0]
    if rng.queued is None:
        rng.batch.copy_(torch.from_numpy(draw(size)))
        rng.cursor = 0
        return
    rows, rest = rng.queued[:size], rng.queued[size:]
    rng.queued = rest if len(rest) else None
    rng.cursor = size - len(rows)
    rng.batch[rng.cursor:].copy_(torch.from_numpy(np.ascontiguousarray(rows)))


def fixed_rows(state: SimulationState) -> torch.Tensor | None:
    """``(N, 1)`` bool, True for the ``FixAtoms`` atoms (from ``state.free_mask``), or ``None`` without them.

    The integrators' mask path (plan task 4-T04, ASE's ``FixAtoms``): the fixed rows of the positions are reset
    to their values before the drift (``adjust_positions``) and the fixed rows of the velocities set to zero
    after the last kick (``adjust_momenta``), both with ``where``/``masked_fill``, so free atoms keep their bits.
    Without ``FixAtoms`` no step does anything extra."""
    return None if state.free_mask is None else state.free_mask == 0


def kinetic_energy(state: SimulationState) -> torch.Tensor:
    return (0.5 * (state.masses * state.velocities ** 2).sum()).reshape(1)


def base_output_values(state: SimulationState) -> dict[str, torch.Tensor]:
    """GPULangevin._snapshot_values: positions, momenta, forces, energy, kinetic energy."""
    return dict(positions=state.positions, momenta=state.velocities * state.masses, forces=state.forces,
                energy=state.energy.reshape(-1)[:1], kinetic=kinetic_energy(state))


def check_ensemble(config: IntegratorConfig, ensemble: str, name: str) -> None:
    if config.ensemble != ensemble:
        raise ValueError(f"{name} integrates the {ensemble!r} ensemble, the config asks for {config.ensemble!r}")


class Langevin(Integrator):
    """``ase.md.langevin.Langevin`` with the state on the device.

    ``config`` gives the timestep (fs), the friction (inverse ASE time units), ``fixcm``, its
    ``fixcm_projection`` and ``noise_batch``. ``fixcm_projection="ase"`` is ASE's projection bit for bit;
    ``"mass_weighted"`` removes the mass-weighted mean of both random terms without ASE's scale factor,
    which is canonical at fixed centre of mass for unequal masses too (docs/design/fixcm-projection.md).
    ``FixAtoms`` (``fixed_rows``; ``fixcm`` is refused with it): the fixed atoms go back to their positions
    after the drift, as ASE's ``set_positions``, so their mid-step velocity is ``-rnd_pos / dt`` as in ASE, and
    their velocities are zero after the second kick, as ASE's ``set_momenta``. The noise is drawn for every atom.

    Random numbers come from ``state.rng`` ``noise_batch`` steps at a time, in ASE's order (per step
    ``xi``, then ``eta``, each ``(N, 3)``): a ``(k, 2, N, 3)`` draw gives the same numbers as k pairs of
    ``(N, 3)`` draws. The generator therefore ends up further advanced than
    ASE's, but every step uses ASE's numbers.
    """

    def __init__(self, state: SimulationState, config: IntegratorConfig | None = None, *,
                 temperature_K: float = 0.0):
        config = IntegratorConfig() if config is None else config
        check_ensemble(config, "nvt", "Langevin")
        self.config = config
        self.dt = config.timestep_fs * units.fs
        self.fix_com = config.fixcm
        self.mass_weighted_com = config.fixcm_projection == "mass_weighted"
        self.natoms = state.natoms
        self.noise_batch = noise_steps(config.noise_batch, 6 * self.natoms)
        masses = state.masses
        self.x_old = torch.zeros_like(state.positions)
        self.xi = torch.zeros_like(state.positions)
        self.eta = torch.zeros_like(state.positions)
        self.rnd_pos = torch.zeros_like(state.positions)
        self.rnd_vel = torch.zeros_like(state.positions)
        self.fixed_rows = fixed_rows(state)
        allocate_noise(state, (self.noise_batch, 2, self.natoms, 3))

        # ASE coefficients (Langevin.updatevars). c1, c2 do not depend on T;
        # c3, c4, c5 are proportional to sigma = sqrt(2 kT fr / m), so they are
        # stored for kT = 1 eV and scaled by sqrt(kT) on the device.
        self._friction_constants()
        self.c3_unit, self.c4_unit, self.c5_unit = self._noise_coefficients(masses)
        self.sqrt_kT = torch.zeros((), dtype=torch.float64, device=state.positions.device)
        self.set_temperature(temperature_K)

    def _friction_constants(self) -> None:
        """``fr``, ``c1`` and ``c2`` from the config: host floats, baked into the captured stages."""
        dt, fr = self.dt, self.config.friction
        self.fr = fr
        self.c1 = dt / 2.0 - dt * dt * fr / 8.0
        self.c2 = dt * fr / 2 - dt * dt * fr * fr / 8.0

    def _noise_coefficients(self, masses):
        """``c3_unit``, ``c4_unit``, ``c5_unit`` from the masses and the friction: new tensors, the same
        expressions at construction and in a refresh, so both give the same bits."""
        dt, fr = self.dt, self.fr
        sigma_unit = torch.sqrt(2 * fr / masses)
        c3 = math.sqrt(dt) * sigma_unit / 2.0 - dt ** 1.5 * fr * sigma_unit / 8.0
        c5 = dt ** 1.5 * sigma_unit / (2 * math.sqrt(3))
        return c3, fr / 2.0 * c5, c5

    def register_derived(self, registry: Any, state: SimulationState) -> None:
        """The mass- and friction-dependent noise coefficients (SPEC §8.5), refreshed in place."""
        for index, name in enumerate(("c3_unit", "c4_unit", "c5_unit")):
            buffer = getattr(self, name)

            def refresh(buffer=buffer, index=index):
                buffer.copy_(self._noise_coefficients(state.masses)[index])

            registry.register(f"integrator.{name}", buffer, refresh, depends_on=("masses", "friction"))

    def set_friction(self, friction: float) -> None:
        """A friction event's host part (inverse ASE time units): the config, ``fr``, ``c1`` and ``c2``. The
        derived-state registry refreshes the noise coefficients; the captured stages are dropped."""
        self.config = dataclasses.replace(self.config, friction=float(friction))
        self._friction_constants()

    @property
    def variable_cell(self) -> bool:
        return False

    def set_temperature(self, temperature_K: float) -> None:
        self.temperature_K = float(temperature_K)
        self.sqrt_kT.fill_(math.sqrt(units.kB * temperature_K))

    def prepare_step(self, state: SimulationState) -> None:
        rng = state.rng
        if rng.cursor == self.noise_batch:
            refill_noise(rng, lambda steps: rng.generator.standard_normal(size=(steps, 2, self.natoms, 3)))
        self.xi.copy_(rng.batch[rng.cursor, 0])
        self.eta.copy_(rng.batch[rng.cursor, 1])
        rng.cursor += 1

    def first_half(self, state: SimulationState) -> None:
        x, v, masses = state.positions, state.velocities, state.masses
        sqrt_kT = self.sqrt_kT
        rnd_pos = (self.c5_unit * sqrt_kT) * self.eta
        rnd_vel = (self.c3_unit * sqrt_kT) * self.xi - (self.c4_unit * sqrt_kT) * self.eta
        if self.fix_com and self.mass_weighted_com:
            # The noise conditioned on a fixed centre of mass: both terms have covariance proportional to
            # 1/m_i per atom, so the conditional mean is the same mass-weighted mean for every atom.
            total = masses.sum()
            rnd_pos = rnd_pos - (rnd_pos * masses).sum(dim=0) / total
            rnd_vel = rnd_vel - (rnd_vel * masses).sum(dim=0) / total
        elif self.fix_com:
            n = self.natoms
            factor = math.sqrt(n / (n - 1.0))
            rnd_pos = (rnd_pos - rnd_pos.sum(dim=0) / n) * factor
            rnd_vel = (rnd_vel - (rnd_vel * masses).sum(dim=0) / (masses * n)) * factor
        self.rnd_pos.copy_(rnd_pos)
        self.rnd_vel.copy_(rnd_vel)
        # First half step in the velocity, full step in the positions.
        v.add_(self.c1 * state.forces / masses - self.c2 * v + self.rnd_vel)
        self.x_old.copy_(x)
        x.add_(self.dt * v + self.rnd_pos)
        if self.fixed_rows is not None:
            x.copy_(torch.where(self.fixed_rows, self.x_old, x))
        v.copy_((x - self.x_old - self.rnd_pos) / self.dt)

    def second_half(self, state: SimulationState) -> None:
        v = state.velocities
        v.add_(self.c1 * state.forces / state.masses - self.c2 * v + self.rnd_vel)
        if self.fixed_rows is not None:
            v.masked_fill_(self.fixed_rows, 0.0)

    def mid_step_tensors(self, state: SimulationState) -> tuple[torch.Tensor, ...]:
        return (state.velocities,)

    def buffers(self, state: SimulationState) -> Mapping[str, torch.Tensor]:
        return dict(positions=state.positions, velocities=state.velocities, x_old=self.x_old,
                    rnd_pos=self.rnd_pos, rnd_vel=self.rnd_vel, xi=self.xi, eta=self.eta)

    def output_values(self, state: SimulationState) -> Mapping[str, torch.Tensor]:
        return base_output_values(state)

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if state_dict:
            raise ValueError(f"Langevin has no internal state, got {sorted(state_dict)}")
