"""ASE's ``VelocityVerlet`` (NVE) as graph-capturable half steps on a SimulationState (SPEC §10.2; plan task 4-T02).

``first_half`` is ``VelocityVerlet.step`` up to the force call (half kick, drift), ``second_half`` the kick
after it. ASE works in momenta, this port in ``state.velocities``; the operations are ASE's, so a float64
run follows ASE's to rounding. ``fixcm`` (owner decision D2, recommended (a)), which ASE's class lacks:
``start`` removes the total momentum, and each half step projects it out again after its kick, so float
drift of the force sum never accumulates. CSVR and Berendsen build on the helpers here.

``FixAtoms`` (plan task 4-T04; ``fixcm`` is refused with it): ASE's step with a constraint. The fixed atoms'
forces are zero (``get_forces(md=True)``), so after the first kick their velocities are zero and the drift
leaves them where they are; then every velocity is recomputed from the displacement,
``p = (r_new - r) m / dt``, as ``VelocityVerlet.step`` does whenever ``atoms.constraints`` is not empty; after
the second kick the fixed velocities are zero again (``set_momenta``). Without ``FixAtoms`` none of this runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from ase import units

from fastmace.config import IntegratorConfig

from .base import CONSERVED_KEY, Integrator
from .langevin import base_output_values, check_ensemble, fixed_rows, kinetic_energy

if TYPE_CHECKING:
    from fastmace.state import SimulationState


def remove_momentum(state: SimulationState) -> None:
    """Subtract the centre-of-mass velocity: the total momentum becomes zero (to rounding)."""
    v, masses = state.velocities, state.masses
    v.sub_((masses * v).sum(dim=0) / masses.sum())


def half_kick(state: SimulationState, dt: float) -> None:
    """``p += 0.5 dt F`` as velocities."""
    state.velocities.add_(0.5 * dt * state.forces / state.masses)


def drift(state: SimulationState, dt: float) -> None:
    """``r += dt p / m`` as velocities."""
    state.positions.add_(dt * state.velocities)


def constrained_drift(state: SimulationState, dt: float, fixed: torch.Tensor, x_old: torch.Tensor) -> None:
    """``FixAtoms``' drift, as ASE's ``VelocityVerlet`` with a constraint: the fixed velocities zero (their forces
    are zero in ASE), ``r += dt p / m``, then ``p = (r_new - r) m / dt`` for every atom."""
    state.velocities.masked_fill_(fixed, 0.0)
    x_old.copy_(state.positions)
    drift(state, dt)
    state.velocities.copy_((state.positions - x_old) / dt)


class VelocityVerlet(Integrator):
    """``ase.md.verlet.VelocityVerlet`` with the state on the device.

    ``config``: the timestep (fs) and ``fixcm``. No thermostat: ``set_temperature`` raises, and the
    keyword ``temperature_K`` (the planner passes the plan's) is ignored. The conserved quantity is
    E_kin + E_pot. No noise and no state of its own.
    """

    def __init__(self, state: SimulationState, config: IntegratorConfig | None = None, *,
                 temperature_K: float | None = None):
        config = IntegratorConfig(ensemble="nve") if config is None else config
        check_ensemble(config, "nve", "VelocityVerlet")
        self.config = config
        self.dt = config.timestep_fs * units.fs
        self.fix_com = config.fixcm
        self.temperature_K = None  # no target (the restart's integrator record stores none)
        self.fixed_rows = fixed_rows(state)
        self.x_old = None if self.fixed_rows is None else torch.zeros_like(state.positions)

    @property
    def variable_cell(self) -> bool:
        return False

    def set_temperature(self, temperature_K: float) -> None:
        raise NotImplementedError("VelocityVerlet (NVE) has no thermostat: it has no temperature target")

    def start(self, state: SimulationState) -> None:
        """``fixcm``: remove the starting total momentum (D2); a zero-momentum start is unchanged to rounding."""
        if self.fix_com:
            remove_momentum(state)

    def prepare_step(self, state: SimulationState) -> None:
        """Nothing: no noise."""

    def first_half(self, state: SimulationState) -> None:
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
        buffers = dict(positions=state.positions, velocities=state.velocities)
        if self.x_old is not None:
            buffers["x_old"] = self.x_old
        return buffers

    def output_values(self, state: SimulationState) -> Mapping[str, torch.Tensor]:
        values = base_output_values(state)
        values[CONSERVED_KEY] = values["kinetic"] + values["energy"]
        return values

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if state_dict:
            raise ValueError(f"VelocityVerlet has no internal state, got {sorted(state_dict)}")


__all__ = ("VelocityVerlet", "constrained_drift", "drift", "half_kick", "kinetic_energy", "remove_momentum")
