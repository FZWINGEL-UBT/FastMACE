# SPDX-License-Identifier: LGPL-2.1-or-later
# Provenance: port of ASE 3.27's ase/md/nvtberendsen.py and nptberendsen.py, LGPL-2.1-or-later; see NOTICE.md.
"""ASE 3.27's Berendsen classes as graph-capturable half steps on a SimulationState (SPEC §10.2; 4-T02).

Berendsen et al., J. Chem. Phys. 81, 3684 (1984). Non-canonical: for equilibration only (the scheme record
says so; the planner warns and the report shows the line).

- ``NVTBerendsen`` (no pressure): at the start of each step the velocities are scaled by
  ``lambda = sqrt(1 + (T0 / T - 1) dt / taut)``, clamped to [0.9, 1.1] as ASE does, with T the kinetic
  temperature over ``ndof`` (D2; ``dof="ase"`` is ASE's ``get_temperature``), then one velocity Verlet
  step. ``fixcm`` is ASE's: after the first kick the mean momentum ``sum(p) / N`` is subtracted.
  ``FixAtoms`` (4-T04, NVT only; ``fixcm`` is refused with it): ASE's fixed atoms have zero forces and momenta,
  so their velocities are set to zero after each kick and the drift leaves them in place. ASE's
  ``NVTBerendsen`` does not recompute the momenta from the displacement, and neither does this port.
- ``NPTBerendsen`` (``pressure_GPa``, cell mode ``"hydrostatic"``): after the velocity scaling the cell and
  the positions are scaled by ``1 - dt kappa / (3 taup) (P0 - P)``, P from the stress including the
  ideal-gas term of the scaled velocities (``get_stress(include_ideal_gas=True)``).
- ``Inhomogeneous_NPTBerendsen`` (``mask``, cell mode ``"full"``): each cell vector i is scaled by
  ``1 - dt kappa / (3 taup) (P0 - sigma_ii) pbc_i mask_i`` with sigma = -stress; the angles stay.

**Deviation from ASE (conductor, 23 October 2026):** ASE's NPT step evaluates the forces twice: the cell
scaling invalidates its cache, so the first kick uses forces at the scaled geometry. This port makes one
force call per step, as every FastMACE integrator: the first kick uses the forces of the previous step, at
the unscaled cell. The difference is of order ``dt kappa |P0 - P| / taup`` in the forces of one half kick;
the CPU tests compare it to ASE's class reordered the same way (to 1e-12 Angstrom) and record its distance
from ASE's own. NVT Berendsen keeps ASE's step exactly (the velocity scaling leaves the forces valid).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from ase import units

from fastmace.config import IntegratorConfig
from fastmace.neighbors.skin import det_3x3, inverse_3x3, matmul_small, times_3x3

from .base import Integrator
from .langevin import base_output_values, fixed_rows
from .nve import drift, half_kick

if TYPE_CHECKING:
    from fastmace.state import SimulationState


def _positive(config, name):
    value = config.parameter(name)
    if value is None or not value > 0:
        raise ValueError(f"Berendsen needs a positive {name} parameter, got {value!r}")
    return float(value)


class Berendsen(Integrator):
    """``ase.md.nvtberendsen.NVTBerendsen``, ``ase.md.nptberendsen.NPTBerendsen`` and
    ``Inhomogeneous_NPTBerendsen`` with the state on the device, one force call per step (module docstring).

    ``config``: the ensemble (``"nvt"`` or ``"npt"``), the timestep (fs), ``fixcm``, ``dof`` and the
    parameters ``taut_fs``; for NPT ``pressure_GPa``, ``taup_fs``, ``compressibility_per_GPa`` and ``mask``
    (``None``: isotropic). The temperature and pressure targets are device scalars.
    """

    def __init__(self, state: SimulationState, config: IntegratorConfig | None = None, *,
                 temperature_K: float | None = None):
        config = IntegratorConfig(scheme="berendsen") if config is None else config
        if config.ensemble not in ("nvt", "npt"):
            raise ValueError(f"Berendsen integrates the 'nvt' or 'npt' ensemble, the config asks for "
                             f"{config.ensemble!r}")
        self.config = config
        self.dt = config.timestep_fs * units.fs
        self.fix_com = config.fixcm
        self.natoms = state.natoms
        self.fixed_rows = fixed_rows(state)
        self.tautscl = self.dt / (_positive(config, "taut_fs") * units.fs)
        self.ndof = self.degrees_of_freedom(state)
        f64 = dict(dtype=torch.float64, device=state.positions.device)
        self.target_temperature = torch.zeros((), **f64)
        self.barostat = config.ensemble == "npt"
        self.mask = None
        if self.barostat:
            if state.stress is None:
                raise ValueError("NPT Berendsen needs a state with stress (a force field with compute_stress=True)")
            taup = _positive(config, "taup_fs") * units.fs
            compressibility = _positive(config, "compressibility_per_GPa") / units.GPa  # ASE's compressibility_au
            mask = config.parameter("mask")
            if mask is None:
                # NPTBerendsen.scale_positions_and_cell: taupscl * compressibility / 3, in ASE's order
                self.pressure_coefficient = self.dt / taup * compressibility / 3.0
            else:
                self.mask = tuple(int(flag) for flag in mask)
                # Inhomogeneous_NPTBerendsen.scale_positions_and_cell
                self.pressure_coefficient = self.dt * compressibility / taup / 3.0
                flags = [float(bool(pbc)) * flag for pbc, flag in zip(state.pbc, self.mask)]
                self.flags = torch.tensor(flags, **f64)
            self.berendsen_pressure = torch.zeros((), **f64)
            self.set_pressure(config.pressure_GPa)
        if temperature_K is None:
            temperature_K = 0.0 if config.temperature_K is None else config.temperature_K
        self.set_temperature(temperature_K)

    @property
    def variable_cell(self) -> bool:
        return self.barostat

    def set_temperature(self, temperature_K: float) -> None:
        self.temperature_K = float(temperature_K)
        self.target_temperature.fill_(self.temperature_K)

    def set_pressure(self, pressure_GPa: float) -> None:
        """Fill the pressure target (GPa; a device fill between steps)."""
        if not self.barostat:
            raise NotImplementedError("NVT Berendsen keeps a fixed cell: it has no pressure target")
        if (isinstance(pressure_GPa, bool) or not isinstance(pressure_GPa, (int, float))
                or not np.isfinite(pressure_GPa)):
            raise ValueError(f"pressure_GPa must be a finite number of GPa, got {pressure_GPa!r}")
        self.restore_externalstress(-pressure_GPa * units.GPa)

    def set_external_stress(self, stress_GPa: Any) -> None:
        """Berendsen's barostats take a scalar pressure only: ``stress_GPa`` is ``-pressure_GPa``."""
        if isinstance(stress_GPa, bool) or not isinstance(stress_GPa, (int, float)):
            raise ValueError(f"Berendsen takes a scalar external stress (-pressure), got {stress_GPa!r}")
        self.set_pressure(-stress_GPa)

    def restore_externalstress(self, externalstress: Any) -> None:
        """The target in ASE units and sign (``-pressure``), as the restart's integrator record stores it."""
        if not self.barostat:
            raise NotImplementedError("NVT Berendsen keeps a fixed cell: it has no pressure target")
        value = np.asarray(externalstress, dtype=np.float64)
        if value.shape != () or not np.isfinite(value):
            raise ValueError(f"Berendsen's external stress is a finite scalar, got {externalstress!r}")
        self.berendsen_pressure.fill_(-float(value))
        self.externalstress_host = value.copy()

    def prepare_step(self, state: SimulationState) -> None:
        """Nothing: no noise."""

    def _scale_velocities(self, state):
        """NVTBerendsen.scale_velocities."""
        v = state.velocities
        kinetic = 0.5 * (state.masses * v ** 2).sum()
        old_temperature = 2 * kinetic / (self.ndof * units.kB)
        scale = torch.sqrt(1.0 + (self.target_temperature / old_temperature - 1.0) * self.tautscl)
        v.mul_(torch.clamp(scale, 0.9, 1.1))

    def _scale_positions_and_cell(self, state):
        """NPTBerendsen / Inhomogeneous_NPTBerendsen.scale_positions_and_cell, with the stress of the last
        force call and the ideal-gas term of the scaled velocities; positions as ``set_cell(scale_atoms=True)``."""
        cell, v = state.cell, state.velocities
        volume = det_3x3(cell)
        mv = state.masses * v
        kinetic = torch.stack([(v[:, a:a + 1] * mv).sum(dim=0) for a in range(3)])
        stress = state.stress - kinetic / volume
        if self.mask is None:
            old_pressure = -torch.diagonal(stress).sum() / 3
            factor = 1.0 - self.pressure_coefficient * (self.berendsen_pressure - old_pressure)
            new_cell = factor * cell
        else:
            factor = 1.0 - self.pressure_coefficient * (self.berendsen_pressure + torch.diagonal(stress)) * self.flags
            new_cell = factor[:, None] * cell
        transform = matmul_small(inverse_3x3(cell), new_cell)
        state.positions.copy_(times_3x3(state.positions, transform))
        cell.copy_(new_cell)

    def first_half(self, state: SimulationState) -> None:
        self._scale_velocities(state)
        if self.barostat:
            self._scale_positions_and_cell(state)
        half_kick(state, self.dt)
        if self.fix_com:
            # ASE's fixcm: subtract the mean momentum sum(p) / N
            v, masses = state.velocities, state.masses
            v.sub_((masses * v).sum(dim=0) / self.natoms / masses)
        if self.fixed_rows is not None:
            state.velocities.masked_fill_(self.fixed_rows, 0.0)
        drift(state, self.dt)

    def second_half(self, state: SimulationState) -> None:
        half_kick(state, self.dt)
        if self.fixed_rows is not None:
            state.velocities.masked_fill_(self.fixed_rows, 0.0)

    def mid_step_tensors(self, state: SimulationState) -> tuple[torch.Tensor, ...]:
        return (state.velocities,)

    def buffers(self, state: SimulationState) -> Mapping[str, torch.Tensor]:
        buffers = dict(positions=state.positions, velocities=state.velocities)
        if self.barostat:
            buffers["cell"] = state.cell
        return buffers

    def output_values(self, state: SimulationState) -> Mapping[str, torch.Tensor]:
        values = base_output_values(state)
        if self.barostat:
            values.update(cell=state.cell, stress=state.stress)
        return values

    def volume(self, state: SimulationState) -> torch.Tensor:
        return det_3x3(state.cell)

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if state_dict:
            raise ValueError(f"Berendsen has no internal state, got {sorted(state_dict)}")
