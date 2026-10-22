"""Graph-capturable MD integrators over SimulationState and a ForceField (SPEC §10).

This package's namespace is ``fm.integrators``: the public integrator records (plan task 1b-T01),
which a ``Simulation`` takes as ``integrator=`` and which map onto ``IntegratorConfig``. The
implementations (``langevin.py``, ``baoab.py``) are built from that config by the planner only
(SPEC §5); importing this package imports neither them nor torch.

Units at the API (SPEC §7.3): fs, K, GPa and friction per fs. ``IntegratorConfig.friction`` is in
inverse ASE time units, as in ``main.py``; ``to_config`` converts (``friction_per_fs / units.fs``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from fastmace.config import IntegratorConfig, _check_choices, external_stress

__all__ = ["Langevin", "LangevinBAOAB"]


def _validate(record):
    _check_choices(record)
    name = type(record).__name__
    for field in ("timestep_fs", "friction_per_fs"):
        value = getattr(record, field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name}.{field} must be positive and finite, got {value!r}")
    value = record.temperature_K
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name}.temperature_K must be finite and nonnegative, got {value!r}")


def _friction(friction_per_fs):
    from ase import units  # here: importing fm.integrators stays free of ASE and torch

    return friction_per_fs / units.fs


@dataclass(frozen=True, kw_only=True)
class Langevin:
    """Langevin NVT, ASE's ``Langevin`` on the graph engine (SPEC §10.2).

    ``fm.integrators.Langevin(timestep_fs=0.5, temperature_K=300, friction_per_fs=0.02)``.
    ``temperature_K`` is the starting target; a schedule (``Simulation.run(schedule=…)``) moves it.
    ``fixcm_projection`` is how ``fixcm`` projects the noise (``IntegratorConfig.fixcm_projection``;
    docs/design/fixcm-projection.md): ``"ase"``, ASE's, ``"mass_weighted"``, canonical at a fixed centre
    of mass for unequal masses too, or ``"auto"``, the planner's choice (``"mass_weighted"`` for MC plans
    under ``fixcm``, else ``"ase"``).
    """

    timestep_fs: float = 0.5
    temperature_K: float
    friction_per_fs: float
    fixcm: bool = True
    fixcm_projection: Literal["auto", "ase", "mass_weighted"] = "auto"

    def __post_init__(self):
        _validate(self)

    def to_config(self, *, seed: int) -> IntegratorConfig:
        """The ``IntegratorConfig`` of this record with ``seed`` (the ``Simulation``'s)."""
        return IntegratorConfig(ensemble="nvt", timestep_fs=float(self.timestep_fs),
                                friction=_friction(self.friction_per_fs), seed=seed, fixcm=self.fixcm,
                                fixcm_projection=self.fixcm_projection, temperature_K=float(self.temperature_K))


@dataclass(frozen=True, kw_only=True)
class LangevinBAOAB:
    """Langevin-Hoover NPT, ASE 3.27's ``LangevinBAOAB`` on the graph engine (SPEC §10.2).

    - ``friction_per_fs``: the thermostat friction; ASE's ``T_tau`` is ``1 / friction``, as in
      ``main.py`` (``workflow.baoab_settings``);
    - ``pressure_GPa`` or ``external_stress_GPa`` (ASE's ``externalstress`` sign: a scalar or six
      Voigt components), not both;
    - ``cell_mode``: ``"hydrostatic"`` (ASE ``hydrostatic=True``) or ``"full"``;
    - ``barostat_tau_fs``: ASE's ``P_tau``, ``None`` = ASE's default 20 ``T_tau``;
    - ``barostat_mass_factor``: ASE's ``P_mass_factor``;
    - ``disable_cell_langevin``: as ASE.

    The production drag term is the corrected one (SPEC §12.1, 1b-T09); ASE's own variant stays an
    ``IntegratorConfig`` option for comparisons with ASE.
    """

    timestep_fs: float = 0.5
    temperature_K: float
    friction_per_fs: float
    pressure_GPa: float = 0.0
    external_stress_GPa: float | tuple[float, float, float, float, float, float] | None = None
    cell_mode: Literal["hydrostatic", "full"] = "hydrostatic"
    barostat_tau_fs: float | None = None
    barostat_mass_factor: float = 1.0
    disable_cell_langevin: bool = False
    fixcm: bool = True

    def __post_init__(self):
        _validate(self)
        object.__setattr__(self, "external_stress_GPa",
                           external_stress(self.external_stress_GPa, "LangevinBAOAB.external_stress_GPa"))
        # The remaining checks are IntegratorConfig's, so record and config cannot disagree.
        self._config(seed=0, friction=1.0)

    def _config(self, seed, friction):
        return IntegratorConfig(ensemble="npt", timestep_fs=float(self.timestep_fs), friction=friction, seed=seed,
                                fixcm=self.fixcm, pressure_GPa=float(self.pressure_GPa), cell_mode=self.cell_mode,
                                barostat_tau_fs=self.barostat_tau_fs, barostat_mass_factor=self.barostat_mass_factor,
                                temperature_K=float(self.temperature_K), external_stress_GPa=self.external_stress_GPa,
                                disable_cell_langevin=self.disable_cell_langevin)

    def to_config(self, *, seed: int) -> IntegratorConfig:
        """The ``IntegratorConfig`` of this record with ``seed`` (the ``Simulation``'s)."""
        return self._config(seed, _friction(self.friction_per_fs))
