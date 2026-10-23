"""Graph-capturable MD integrators over SimulationState and a ForceField (SPEC §10).

This package's namespace is ``fm.integrators``: the public integrator records (plan task 1b-T01),
which a ``Simulation`` takes as ``integrator=`` and which map onto ``IntegratorConfig``. The
implementations (``langevin.py``, ``baoab.py``) are built from that config by the planner only
(SPEC §5); importing this package imports neither them nor torch.

Units at the API (SPEC §7.3): fs, K, GPa and friction per fs. ``IntegratorConfig.friction`` is in
inverse ASE time units, as in ``main.py``; ``to_config`` converts (``friction_per_fs / units.fs``).

Phase 4 (plan task 4-T01, conductor's choice 1) adds the records of the new schemes: ``NVE``, ``CSVR``,
``NoseHooverChain``, ``MTK`` and ``Berendsen``. Their scheme settings travel in
``IntegratorConfig.parameters`` in the record's units (fs, K, GPa); the implementations (4-T02, 4-T03)
convert. Time constants have no defaults, as in ASE: physics is never chosen silently. A record whose
scheme is not registered yet plans to a ``scheme-unknown`` refusal (rule 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from fastmace.config import IntegratorConfig, _check_choices, external_stress

__all__ = ["CSVR", "MTK", "NVE", "RECORDS", "Berendsen", "Langevin", "LangevinBAOAB", "NoseHooverChain"]


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
    - ``disable_cell_langevin``: as ASE;
    - ``max_volume_ratio`` and ``max_cell_length_A``: the cell-divergence stop (3b-T07). A
      ``Simulation`` stops with ``CellDivergedError`` (restart and metadata written first) once the
      volume leaves [V0 / ratio, V0 ratio], V0 the run's starting volume, or a cell vector is longer
      than ``max_cell_length_A`` (1000 Å: float32 coordinates still resolve 6e-5 Å there). A vapour at
      a low pressure expands without bound otherwise, until the coordinates lose their precision.
      ``None`` turns a limit off.

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
    max_volume_ratio: float | None = 20.0
    max_cell_length_A: float | None = 1000.0

    def __post_init__(self):
        _validate(self)
        for field, low in (("max_volume_ratio", 1.0), ("max_cell_length_A", 0.0)):
            value = getattr(self, field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value <= low):
                raise ValueError(f"LangevinBAOAB.{field} must be None or finite and above {low:g}, got {value!r}")
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


def _number(record, name, *, minimum=None, positive=False, required=True):
    value = getattr(record, name)
    if value is None and not required:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{type(record).__name__}.{name} must be a finite number, got {value!r}")
    if positive and value <= 0:
        raise ValueError(f"{type(record).__name__}.{name} must be positive, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{type(record).__name__}.{name} must be at least {minimum}, got {value!r}")


def _chain(record, *names):
    for name in names:
        value = getattr(record, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{type(record).__name__}.{name} must be an int of at least 1, got {value!r}")


def _common(record, *, thermostat=True):
    _check_choices(record)
    _number(record, "timestep_fs", positive=True)
    if thermostat:
        _number(record, "temperature_K", minimum=0)


def _base_config(record, seed, *, ensemble, scheme, parameters, **fields):
    return IntegratorConfig(ensemble=ensemble, scheme=scheme, timestep_fs=float(record.timestep_fs), seed=seed,
                            fixcm=record.fixcm, dof=record.dof, parameters=parameters, **fields)


@dataclass(frozen=True, kw_only=True)
class NVE:
    """Velocity Verlet, ASE's ``VelocityVerlet`` on the graph engine (SPEC §10.2; 4-T02).

    ``fixcm``: remove the total momentum at the start and project it out each step (owner decision D2);
    ``dof``: ``"auto"`` (3N - 3 under ``fixcm``) or ``"ase"`` (ASE's count), for the reported temperature.
    The conserved quantity is E_kin + E_pot.
    """

    timestep_fs: float
    fixcm: bool = True
    dof: Literal["auto", "ase"] = "auto"

    def __post_init__(self):
        _common(self, thermostat=False)

    def to_config(self, *, seed: int) -> IntegratorConfig:
        """The ``IntegratorConfig`` of this record with ``seed`` (the ``Simulation``'s)."""
        return _base_config(self, seed, ensemble="nve", scheme="nve", parameters={})


@dataclass(frozen=True, kw_only=True)
class CSVR:
    """Canonical sampling through velocity rescaling, ASE 3.27's ``Bussi`` on the graph engine (4-T02).

    ``tau_fs`` is ASE's ``taut``. The noise is ASE's stream from the ``Simulation``'s seed (conductor's
    choice 3). The conserved quantity adds the energy the thermostat transferred, as ASE's ``Bussi``.
    """

    timestep_fs: float
    temperature_K: float
    tau_fs: float
    fixcm: bool = True
    dof: Literal["auto", "ase"] = "auto"

    def __post_init__(self):
        _common(self)
        _number(self, "tau_fs", positive=True)

    def to_config(self, *, seed: int) -> IntegratorConfig:
        """The ``IntegratorConfig`` of this record with ``seed`` (the ``Simulation``'s)."""
        return _base_config(self, seed, ensemble="nvt", scheme="csvr", parameters={"tau_fs": float(self.tau_fs)},
                            temperature_K=float(self.temperature_K))


@dataclass(frozen=True, kw_only=True)
class NoseHooverChain:
    """Nose-Hoover chains, ASE 3.27's ``NoseHooverChainNVT`` on the graph engine (4-T03).

    ``tdamp_fs``, ``tchain`` and ``tloop`` as ASE's ``tdamp``, ``tchain``, ``tloop``. The thermostat
    masses come from the construction temperature, as ASE's (conductor's choice 4). ``fixcm`` and ``dof``
    follow D2.
    """

    timestep_fs: float
    temperature_K: float
    tdamp_fs: float
    tchain: int = 3
    tloop: int = 1
    fixcm: bool = True
    dof: Literal["auto", "ase"] = "auto"

    def __post_init__(self):
        _common(self)
        _number(self, "tdamp_fs", positive=True)
        _chain(self, "tchain", "tloop")

    def to_config(self, *, seed: int) -> IntegratorConfig:
        """The ``IntegratorConfig`` of this record with ``seed`` (the ``Simulation``'s)."""
        return _base_config(self, seed, ensemble="nvt", scheme="nhc", temperature_K=float(self.temperature_K),
                            parameters={"tdamp_fs": float(self.tdamp_fs), "tchain": self.tchain,
                                        "tloop": self.tloop})


@dataclass(frozen=True, kw_only=True)
class MTK:
    """Martyna-Tobias-Klein NPT with Nose-Hoover chains, ASE 3.27's ``IsotropicMTKNPT`` (``cell_mode=
    "hydrostatic"``) and ``MTKNPT`` (``"full"``) on the graph engine (4-T03). Experimental: not yet
    qualified, as BAOAB (SPEC §10.2).

    ``pressure_GPa`` is ASE's ``pressure_au`` in GPa; ``tdamp_fs``, ``pdamp_fs``, ``tchain``, ``pchain``,
    ``tloop`` and ``ploop`` as ASE's. ``fixcm`` and ``dof`` follow D2.
    """

    timestep_fs: float
    temperature_K: float
    pressure_GPa: float
    tdamp_fs: float
    pdamp_fs: float
    tchain: int = 3
    pchain: int = 3
    tloop: int = 1
    ploop: int = 1
    cell_mode: Literal["hydrostatic", "full"] = "hydrostatic"
    fixcm: bool = True
    dof: Literal["auto", "ase"] = "auto"

    def __post_init__(self):
        _common(self)
        _number(self, "pressure_GPa")
        _number(self, "tdamp_fs", positive=True)
        _number(self, "pdamp_fs", positive=True)
        _chain(self, "tchain", "pchain", "tloop", "ploop")

    def to_config(self, *, seed: int) -> IntegratorConfig:
        """The ``IntegratorConfig`` of this record with ``seed`` (the ``Simulation``'s)."""
        return _base_config(self, seed, ensemble="npt", scheme="mtk", temperature_K=float(self.temperature_K),
                            pressure_GPa=float(self.pressure_GPa), cell_mode=self.cell_mode,
                            parameters={"tdamp_fs": float(self.tdamp_fs), "pdamp_fs": float(self.pdamp_fs),
                                        "tchain": self.tchain, "pchain": self.pchain, "tloop": self.tloop,
                                        "ploop": self.ploop})


@dataclass(frozen=True, kw_only=True)
class Berendsen:
    """Berendsen thermostat and barostat: ASE 3.27's ``NVTBerendsen`` (no pressure), ``NPTBerendsen``
    (``pressure_GPa``) and ``Inhomogeneous_NPTBerendsen`` (``pressure_GPa`` and ``mask``) on the graph
    engine (4-T02). Non-canonical: equilibration only; the plan warns.

    ``taut_fs``, ``taup_fs`` and ``compressibility_per_GPa`` as ASE's ``taut``, ``taup`` and
    ``compressibility_au`` in fs and 1/GPa; ``mask``: three 0/1 flags of the scaled cell directions,
    ASE's ``mask``. A pressure needs ``taup_fs`` and ``compressibility_per_GPa``.
    """

    timestep_fs: float
    temperature_K: float
    taut_fs: float
    pressure_GPa: float | None = None
    taup_fs: float | None = None
    compressibility_per_GPa: float | None = None
    mask: tuple[int, int, int] | None = None
    fixcm: bool = True
    dof: Literal["auto", "ase"] = "auto"

    def __post_init__(self):
        _common(self)
        _number(self, "taut_fs", positive=True)
        _number(self, "pressure_GPa", required=False)
        _number(self, "taup_fs", positive=True, required=False)
        _number(self, "compressibility_per_GPa", positive=True, required=False)
        barostat = (self.taup_fs, self.compressibility_per_GPa)
        if self.pressure_GPa is None and (any(value is not None for value in barostat) or self.mask is not None):
            raise ValueError("Berendsen: taup_fs, compressibility_per_GPa and mask need pressure_GPa")
        if self.pressure_GPa is not None and any(value is None for value in barostat):
            raise ValueError("Berendsen: pressure_GPa needs taup_fs and compressibility_per_GPa (ASE requires "
                             "the compressibility; no default is chosen)")
        if self.mask is not None:
            mask = tuple(self.mask)
            if len(mask) != 3 or any(isinstance(flag, bool) or flag not in (0, 1) for flag in mask):
                raise ValueError(f"Berendsen.mask must be three 0/1 flags, got {self.mask!r}")
            object.__setattr__(self, "mask", tuple(int(flag) for flag in mask))

    def to_config(self, *, seed: int) -> IntegratorConfig:
        """The ``IntegratorConfig`` of this record with ``seed`` (the ``Simulation``'s)."""
        parameters = {"taut_fs": float(self.taut_fs)}
        if self.pressure_GPa is None:
            return _base_config(self, seed, ensemble="nvt", scheme="berendsen", parameters=parameters,
                                temperature_K=float(self.temperature_K))
        parameters.update(taup_fs=float(self.taup_fs), compressibility_per_GPa=float(self.compressibility_per_GPa),
                          mask=self.mask)
        return _base_config(self, seed, ensemble="npt", scheme="berendsen", parameters=parameters,
                            temperature_K=float(self.temperature_K), pressure_GPa=float(self.pressure_GPa),
                            cell_mode="hydrostatic" if self.mask is None else "full")


# Every public integrator record, by class name (Simulation's records and resume).
RECORDS = {cls.__name__: cls for cls in (Langevin, LangevinBAOAB, NVE, CSVR, NoseHooverChain, MTK, Berendsen)}
