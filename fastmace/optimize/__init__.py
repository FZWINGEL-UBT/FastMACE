"""GPU structure relaxations, fixed and variable cell (SPEC §10.3).

This package's namespace is ``fm.optimize``: the public optimiser records (plan task 4-T01, conductor's
choice 1), which a ``Simulation`` takes as ``optimizer=`` (4-T05) and which map onto ``OptimizerConfig``.
The implementations (``lbfgs.py``, 4-T05; ``fire.py``, 4-T06; ``cell.py``, 4-T07) are built by the planner
only, over a ``base.DOFView`` (SPEC §5); importing this package imports neither them nor torch.

The optimisers take ASE 3.27's arguments and defaults; lengths are Angstrom. FIRE's ``dt`` and ``dtmax`` are
in ASE time units, as ASE's (they are step-size parameters, not physical times). ``FrechetCell`` is ASE's
``FrechetCellFilter`` with the pressure in GPa.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from fastmace.config import OptimizerConfig

__all__ = ["FIRE", "LBFGS", "RECORDS", "FrechetCell"]


def _number(record, name, *, positive=False, minimum=None, required=True):
    value = getattr(record, name)
    if value is None and not required:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{type(record).__name__}.{name} must be a finite number, got {value!r}")
    if positive and value <= 0:
        raise ValueError(f"{type(record).__name__}.{name} must be positive, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{type(record).__name__}.{name} must be at least {minimum}, got {value!r}")


def _flag(record, name):
    if not isinstance(getattr(record, name), bool):
        raise ValueError(f"{type(record).__name__}.{name} must be a bool, got {getattr(record, name)!r}")


@dataclass(frozen=True, kw_only=True)
class FrechetCell:
    """ASE 3.27's ``FrechetCellFilter`` on the device (4-T07): positions and the log of the deformation
    gradient, both optimisers unchanged on it.

    ``scalar_pressure_GPa`` is ASE's ``scalar_pressure`` in GPa; ``hydrostatic_strain``, ``constant_volume``,
    ``mask`` (six Voigt flags or a 3x3 of flags, as ASE) and ``exp_cell_factor`` (``None``: N, ASE's
    default) as ASE's.
    """

    scalar_pressure_GPa: float = 0.0
    hydrostatic_strain: bool = False
    constant_volume: bool = False
    mask: tuple | None = None
    exp_cell_factor: float | None = None

    def __post_init__(self):
        _number(self, "scalar_pressure_GPa")
        _flag(self, "hydrostatic_strain")
        _flag(self, "constant_volume")
        _number(self, "exp_cell_factor", positive=True, required=False)
        if self.mask is not None:
            mask = tuple(tuple(row) if isinstance(row, (list, tuple)) else row for row in self.mask)
            flat = [flag for row in mask for flag in (row if isinstance(row, tuple) else (row,))]
            shape_ok = (len(mask) == 6 and len(flat) == 6) or (len(mask) == 3 and len(flat) == 9)
            if not shape_ok or any(isinstance(flag, bool) or flag not in (0, 1) for flag in flat):
                raise ValueError(f"FrechetCell.mask must be six Voigt 0/1 flags or a 3x3 of 0/1 flags, got "
                                 f"{self.mask!r}")
            object.__setattr__(self, "mask", mask)

    def parameters(self) -> dict:
        return {"scalar_pressure_GPa": float(self.scalar_pressure_GPa), "hydrostatic_strain": self.hydrostatic_strain,
                "constant_volume": self.constant_volume, "mask": self.mask,
                "exp_cell_factor": None if self.exp_cell_factor is None else float(self.exp_cell_factor)}


def _cell_filter(record):
    cell = record.cell_filter
    if cell is not None and not isinstance(cell, FrechetCell):
        raise TypeError(f"{type(record).__name__}.cell_filter must be fm.optimize.FrechetCell or None, got "
                        f"{type(cell).__name__}")
    return (None, {}) if cell is None else ("frechet", cell.parameters())


@dataclass(frozen=True, kw_only=True)
class LBFGS:
    """ASE 3.27's ``LBFGS`` on the graph engine (4-T05): ``maxstep_A`` (ASE ``maxstep``), ``memory``,
    ``damping`` and ``alpha`` with ASE's defaults. ``use_line_search`` is not offered (refused in the
    ``fastmace.ase`` class). ``cell_filter``: ``None`` (fixed cell) or ``FrechetCell(...)``."""

    maxstep_A: float = 0.2
    memory: int = 100
    damping: float = 1.0
    alpha: float = 70.0
    cell_filter: FrechetCell | None = None

    def __post_init__(self):
        _number(self, "maxstep_A", positive=True)
        _number(self, "damping", positive=True)
        _number(self, "alpha", positive=True)
        if isinstance(self.memory, bool) or not isinstance(self.memory, int) or self.memory < 1:
            raise ValueError(f"LBFGS.memory must be an int of at least 1, got {self.memory!r}")
        _cell_filter(self)

    def to_config(self, *, fmax: float = 0.05, steps: int = 100_000_000) -> OptimizerConfig:
        """The ``OptimizerConfig`` with ASE's ``run(fmax=, steps=)``."""
        cell, cell_parameters = _cell_filter(self)
        return OptimizerConfig(scheme="lbfgs", parameters={"maxstep_A": float(self.maxstep_A), "memory": self.memory,
                                                           "damping": float(self.damping), "alpha": float(self.alpha)},
                               cell_filter=cell, cell_filter_parameters=cell_parameters, fmax=fmax, steps=steps)


@dataclass(frozen=True, kw_only=True)
class FIRE:
    """ASE 3.27's ``FIRE`` on the graph engine (4-T06), with ASE's arguments and defaults: ``dt`` and
    ``dtmax`` in ASE time units, ``maxstep_A`` (ASE ``maxstep``), ``Nmin``, ``finc``, ``fdec``, ``astart``,
    ``fa``, ``a`` and ``downhill_check``. ASE's deprecated ``maxmove`` and ``position_reset_callback`` are not
    offered. ``cell_filter``: ``None`` or ``FrechetCell(...)``."""

    dt: float = 0.1
    maxstep_A: float = 0.2
    dtmax: float = 1.0
    Nmin: int = 5
    finc: float = 1.1
    fdec: float = 0.5
    astart: float = 0.1
    fa: float = 0.99
    a: float = 0.1
    downhill_check: bool = False
    cell_filter: FrechetCell | None = None

    def __post_init__(self):
        for name in ("dt", "maxstep_A", "dtmax", "finc", "fdec", "astart", "fa"):
            _number(self, name, positive=True)
        _number(self, "a", minimum=0)
        if isinstance(self.Nmin, bool) or not isinstance(self.Nmin, int) or self.Nmin < 0:
            raise ValueError(f"FIRE.Nmin must be a nonnegative int, got {self.Nmin!r}")
        _flag(self, "downhill_check")
        _cell_filter(self)

    def to_config(self, *, fmax: float = 0.05, steps: int = 100_000_000) -> OptimizerConfig:
        """The ``OptimizerConfig`` with ASE's ``run(fmax=, steps=)``."""
        cell, cell_parameters = _cell_filter(self)
        parameters = {name: getattr(self, name) for name in ("dt", "maxstep_A", "dtmax", "Nmin", "finc", "fdec",
                                                             "astart", "fa", "a", "downhill_check")}
        return OptimizerConfig(scheme="fire", parameters=parameters, cell_filter=cell,
                               cell_filter_parameters=cell_parameters, fmax=fmax, steps=steps)


# Every public optimiser record, by class name.
RECORDS = {cls.__name__: cls for cls in (LBFGS, FIRE)}
