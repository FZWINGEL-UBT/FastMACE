"""The D3 dispersion record ``fm.D3`` and the ``DispersionTerm`` interface (SPEC §11.2, §7.7).

Plan task 1b-T01 fixes the record; 1b-T20 adds the ``"d3bj"`` expansion from the head's training
functional (``expand``); 3b-T01 adds the explicit Becke–Johnson numbers, the precision semantics,
the identity entry and the ``DispersionTerm`` interface that the D3 list (3b-T02) and kernel
(3b-T03) implement. Until 3b-T04 integrates them, plans with D3 are refused ("D3: phase 3b"), with
the record shown (``describe``). The contract is docs/design/d3.md.

"D3(BJ)" alone does not identify the correction, so every choice is a field, and the whole record
is part of the force-field identity (§10.6). The defaults are the §11.2 table (the 15 October
2026 study, ``torch-dftd`` 0.5.1 as ``mace_mp`` calls it), except ``functional``, which has no
default: it must match the model's training functional, and only the label table knows that.
"""

from __future__ import annotations

import abc
import dataclasses
import math
import time
from dataclasses import dataclass
from typing import Any, Literal

from fastmace.config import _check_choices

# Becke–Johnson parameter sets (s6, s8, a1, a2) by functional, as torch-dftd 0.5.1's
# ``get_dftd3_default_params("bj", xc)`` has them (its rs6 = a1, s18 = s8, rs18 = a2). r2scan:
# Table 1 of doi 10.1063/5.0041008; pbe: Grimme, Ehrlich and Goerigk 2011.
BJ_PARAMETERS = {
    "pbe": {"s6": 1.0, "s8": 0.7875, "a1": 0.4289, "a2": 4.4407},
    "r2scan": {"s6": 1.0, "s8": 0.7898, "a1": 0.4948, "a2": 5.7308},
}
BJ_NUMBERS = ("s6", "s8", "a1", "a2")

# The reference tables (C6 references with their CN, R0, covalent radii, r2/r4) every D3 term uses:
# torch-dftd 0.5.1's ``nn/params/dftd3_params.npz``, by its SHA-256. Part of the identity entry.
REFERENCE_TABLES = ("torch-dftd-0.5.1:dftd3_params.npz:sha256="
                    "52b7b35a957df55b813035ffde3a6b5a41dec585bdb1e281fb8ba67a729f3d92")


@dataclass(frozen=True, kw_only=True)
class D3:
    """A complete D3 dispersion correction.

    - ``functional``: the parameter set, e.g. ``"r2scan"`` (MH-1 ``matpes_r2scan``) or ``"pbe"``;
    - ``damping``: Becke–Johnson;
    - ``pair_cutoff_bohr``, ``cn_cutoff_bohr``: 40 bohr each (``mace_mp`` and ``torch-dftd``);
    - ``three_body``: the ATM term, off (``ForceField`` refuses it on: no D3 term implements it);
    - ``smoothing``: cutoff smoothing, none;
    - ``precision``: the precision of the CN, energy and virial reductions, float64 only. The
      pair terms run in the plan's force precision (conductor's choice 3, docs/design/d3.md);
    - ``autoang``, ``autoev``: ``torch-dftd``'s unit constants (they differ from ASE's ``Bohr`` and
      ``Hartree`` by about 1e-7 relative), recorded with the parameters;
    - ``s6``, ``s8``, ``a1``, ``a2``: the Becke–Johnson numbers. Left out (all four), they are the
      functional's set in ``BJ_PARAMETERS``; given (all four), they are used as given and
      ``functional`` only names them. Either way the record holds the numbers, so equal records
      are equal corrections.
    """

    functional: str
    damping: Literal["bj"] = "bj"
    pair_cutoff_bohr: float = 40.0
    cn_cutoff_bohr: float = 40.0
    three_body: bool = False
    smoothing: Literal["none"] = "none"
    precision: Literal["float64"] = "float64"
    autoang: float = 0.52917726
    autoev: float = 27.21138505
    s6: float | None = None
    s8: float | None = None
    a1: float | None = None
    a2: float | None = None

    def __post_init__(self):
        _check_choices(self)
        if not isinstance(self.functional, str) or not self.functional:
            raise ValueError(f"D3.functional must name a parameter set, got {self.functional!r}")
        for name in ("pair_cutoff_bohr", "cn_cutoff_bohr", "autoang", "autoev"):
            value = getattr(self, name)
            if not _number(value) or value <= 0:
                raise ValueError(f"D3.{name} must be positive and finite, got {value!r}")
        given = [name for name in BJ_NUMBERS if getattr(self, name) is not None]
        if not given:
            if self.functional not in BJ_PARAMETERS:
                raise ValueError(f"D3: no Becke–Johnson parameter set for functional {self.functional!r} (known: "
                                 f"{', '.join(BJ_PARAMETERS)}); give s6, s8, a1, a2 explicitly")
            for name, value in BJ_PARAMETERS[self.functional].items():
                object.__setattr__(self, name, value)
        elif len(given) < len(BJ_NUMBERS):
            raise ValueError(f"D3: give all four of s6, s8, a1, a2 or none, got only {', '.join(given)}")
        for name in BJ_NUMBERS:
            value = getattr(self, name)
            positive = name in ("s6", "a2")
            if not _number(value) or value < 0 or (positive and value == 0):
                raise ValueError(f"D3.{name} must be finite and {'positive' if positive else 'non-negative'}, "
                                 f"got {value!r}")
            object.__setattr__(self, name, float(value))


def _number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def pair_cutoff_A(record: D3) -> float:
    """The pair cutoff in Å, converted with the record's own ``autoang`` (40 bohr: 21.1670904 Å)."""
    return record.pair_cutoff_bohr * record.autoang


def cn_cutoff_A(record: D3) -> float:
    """The coordination-number cutoff in Å, converted with the record's own ``autoang``."""
    return record.cn_cutoff_bohr * record.autoang


def identity_entry(record: D3) -> dict:
    """The record as the force-field identity's ``physics["dispersion"]`` (§10.6): every field and
    the reference tables (``REFERENCE_TABLES``); JSON-safe."""
    entry = {field.name: getattr(record, field.name) for field in dataclasses.fields(record)}
    entry["tables"] = REFERENCE_TABLES
    return entry


# Training functionals (label-table names, ``fastmace.model.labels``) whose D3(BJ) parameter set
# ``"d3bj"`` expands to; the set has the functional's name. SPEC §7.7: ``pbe`` for the PBE-trained
# models and heads, ``r2scan`` for ``matpes_r2scan`` and MatPES r2SCAN.
D3BJ_PARAMETER_SETS = ("pbe", "r2scan")

# Training functionals that already contain dispersion: D3 on top would count it twice (§7.7).
DISPERSION_INCLUDED = {
    "wb97m-d3bj": "D3(BJ) is part of the ωB97M-D3(BJ) training reference (MACE-OFF, MH-1's spice_wB97M head)",
    "wb97m-v": "ωB97M-V includes the VV10 nonlocal dispersion term (OMOL-0, MH-1's omol head)",
}


def describe(record: D3) -> str:
    """The record as one line, every field named: what the report, metadata and refusals show."""
    return "D3(" + ", ".join(f"{field.name}={getattr(record, field.name)}"
                             for field in dataclasses.fields(record)) + ")"


def expand(dispersion, *, functional, model):
    """The complete record for ``fm.md``'s ``dispersion=``, and its warnings: ``(D3 | None, (str, ...))``.

    - ``None``: no dispersion;
    - ``"d3bj"``: Becke–Johnson with the parameter set of ``functional`` (the head's training
      functional from the label table, ``None`` if unknown) and the §11.2 defaults. ``ValueError``,
      naming ``model`` and the reason, when the functional already contains dispersion or is
      unknown or has no D3(BJ) parameter set here; then only an explicit ``fm.D3`` is accepted;
    - an ``fm.D3``: used as given; a functional other than the training one is a warning.
    """
    if dispersion is None:
        return None, ()
    if isinstance(dispersion, D3):
        if functional is not None and dispersion.functional != functional:
            return dispersion, (f"D3 functional {dispersion.functional!r} differs from the training functional "
                                f"{functional!r} of {model} (SPEC §7.7)",)
        return dispersion, ()
    if not isinstance(dispersion, str):
        raise TypeError(f"dispersion must be None, 'd3bj' or an fm.D3 record, got {type(dispersion).__name__}")
    if dispersion != "d3bj":
        raise ValueError(f"dispersion={dispersion!r} is not 'd3bj'; pass dispersion='d3bj' or an explicit "
                         "fm.D3(functional=...) record (SPEC §7.7)")
    if functional in DISPERSION_INCLUDED:
        raise ValueError(f"dispersion='d3bj' for {model}: its training functional {functional!r} already contains "
                         f"dispersion ({DISPERSION_INCLUDED[functional]}), so D3 would count it twice (SPEC §7.7)")
    if functional is None:
        raise ValueError(f"dispersion='d3bj' for {model}: the training functional is unknown (a model outside the "
                         "label table, or a label whose release names none), so no D3 parameter set follows from it; "
                         "pass an explicit fm.D3(functional=...) (SPEC §7.7)")
    if functional not in D3BJ_PARAMETER_SETS:
        raise ValueError(f"dispersion='d3bj' for {model}: no D3(BJ) parameter set is mapped for its training "
                         f"functional {functional!r} (mapped: {', '.join(D3BJ_PARAMETER_SETS)}); pass an explicit "
                         "fm.D3(functional=...) (SPEC §7.7)")
    return D3(functional=functional), ()


# --------------------------------------------------------------------------- the DispersionTerm interface


@dataclass(frozen=True)
class DispersionResult:
    """One D3 evaluation, as device tensors.

    - ``energy``: 0-d float64, eV (reduced in float64);
    - ``forces``: (N, 3) in the plan's force precision, eV/Å;
    - ``virial``: (3, 3) float64, eV, in MACE's convention ``-dE/d(strain)``, or ``None`` when not
      asked for. The stress in ASE's sign is ``-virial / volume`` (``stress``).
    """

    energy: Any
    forces: Any
    virial: Any = None

    def stress(self, volume):
        """The stress in ASE's sign (positive under tension), eV/Å³, as a (3, 3) tensor."""
        return -self.virial / volume


class DispersionTerm(abc.ABC):
    """A D3 term of a force field: its own pair list and its evaluation (SPEC §11.2, docs/design/d3.md).

    ``DispersionTerm(record, numbers)``: ``record`` is the complete ``fm.D3``; ``numbers`` the atomic
    numbers as an int64 device tensor at a fixed address, which species events change in place
    (the owner then calls ``species_changed``). Positions and cells are float64 device tensors in Å.

    - ``build(positions, cell)``: (re)build the pair list around ``positions`` and ``cell``. Host
      side: it may synchronise and grow buffers. True when the capacity grew or buffer addresses
      otherwise changed, so the engine must capture its graphs again (counted as ``regrowths``).
      ``counters()`` counts and times it;
    - ``needs_rebuild(positions, cell)``: a device bool tensor with no host synchronisation and no
      data-dependent shapes, so it is capturable and can be OR-ed into the fused stage's rebuild
      flag like the MACE list's: has an atom moved or the cell strained beyond what the list's
      skin covers since the last build;
    - ``evaluate(positions, cell, *, virial=True)``: a ``DispersionResult``; capturable in the GPU
      implementations (masks with ``where``, fixed shapes);
    - ``evaluate_energy(positions, cell)``: the energy alone, a 0-d float64 tensor, with no force or
      virial writes (MC energy evaluations); capturable likewise;
    - ``species_changed()``: refresh per-species derived state (C6 tables, radii) after ``numbers``
      changed in place (conductor's choice 5).

    Subclasses implement ``_build``, ``needs_rebuild``, ``evaluate`` and ``evaluate_energy``.
    """

    def __init__(self, record: D3, numbers):
        if not isinstance(record, D3):
            raise TypeError(f"record must be an fm.D3, got {type(record).__name__}")
        self.record = record
        self.numbers = numbers
        self.builds = 0
        self.regrowths = 0
        self.build_seconds = 0.0

    @property
    def rebuilds(self) -> int:
        """Builds after the first."""
        return max(self.builds - 1, 0)

    def build(self, positions, cell) -> bool:
        """Build the list (class docstring); counts the build, a regrowth and its host wall time.

        True when buffer addresses changed (the capacity grew): captured graphs must be captured again.
        """
        start = time.perf_counter()
        moved = bool(self._build(positions, cell))
        self.build_seconds += time.perf_counter() - start
        self.builds += 1
        self.regrowths += moved
        return moved

    def counters(self) -> dict:
        """The runtime counters (§9.6, §13): ``builds``, ``rebuilds``, ``regrowths`` (builds that moved
        buffers, the first included) and ``build_seconds``."""
        return {"builds": self.builds, "rebuilds": self.rebuilds, "regrowths": self.regrowths,
                "build_seconds": self.build_seconds}

    def species_changed(self) -> None:
        """Refresh per-species derived state after ``numbers`` changed in place; none by default."""

    @abc.abstractmethod
    def _build(self, positions, cell) -> bool:
        """The build itself; True when buffer addresses changed."""

    @abc.abstractmethod
    def needs_rebuild(self, positions, cell):
        """A device bool tensor: does the list need a rebuild? Capturable."""

    @abc.abstractmethod
    def evaluate(self, positions, cell, *, virial: bool = True) -> DispersionResult:
        """Energy, forces and, if asked for, the virial of the D3 term."""

    @abc.abstractmethod
    def evaluate_energy(self, positions, cell):
        """The energy alone (0-d float64, eV), with no force or virial writes. Capturable."""
