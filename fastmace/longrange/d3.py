"""The D3 dispersion record, ``fm.D3`` (SPEC §11.2, §7.7 "Dispersion").

Plan task 1b-T01 fixes the record; 1b-T20 adds the ``"d3bj"`` expansion from the head's training
functional (``expand``). The GPU kernel is phase 3b: until then ``fm.md`` refuses a complete
record at plan time ("D3: phase 3b"), with the record shown (``describe``).

"D3(BJ)" alone does not identify the correction, so every choice is a field, and the whole record
is part of the force-field identity (§10.6). The defaults are the §11.2 table (the 15 October
2026 study, ``torch-dftd`` 0.5.1 as ``mace_mp`` calls it), except ``functional``, which has no
default: it must match the model's training functional, and only the label table knows that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from fastmace.config import _check_choices


@dataclass(frozen=True, kw_only=True)
class D3:
    """A complete D3 dispersion correction.

    - ``functional``: the parameter set, e.g. ``"r2scan"`` (MH-1 ``matpes_r2scan``) or ``"pbe"``;
    - ``damping``: Becke–Johnson;
    - ``pair_cutoff_bohr``, ``cn_cutoff_bohr``: 40 bohr each (``mace_mp`` and ``torch-dftd``);
    - ``three_body``: the ATM term, off;
    - ``smoothing``: cutoff smoothing, none;
    - ``precision``: float64;
    - ``autoang``, ``autoev``: ``torch-dftd``'s unit constants (they differ from ASE's ``Bohr`` and
      ``Hartree`` by about 1e-7 relative), recorded with the parameters.
    """

    functional: str
    damping: Literal["bj"] = "bj"
    pair_cutoff_bohr: float = 40.0
    cn_cutoff_bohr: float = 40.0
    three_body: bool = False
    smoothing: Literal["none"] = "none"
    precision: Literal["float64", "float32"] = "float64"
    autoang: float = 0.52917726
    autoev: float = 27.21138505

    def __post_init__(self):
        _check_choices(self)
        if not isinstance(self.functional, str) or not self.functional:
            raise ValueError(f"D3.functional must name a parameter set, got {self.functional!r}")
        for name in ("pair_cutoff_bohr", "cn_cutoff_bohr", "autoang", "autoev"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                    or value <= 0:
                raise ValueError(f"D3.{name} must be positive and finite, got {value!r}")


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
    import dataclasses

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
