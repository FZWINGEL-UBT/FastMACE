"""The D3 dispersion record, ``fm.D3`` (SPEC §11.2, §7.7 "Dispersion").

Plan task 1b-T01 fixes the record; 1b-T20 adds the ``"d3bj"`` expansion from the head's training
functional. The GPU kernel is phase 3b: until then a complete record is refused at plan time
("D3: phase 3b"), with the record shown.

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
