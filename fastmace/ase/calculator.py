"""``FastMACECalculator`` and ``fm.calc``: the ASE calculator, compiled tier (SPEC §7.4, §7.7 level 0).

Plan task 1b-T01 fixes the names; plan task 1b-T18 implements them (an ASE ``Calculator`` whose
first calculation makes a minimal plan through the planner: compiled mode, no CUDA graphs).
"""

from __future__ import annotations

from typing import Any

_TASK = "fastmace.ase.FastMACECalculator is a stub until plan task 1b-T18"


class FastMACECalculator:
    """``FastMACECalculator(model, *, head=None, rewrites="auto", precision="float32")``: energy,
    free_energy, forces and Voigt stress in ASE units; full 3D PBC only (§7.6)."""

    def __init__(self, model: Any, *, head: str | None = None, rewrites: Any = "auto", precision: str = "float32"):
        raise NotImplementedError(_TASK)


def calc(model: Any, *, head: str | None = None, **options: Any) -> FastMACECalculator:
    """``atoms.calc = fm.calc("mh1", head="matpes_r2scan")``: a ``FastMACECalculator`` for ``model``
    (label, path or URL, loaded as ``fm.load`` does); prints the compatibility-tier note once."""
    raise NotImplementedError(_TASK)
