"""``fm.md`` and ``RunResult``: MD in one call (SPEC §7.7 level 2).

Plan task 1b-T01 fixes the names and the signature; plan task 1b-T20 implements ``md``. No engine
code here: ``md`` builds a ``ForceField``, ``Workload``, integrator record, ``Output`` and
``Simulation`` from its arguments, runs it, and returns what it built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_TASK = "fastmace.md is a stub until plan task 1b-T20"


@dataclass(frozen=True)
class RunResult:
    """What ``fm.md`` returns: the report and the expert objects it built.

    - ``summary``: the one-screen report (§7.7, 1b-T15);
    - ``plan``, ``gate``: the ``Plan`` and gate records (§9.5, §12.3);
    - ``simulation``: the ``Simulation``, for restarts and the expert API;
    - ``metadata``: the dict written to ``metadata.json`` (§13);
    - ``output``: the output directory.
    """

    summary: str
    plan: Any
    gate: Any
    simulation: Any
    metadata: dict
    output: str


def md(structure: Any, *, model: Any, head: str | None = None, charge: int | None = None,
       spin_multiplicity: int | None = None, sha256: str | None = None, trusted: bool = False,
       ensemble: str = "NVT", temperature_K: float | None = None, schedule: Any = None,
       pressure_GPa: float = 0.0, cell_mode: str = "hydrostatic", timestep_fs: float = 0.5,
       steps: int | None = None, mc: Any = None, min_mc_temperature_K: float | None = None,
       dispersion: Any = None, repeat: tuple[int, int, int] = (1, 1, 1), output: Any = None,
       seed: int | None = None, check: str = "skip", integrator: Any = None) -> RunResult:
    """Run MD on ``structure`` (path or ``Atoms``) with ``model`` and return a ``RunResult``.

    Arguments as SPEC §7.7 level 2: ``model`` with ``head``, ``charge``, ``spin_multiplicity``,
    ``sha256``/``trusted`` as ``fm.load``; ``ensemble`` ``"NVT"`` (Langevin) or ``"NPT"`` (BAOAB;
    ``"NVE"`` is phase 4); ``temperature_K`` or ``schedule``; ``pressure_GPa`` and ``cell_mode``
    for NPT; ``steps`` (optional with a schedule); ``mc`` and ``min_mc_temperature_K``;
    ``dispersion`` ``None``, ``"d3bj"`` or an ``fm.D3``; ``repeat`` (never expanded silently);
    ``check`` as ``Simulation``; ``integrator`` to choose one explicitly. The defaults of
    ``output`` and ``seed`` (recorded either way) are 1b-T20's.
    """
    raise NotImplementedError(_TASK)
