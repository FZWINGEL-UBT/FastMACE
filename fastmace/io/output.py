"""Output: where a ``Simulation`` writes and how often (SPEC §7.2, §10.6).

Plan task 1b-T01 fixes the record (``fm.Output``); plan task 1b-T06 adds the chunked writer
here: ``trajectory.<n>.traj`` and ``thermo.<n>.csv``, with chunk boundaries exactly at restart
points, so that a resume deletes later chunks and never rewrites a file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Output:
    """Output directory and cadences, in MD steps (defaults: the production settings of §7.7).

    ``trajectory_every`` and ``restart_every`` may be ``None`` (no trajectory, no restart files);
    thermo rows are always written. ``directory`` is kept as given, as a string.
    """

    directory: str
    trajectory_every: int | None = 20
    thermo_every: int = 20
    restart_every: int | None = 10_000

    def __post_init__(self):
        directory = os.fspath(self.directory) if isinstance(self.directory, os.PathLike) else self.directory
        if not isinstance(directory, str) or not directory:
            raise ValueError(f"Output.directory must be a non-empty path, got {self.directory!r}")
        object.__setattr__(self, "directory", directory)
        for name, optional in (("trajectory_every", True), ("thermo_every", False), ("restart_every", True)):
            value = getattr(self, name)
            if value is None and optional:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                allowed = "a positive int or None" if optional else "a positive int"
                raise ValueError(f"Output.{name} must be {allowed} (steps), got {value!r}")
