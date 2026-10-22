"""Output: where a ``Simulation`` writes and how often, and the chunked writer (SPEC §7.2, §10.6).

``Output`` is the record (plan task 1b-T01). ``ChunkedOutput`` (plan task 1b-T06) writes the
thermo rows and trajectory frames of a run as chunks, ``thermo.<n>.csv`` and
``trajectory.<n>.traj`` for n = 0, 1, ..., with a chunk boundary exactly at every restart point
(SPEC §10.6 "Output files at restart points"):

- ``restart_point()`` closes the current chunks, fsyncs them and their directory (plan task
  2-T10; rows and frames in between are flushed, not fsynced) and opens the next pair; it
  returns the indices the restart file stores (``io.restart.OUTPUT_KEYS``): ``output_chunk``, the last closed chunk,
  and ``output_rows`` and ``output_frames``, the rows and frames in chunks 0 to ``output_chunk``.
  Call it after draining the output queue (``AsyncOutput.restart_point`` does both), then write
  the restart file. Every restart write is a chunk boundary, a stop request's included.
- Resume (``resume_from=`` those indices) checks the chunks up to ``output_chunk`` (present, with
  the stored numbers of rows and frames) and that every later chunk can be deleted, before it
  touches anything; otherwise it raises ``RestartMismatch``. Then it deletes the later chunks and
  opens chunk ``output_chunk + 1``. Nothing is ever rewritten: the research engine's resume reread
  and rewrote ``md.traj`` (``workflow.truncate_outputs``, up to 770 MB on the NAS).

Rows and frames written after the last restart point (by the asynchronous writer, or before a
crash) are thus in later chunks only. A crash between ``restart_point`` and the restart write
leaves the older restart file, and resume from it deletes the newer chunks as well.

Every chunk is complete on its own: each thermo chunk starts with the header, and a chunk without
rows or frames still has both files (an empty trajectory reads as no frames). ``read_thermo`` and
``read_frames`` concatenate the chunks in order; ``ase.io.read(path, index=":")`` reads one.
This module imports neither torch nor ASE at import time (``fm.Output`` is a light name).
"""

from __future__ import annotations

import csv
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastmace.errors import RestartMismatch
from fastmace.io import durable

if TYPE_CHECKING:
    from ase import Atoms

THERMO_PATTERN = "thermo.{}.csv"
TRAJECTORY_PATTERN = "trajectory.{}.traj"
_CHUNK = re.compile(r"^(?:thermo\.(\d+)\.csv|trajectory\.(\d+)\.traj)$")
# The keys of the indices ``restart_point`` returns: io.restart.OUTPUT_KEYS (not imported, it needs torch).
_KEYS = ("output_chunk", "output_rows", "output_frames")


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


def chunk_paths(directory: str | Path, index: int) -> tuple[Path, Path]:
    """The thermo and trajectory files of chunk ``index``."""
    directory = Path(directory)
    return directory / THERMO_PATTERN.format(index), directory / TRAJECTORY_PATTERN.format(index)


def chunk_files(directory: str | Path) -> dict[int, list[Path]]:
    """Every chunk file in ``directory``, by chunk index (sorted)."""
    found: dict[int, list[Path]] = {}
    for path in Path(directory).iterdir():
        match = _CHUNK.match(path.name)
        if match:
            found.setdefault(int(match.group(1) or match.group(2)), []).append(path)
    return {index: sorted(found[index]) for index in sorted(found)}


def chunk_indices(directory: str | Path) -> list[int]:
    """The indices of the chunks in ``directory`` (either file present), in order."""
    return list(chunk_files(directory))


def _thermo_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _frame_count(path: Path) -> int:
    from ase.io.trajectory import Trajectory
    with Trajectory(str(path)) as trajectory:
        return len(trajectory)


def read_thermo(directory: str | Path) -> tuple[list[str], list[list[str]]]:
    """The header and every thermo row over all chunks, in order (rows as strings)."""
    header: list[str] = []
    rows: list[list[str]] = []
    for index in chunk_indices(directory):
        path = chunk_paths(directory, index)[0]
        if path.is_file():
            chunk_header, chunk_rows = _thermo_rows(path)
            header = header or chunk_header
            rows += chunk_rows
    return header, rows


def read_frames(directory: str | Path) -> list[Atoms]:
    """Every trajectory frame over all chunks, in order."""
    from ase.io import read
    frames = []
    for index in chunk_indices(directory):
        path = chunk_paths(directory, index)[1]
        if path.is_file():
            frames += read(path, index=":")
    return frames


def _check_resume(directory: Path, indices: Mapping[str, int] | None) -> list[Path]:
    """The chunk files to delete on resume; ``RestartMismatch`` if the outputs do not fit the restart."""
    if not indices or any(key not in indices for key in _KEYS):
        raise RestartMismatch(
            f"the restart in {directory} has no output chunk indices",
            why="It was written without chunked output (a phase-1a or main.py file), so the outputs after "
                "its step cannot be found without rewriting files.",
            next_step="Resume it with the run loop that wrote it, or start a new run in a new directory.",
            record=str(directory))
    last, rows, frames = (int(indices[key]) for key in _KEYS)
    counted_rows = counted_frames = 0
    for index in range(last + 1):
        thermo, trajectory = chunk_paths(directory, index)
        for path in (thermo, trajectory):
            if not path.is_file():
                raise RestartMismatch(
                    f"output chunk {path.name} is missing in {directory}",
                    why=f"The restart's outputs end with chunk {last}; without every chunk up to it the "
                        "trajectory and thermo files would have a gap.",
                    next_step="Restore the missing chunk, or start a new run in a new directory.",
                    record=str(directory))
        counted_rows += len(_thermo_rows(thermo)[1])
        counted_frames += _frame_count(trajectory)
    if (counted_rows, counted_frames) != (rows, frames):
        raise RestartMismatch(
            f"output chunks 0-{last} in {directory} hold {counted_rows} rows and {counted_frames} frames; "
            f"the restart expects {rows} rows and {frames} frames",
            why="The chunks up to the restart point were changed after the restart was written.",
            next_step="Restore the chunks, or start a new run in a new directory.",
            record=str(directory))
    later = [path for index, paths in chunk_files(directory).items() if index > last for path in paths]
    # Deleting a file needs a writable directory (POSIX) and a file without the read-only attribute
    # (Windows); creating the next chunk needs the directory either way.
    blocked = [path for path in later if not os.access(path, os.W_OK)]
    if not os.access(directory, os.W_OK | os.X_OK):
        blocked = [directory]
    if blocked:
        raise RestartMismatch(
            f"cannot delete the output chunks after the restart in {directory}",
            why=f"The outputs are ahead of the restart (chunk {last}), and {', '.join(map(str, blocked))} "
                "is read-only, so they cannot be truncated; nothing was changed.",
            next_step="Make the run directory writable, or copy it and resume the copy.",
            record=str(directory))
    return later


class ChunkedOutput:
    """The thermo rows and trajectory frames of one run, in chunks that end at restart points.

    ``atoms`` and ``header`` are ``io.trajectory.OutputWriter``'s, which writes each chunk.
    ``resume_from`` is ``io.restart.output_indices`` of the restart file a resumed run starts from;
    ``None`` starts a new run, which refuses a directory that already holds chunks
    (``FileExistsError``). ``write`` takes ``OutputWriter.write``'s arguments, so ``AsyncOutput``
    works on either writer. ``degrees`` is the temperature's degrees of freedom (``OutputWriter``).
    """

    def __init__(self, directory: str | Path, atoms: Atoms, header: Sequence[str], *,
                 resume_from: Mapping[str, int] | None = None, degrees: int | None = None):
        self.directory = Path(directory)
        self.degrees = degrees
        self.atoms = atoms
        self.header = list(header)
        self._writer = None
        if resume_from is None:
            existing = chunk_files(self.directory)
            if existing:
                names = ", ".join(path.name for paths in existing.values() for path in paths)
                raise FileExistsError(f"{self.directory} already holds output chunks ({names}); "
                                      "resume the run or use a new directory")
            self.chunk, self.rows, self.frames = 0, 0, 0
        else:
            for path in _check_resume(self.directory, resume_from):
                path.unlink()
            self.chunk = int(resume_from["output_chunk"]) + 1
            self.rows, self.frames = int(resume_from["output_rows"]), int(resume_from["output_frames"])
        self._open()

    def _open(self) -> None:
        from fastmace.io.trajectory import OutputWriter
        thermo, trajectory = chunk_paths(self.directory, self.chunk)
        self._writer = OutputWriter(self.directory, self.atoms, self.header, thermo_name=thermo.name,
                                    trajectory_name=trajectory.name, degrees=self.degrees)

    def write(self, values: Mapping[str, Any], sample: Sequence[Any] | None = None,
              frame: bool = False, *, tail: Sequence[Any] = ()) -> tuple[Any, ...] | None:
        """``OutputWriter.write`` into the current chunk; counts the row and the frame."""
        row = self._writer.write(values, sample, frame, tail=tail)
        self.rows += row is not None
        self.frames += bool(frame)
        return row

    def restart_point(self) -> dict[str, int]:
        """Close the current chunk and open the next; returns the indices for the restart file.

        Call it after the output queue is drained; the result is ``write_restart``'s ``outputs``.
        """
        self._close_durably()
        indices = dict(output_chunk=self.chunk, output_rows=self.rows, output_frames=self.frames)
        self.chunk += 1
        self._open()
        return indices

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._close_durably()
            self._writer = None

    def _close_durably(self) -> None:
        """Close the current chunk, then fsync both files and the directory (``io.durable``)."""
        self._writer.close()
        for path in chunk_paths(self.directory, self.chunk):
            durable.fsync_file(path)
        durable.fsync_directory(self.directory)

    def __enter__(self) -> ChunkedOutput:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
