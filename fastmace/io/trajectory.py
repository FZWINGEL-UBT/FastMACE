"""Thermo rows and trajectory frames, written synchronously or one step behind (SPEC §8.1, §10.6).

Phase 1a keeps the research engine's files and their order: ``temperature.csv`` (one row per
sample step) and ``md.traj`` (an ASE trajectory, one frame per frame step, with energy, forces and,
for variable-cell runs, cell and stress). A row or frame is made from one dict of output values
(``Integrator.output_values``: positions, momenta, forces, energy, kinetic; cell and stress for
variable-cell runs) as host arrays, so the synchronous path (``Engine.sync``) and the asynchronous
one (``Engine.snapshot``, written after the next step has been launched) go through the same code
and give the same bytes.

Asynchronous output: ``copy_to_host`` queues pinned device-to-host copies of the values behind the
step and records an event; the engine wraps them in a ``Snapshot``. ``AsyncOutput`` holds the
snapshots and writes them, oldest first, when the caller drains it, which the run loop does right
after launching the next step (so the copies overlap the GPU work) and before every restart file.

Chunks (plan task 1b-T06, SPEC §10.6): ``fastmace.io.output.ChunkedOutput`` makes every restart
point a chunk boundary of both files (``thermo.<n>.csv``, ``trajectory.<n>.traj``), each chunk
written by an ``OutputWriter``, and resume deletes the later chunks instead of rewriting files.
``OutputWriter`` with the 1a names and ``truncate_outputs`` (which rereads and rewrites the whole
trajectory, as the research engine does) stay for ResearchModule's ``main.py``.
"""

from __future__ import annotations

import csv
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from ase import Atoms

    from fastmace.engine.base import Snapshot

THERMO_NAME = "temperature.csv"
TRAJECTORY_NAME = "md.traj"


def copy_to_host(values: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], Any]:
    """Copy device tensors to new host tensors without draining the GPU.

    On CUDA the host tensors are pinned, the copies are non-blocking on the current stream, and the
    returned CUDA event is recorded behind them: wait for it before reading. On the CPU the copies
    are plain and the event is ``None``. The copies do not change the device tensors.
    """
    host = {}
    cuda = any(value.is_cuda for value in values.values())
    for key, value in values.items():
        host[key] = torch.empty(value.shape, dtype=value.dtype, pin_memory=cuda)
        host[key].copy_(value, non_blocking=cuda)
    event = None
    if cuda:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
    return host, event


def wait(snapshot: Snapshot) -> Mapping[str, torch.Tensor]:
    """Block until a snapshot's copies are done; return its host values."""
    if snapshot.event is not None:
        snapshot.event.synchronize()
    return snapshot.values


def _host(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.numpy()
    return np.asarray(value)


def observables(values: Mapping[str, Any], natoms: int, variable_cell: bool = False) -> tuple[float, ...]:
    """Temperature (K), potential and kinetic energy per atom (eV); variable cell: volume per atom
    (Å³) and the instantaneous pressure with its kinetic part (GPa).

    ``values`` are host output values. Degrees of freedom 3N, as ``Atoms.get_temperature`` without
    constraints; the volume is ``ase.cell.Cell.volume`` of the snapshot cell.
    """
    from ase import units
    from ase.cell import Cell
    kinetic = float(_host(values["kinetic"])[0])
    energy = float(_host(values["energy"])[0])
    result = (2 * kinetic / (3 * natoms * units.kB), energy / natoms, kinetic / natoms)
    if variable_cell:
        volume = float(Cell(_host(values["cell"])).volume)
        stress = _host(values["stress"])
        pressure = (2.0 * kinetic - volume * float(sum(stress[i][i] for i in range(3)))) / (3.0 * volume)
        result += (volume / natoms, pressure / units.GPa)
    return result


class OutputWriter:
    """The thermo CSV and the ASE trajectory of one run, in ``directory``.

    ``atoms`` is the run's ``Atoms`` (numbers, masses, pbc): each frame puts the step's cell,
    positions, momenta and a single-point calculator with energy, forces and stress into it and
    writes it, as the research engine does. ``header`` is written first unless ``append`` (a
    resumed run, after ``truncate_outputs``). ``thermo_name`` and ``trajectory_name`` are the 1a
    files by default; ``io.output.ChunkedOutput`` opens one writer per chunk with the chunk's names.
    """

    def __init__(self, directory: str | Path, atoms: Atoms, header: Sequence[str], append: bool = False, *,
                 thermo_name: str = THERMO_NAME, trajectory_name: str = TRAJECTORY_NAME):
        from ase.io.trajectory import Trajectory
        directory = Path(directory)
        mode = "a" if append else "w"
        self.atoms = atoms
        self._trajectory = Trajectory(str(directory / trajectory_name), mode, atoms)
        self._handle = (directory / thermo_name).open(mode, newline="")
        self._csv = csv.writer(self._handle)
        if not append:
            self._csv.writerow(list(header))
            self._handle.flush()  # a chunk opened at a restart point is complete on disk

    def write(self, values: Mapping[str, Any], sample: Sequence[Any] | None = None,
              frame: bool = False) -> tuple[Any, ...] | None:
        """Write one step's row and/or frame from host output values.

        ``sample`` is the row's leading columns (step, time in ps, phase, target temperature), or
        ``None`` for no row; the row adds temperature, potential, kinetic and total energy per atom
        and, for variable-cell values, volume per atom and pressure. Returns the row. Nonfinite
        observables raise ``RuntimeError`` before anything is written.
        """
        values = {key: _host(value) for key, value in values.items()}
        variable_cell = "cell" in values and "stress" in values
        row = None
        if sample is not None:
            result = observables(values, len(self.atoms), variable_cell)
            if not np.isfinite(result).all():
                raise RuntimeError(f"Nonfinite MD observables at step {sample[0]}")
            temperature, epot, ekin = result[:3]
            row = (*sample, temperature, epot, ekin, epot + ekin, *result[3:])
            self._csv.writerow(row)
            self._handle.flush()
        if frame:
            self._write_frame(values, variable_cell)
        return row

    def _write_frame(self, values, variable_cell):
        from ase.calculators.singlepoint import SinglePointCalculator
        atoms = self.atoms
        extra = {}
        if variable_cell:
            from ase.stress import full_3x3_to_voigt_6_stress
            atoms.set_cell(values["cell"])
            extra["stress"] = full_3x3_to_voigt_6_stress(values["stress"])
        atoms.set_positions(values["positions"])
        atoms.set_momenta(values["momenta"])
        atoms.calc = SinglePointCalculator(atoms, energy=float(values["energy"][0]), forces=values["forces"],
                                           **extra)
        self._trajectory.write(atoms)

    def flush(self) -> None:
        """Flush the thermo file (trajectory frames are on disk once written)."""
        self._handle.flush()

    def close(self) -> None:
        self._trajectory.close()
        self._handle.close()

    def __enter__(self) -> OutputWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class AsyncOutput:
    """Snapshots waiting to be written, oldest first (``--async-output``).

    The run loop calls ``drain`` after launching each step, then ``submit`` for a sample or frame
    step, and ``drain`` again before a restart file and at the end. Values, order and files are those
    of synchronous writing.
    """

    def __init__(self, writer: OutputWriter):
        self.writer = writer
        self._pending: deque = deque()

    def submit(self, snapshot: Snapshot, sample: Sequence[Any] | None = None, frame: bool = False) -> None:
        self._pending.append((snapshot, sample, frame))

    def drain(self) -> list[tuple[int, tuple[Any, ...] | None]]:
        """Wait for and write every queued snapshot; returns ``(step, row)`` for each."""
        written = []
        while self._pending:
            snapshot, sample, frame = self._pending.popleft()
            written.append((snapshot.step, self.writer.write(wait(snapshot), sample, frame)))
        return written

    def restart_point(self) -> dict[str, int]:
        """Drain, then close the writer's chunk (``ChunkedOutput.restart_point``); returns its indices."""
        self.drain()
        return self.writer.restart_point()

    def __len__(self) -> int:
        return len(self._pending)


def output_steps(total_steps: int, interval: int, heat_steps: int, hold_steps: int = 0) -> list[int]:
    """Steps at which a sample (or trajectory frame) is written: 0, every interval, the phase ends."""
    steps = {0, heat_steps, heat_steps + hold_steps, total_steps}
    steps.update(range(interval, total_steps + 1, interval))
    return sorted(steps)


def truncate_outputs(directory: str | Path, step: int, frame_steps: Iterable[int]) -> list[tuple[Any, ...]]:
    """Drop thermo rows and trajectory frames written after the restart ``step`` (they are rewritten).

    ``frame_steps`` are the steps at which the run writes frames (``output_steps``); the frames of
    those up to ``step`` are kept. Rewrites both files, as the research engine does (1b replaces this
    with chunk deletion). Returns the kept rows parsed: step (int), time (float), phase (str), then
    floats.
    """
    from ase.io import read
    from ase.io.trajectory import Trajectory
    directory = Path(directory)
    csv_path = directory / THERMO_NAME
    with csv_path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    header, data = rows[0], [r for r in rows[1:] if int(r[0]) <= step]
    with csv_path.open("w", newline="") as handle:
        csv.writer(handle).writerows([header] + data)
    keep = sum(1 for s in frame_steps if s <= step)
    traj_path = directory / TRAJECTORY_NAME
    frames = read(traj_path, index=f":{keep}")
    with Trajectory(str(traj_path), "w") as trajectory:
        for frame in frames:
            trajectory.write(frame)
    return [(int(r[0]), float(r[1]), r[2], *map(float, r[3:])) for r in data]
