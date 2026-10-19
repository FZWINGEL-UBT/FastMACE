"""A fake MD run with chunked output and restart files (plan task 1b-T06), shared by the tests and
the crash test's child process.

The schedule, fake dynamics and restart cadence are those of ``tests/cpu/test_io.py`` (1a-T14):
20 steps, samples and frames every 2 steps, a restart every 4, noise drawn in batches of three steps
(so the step-8 restart falls inside a partly consumed batch). ``run`` mirrors the run loop of
``workflow.run_crystal`` with ``fastmace.io.output.ChunkedOutput`` in place of the two files: every
restart write closes the current chunks and stores their indices in the restart file; resume deletes
the later chunks and opens new ones.

Run as a script (``python chunk_run.py DIRECTORY KILL_STEP ASYNC``) it is the crash test's child: it
kills itself with SIGKILL when it reaches ``KILL_STEP``, before that step is written.
"""

from __future__ import annotations

import fastmace  # noqa: F401  (before torch: file locks, SPEC §14)

# isort: split
import os
import signal
import sys
from pathlib import Path

import numpy as np
import torch

from fastmace.engine.base import Snapshot
from fastmace.io import output as output_io
from fastmace.io import restart as restart_io
from fastmace.io import trajectory as trajectory_io
from fastmace.state import RandomStream, SimulationState

TIMESTEP_FS = 0.5
HEAT_STEPS, COOL_STEPS, PEAK_K = 10, 10, 300.0
TOTAL_STEPS = HEAT_STEPS + COOL_STEPS
SAMPLE_INTERVAL = TRAJECTORY_INTERVAL = 2
RESTART_INTERVAL = 4
NOISE_BATCH = 3
SEED = 1234
HEADER = ["step", "time_ps", "phase", "target_K", "temperature_K",
          "potential_eV_atom", "kinetic_eV_atom", "total_eV_atom"]
NPT_HEADER = HEADER + ["volume_A3_atom", "pressure_GPa"]


class Crash(RuntimeError):
    pass


def target_temperature(step):
    if step <= HEAT_STEPS:
        return PEAK_K * step / HEAT_STEPS
    return PEAK_K * (TOTAL_STEPS - step) / COOL_STEPS


def prefix(step):
    phase = "heating" if step <= HEAT_STEPS else "cooling"
    return (step, step * TIMESTEP_FS / 1000, phase, target_temperature(step))


def make_state(variable_cell=False):
    from ase.build import bulk

    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.rattle(0.05, seed=3)
    n = len(atoms)
    t = dict(dtype=torch.float64, device="cpu")
    rng = RandomStream(np.random.default_rng(SEED), torch.zeros((NOISE_BATCH, 2, n, 3), **t), NOISE_BATCH)
    state = SimulationState(
        positions=torch.tensor(atoms.positions, **t),
        velocities=torch.zeros((n, 3), **t),
        masses=torch.tensor(atoms.get_masses(), **t).reshape(-1, 1),
        cell=torch.tensor(atoms.cell.array, **t),
        species=torch.tensor(atoms.numbers, dtype=torch.int64),
        forces=torch.zeros((n, 3), **t),
        energy=torch.zeros(1, **t),
        stress=torch.zeros((3, 3), **t) if variable_cell else None,
        rng=rng,
    )
    _evaluate(state)
    return atoms, state


def _evaluate(state):
    x = state.positions
    state.forces.copy_(-0.7 * x + 0.01 * torch.roll(x, 1, 0))
    state.energy.copy_((0.35 * (x * x).sum()).reshape(1))
    if state.stress is not None:
        state.stress.copy_(1e-3 * (x.T @ x) / x.shape[0])


def _step(state):
    rng = state.rng
    if rng.cursor == len(rng.batch):
        rng.batch.copy_(torch.from_numpy(rng.generator.standard_normal(tuple(rng.batch.shape))))
        rng.cursor = 0
    xi = rng.batch[rng.cursor]
    rng.cursor += 1
    dt, sigma = 0.05, 1e-3 * (target_temperature(state.step + 1) + 1.0) ** 0.5
    state.velocities.add_(0.5 * dt * state.forces / state.masses + sigma * xi[0])
    state.positions.add_(dt * state.velocities + 0.1 * sigma * xi[1])
    if state.stress is not None:
        state.cell.mul_(1.0 + 1e-4 * state.stress.diagonal().sum())
    _evaluate(state)
    state.velocities.add_(0.5 * dt * state.forces / state.masses)
    state.step += 1


def _values(state):
    v, m = state.velocities, state.masses
    values = dict(positions=state.positions, momenta=v * m, forces=state.forces,
                  energy=state.energy.reshape(-1)[:1], kinetic=(0.5 * (m * v ** 2).sum()).reshape(1))
    if state.stress is not None:
        values.update(cell=state.cell, stress=state.stress)
    return values


def _snapshot(state):
    host, event = trajectory_io.copy_to_host(_values(state))
    return Snapshot(step=state.step, values=host, event=event)


def _sync(state):
    return {key: value.clone() for key, value in _values(state).items()}


def run(directory, *, asynchronous, resume=False, crash_in=None, kill_in=None, variable_cell=False,
        restart_every=RESTART_INTERVAL, stop_at=None):
    """Run (or resume) the fake schedule in ``directory``; returns the final state.

    ``crash_in`` raises ``Crash`` at the start of that step, ``kill_in`` sends SIGKILL to this
    process there (the crash test's child), ``stop_at`` writes a restart there and returns (a stop
    request, which is a restart point too).
    """
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    restart_path = directory / restart_io.RESTART_NAME
    atoms, state = make_state(variable_cell)
    start, indices = 0, None
    if resume and restart_path.is_file():
        stored = restart_io.read_restart(restart_path, engine="gpu")
        indices = restart_io.output_indices(stored)
        restart_io.load_state(stored, state)
        start = state.step
    header = NPT_HEADER if variable_cell else HEADER
    with output_io.ChunkedOutput(directory, atoms, header, resume_from=indices) as writer:
        queue = trajectory_io.AsyncOutput(writer)
        if not start:
            writer.write(_sync(state), sample=prefix(0), frame=True)
        for step in range(start + 1, TOTAL_STEPS + 1):
            if crash_in == step:
                raise Crash("simulated crash")
            if kill_in == step:
                os.kill(os.getpid(), signal.SIGKILL)
            _step(state)
            boundary = step in (HEAT_STEPS, TOTAL_STEPS)
            want_sample = step % SAMPLE_INTERVAL == 0 or boundary
            want_frame = step % TRAJECTORY_INTERVAL == 0 or boundary
            if asynchronous:
                queue.drain()
                if want_sample or want_frame:
                    queue.submit(_snapshot(state), sample=prefix(step) if want_sample else None, frame=want_frame)
            elif want_sample or want_frame:
                writer.write(_sync(state), sample=prefix(step) if want_sample else None, frame=want_frame)
            stop = stop_at == step
            if (restart_every and step % restart_every == 0 and step < TOTAL_STEPS) or stop:
                outputs = queue.restart_point()
                restart_io.write_restart(restart_path, state, engine="gpu", outputs=outputs)
                if stop:
                    return state
        queue.drain()
    restart_path.unlink(missing_ok=True)
    return state


if __name__ == "__main__":
    run(sys.argv[1], asynchronous=sys.argv[3] == "1", kill_in=int(sys.argv[2]))
