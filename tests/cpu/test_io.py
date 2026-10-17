"""Output files and the minimal restart (plan task 1a-T14; SPEC §8.1 "async output", §10.6).

Ported from the research engine's ``tests/core/test_workflow.py`` (``GPUEngineTests``) at tag
``research-engine-baseline-20261017``, with their schedule and expected values: 20 steps (heating
to step 10, cooling to step 20), samples and frames every 2 steps, a restart file every 4, a crash
in step 10, so the last restart file is the one of step 8. The tiny MACE model is replaced by a
deterministic fake dynamics on a ``SimulationState`` that draws its noise in batches of three
steps, so the step-8 restart falls inside a partly consumed batch (cursor 2 of 3).

``run`` mirrors the loop of the research engine's ``workflow.run_crystal``: a sample or frame step
is written at once (synchronous) or queued as a snapshot and written after the next step
(asynchronous); restart steps drain the queue, flush the files and write the restart file; resume
reads it, truncates the outputs to its step and appends.
"""

from __future__ import annotations

import csv
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch
from ase.io import read

from _support.structures import nacl
from fastmace.engine.base import Snapshot
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


# -- the schedule of workflow.py (target_temperature, schedule_phase) --

def target_temperature(step):
    if step <= HEAT_STEPS:
        return PEAK_K * step / HEAT_STEPS
    return PEAK_K * (TOTAL_STEPS - step) / COOL_STEPS


def phase(step):
    return "heating" if step <= HEAT_STEPS else "cooling"


def prefix(step):
    return (step, step * TIMESTEP_FS / 1000, phase(step), target_temperature(step))


# -- a fake integrator and force field on the 1a SimulationState --

def make_state(device="cpu", variable_cell=False):
    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    n = len(atoms)
    t = dict(dtype=torch.float64, device=device)
    rng = RandomStream(np.random.default_rng(SEED), torch.zeros((NOISE_BATCH, 2, n, 3), **t), NOISE_BATCH)
    state = SimulationState(
        positions=torch.tensor(atoms.positions, **t),
        velocities=torch.zeros((n, 3), **t),
        masses=torch.tensor(atoms.get_masses(), **t).reshape(-1, 1),
        cell=torch.tensor(atoms.cell.array, **t),
        species=torch.tensor(atoms.numbers, dtype=torch.int64, device=device),
        forces=torch.zeros((n, 3), **t),
        energy=torch.zeros(1, **t),
        stress=torch.zeros((3, 3), **t) if variable_cell else None,
        rng=rng,
    )
    evaluate(state)
    return atoms, state


def evaluate(state):
    """A harmonic well around the origin plus a pair term, so forces change every step."""
    x = state.positions
    state.forces.copy_(-0.7 * x + 0.01 * torch.roll(x, 1, 0))
    state.energy.copy_((0.35 * (x * x).sum()).reshape(1))
    if state.stress is not None:
        state.stress.copy_(1e-3 * (x.T @ x) / x.shape[0])


def fake_step(state, target_K, integrator):
    rng = state.rng
    if rng.cursor == len(rng.batch):
        rng.batch.copy_(torch.from_numpy(rng.generator.standard_normal(tuple(rng.batch.shape))))
        rng.cursor = 0
    xi = rng.batch[rng.cursor]
    rng.cursor += 1
    dt, sigma = 0.05, 1e-3 * (target_K + 1.0) ** 0.5
    state.velocities.add_(0.5 * dt * state.forces / state.masses + sigma * xi[0])
    state.positions.add_(dt * state.velocities + 0.1 * sigma * xi[1])
    if state.stress is not None:
        state.cell.mul_(1.0 + integrator.p_eps)
        integrator.p_eps.add_(1e-4 * state.stress.diagonal().sum())
    evaluate(state)
    state.velocities.add_(0.5 * dt * state.forces / state.masses)
    state.step += 1


class FakeIntegrator:
    """Holds a barostat-like scalar in variable-cell runs (state_dict, as BAOAB's p_eps)."""

    def __init__(self, state):
        self.p_eps = torch.zeros(1, dtype=torch.float64, device=state.positions.device)

    def state_dict(self):
        return {"p_eps": self.p_eps.cpu().numpy(), "force_eps": (2.0 * self.p_eps).cpu().numpy()}

    def load_state_dict(self, state_dict):
        if "p_eps" in state_dict:
            self.p_eps.copy_(torch.from_numpy(np.asarray(state_dict["p_eps"])))


def output_values(state):
    """The values of the research engine's ``GPULangevin._snapshot_values`` (BAOAB adds cell, stress)."""
    v, m = state.velocities, state.masses
    values = dict(positions=state.positions, momenta=v * m, forces=state.forces,
                  energy=state.energy.reshape(-1)[:1], kinetic=(0.5 * (m * v ** 2).sum()).reshape(1))
    if state.stress is not None:
        values.update(cell=state.cell, stress=state.stress)
    return values


def snapshot(state):
    host, event = trajectory_io.copy_to_host(output_values(state))
    return Snapshot(step=state.step, values=host, event=event)


def sync(state):
    """Synchronous output values (``Engine.sync``): host copies, the GPU drained."""
    return {key: value.cpu() for key, value in output_values(state).items()}


def run(directory, *, asynchronous, resume=False, crash_in=None, variable_cell=False, device="cpu"):
    """The output and restart part of ``workflow.run_crystal``; returns the final state."""
    directory.mkdir(exist_ok=True)
    restart_path = directory / restart_io.RESTART_NAME
    atoms, state = make_state(device, variable_cell)
    integrator = FakeIntegrator(state)
    start = 0
    if resume and restart_path.is_file():
        stored = restart_io.read_restart(restart_path, engine="gpu")
        start = int(stored["step"])
        trajectory_io.truncate_outputs(directory, start, frame_steps())
        restart_io.load_state(stored, state, integrator)
    mode_append = bool(start)
    with trajectory_io.OutputWriter(directory, atoms, header=NPT_HEADER if variable_cell else HEADER,
                                    append=mode_append) as writer:
        queue = trajectory_io.AsyncOutput(writer)
        if not start:
            writer.write(sync(state), sample=prefix(0), frame=True)
        for step in range(start + 1, TOTAL_STEPS + 1):
            if crash_in == step:
                raise Crash("simulated crash")
            fake_step(state, target_temperature(step), integrator)
            boundary = step in (HEAT_STEPS, TOTAL_STEPS)
            want_sample = step % SAMPLE_INTERVAL == 0 or boundary
            want_frame = step % TRAJECTORY_INTERVAL == 0 or boundary
            if asynchronous:
                queue.drain()
                if want_sample or want_frame:
                    queue.submit(snapshot(state), sample=prefix(step) if want_sample else None, frame=want_frame)
            elif want_sample or want_frame:
                writer.write(sync(state), sample=prefix(step) if want_sample else None, frame=want_frame)
            if step % RESTART_INTERVAL == 0 and step < TOTAL_STEPS:
                queue.drain()
                writer.flush()
                restart_io.write_restart(restart_path, state, engine="gpu", variable_cell=variable_cell,
                                         integrator_state=integrator.state_dict() if variable_cell else None)
        queue.drain()
    if restart_path.is_file():
        restart_path.unlink()
    return atoms, state


def frame_steps():
    return trajectory_io.output_steps(TOTAL_STEPS, TRAJECTORY_INTERVAL, HEAT_STEPS)


def file_bytes(directory):
    names = (trajectory_io.THERMO_NAME, trajectory_io.TRAJECTORY_NAME)
    return {name: (directory / name).read_bytes() for name in names}


def assert_states_equal(a, b):
    for name in ("positions", "velocities", "masses", "cell", "species", "forces", "energy"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert (a.stress is None) == (b.stress is None)
    if a.stress is not None:
        assert torch.equal(a.stress, b.stress)
    assert a.step == b.step
    assert a.rng.cursor == b.rng.cursor
    assert torch.equal(a.rng.batch, b.rng.batch)
    assert a.rng.generator.bit_generator.state == b.rng.generator.bit_generator.state


# -- output files --

def test_gpu_engine_outputs(tmp_path):
    """Samples every 2 steps from 0 to 20 and 11 frames that carry energy and forces."""
    run(tmp_path / "NaCl", asynchronous=False)
    with (tmp_path / "NaCl" / "temperature.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [int(r["step"]) for r in rows] == list(range(0, 21, 2))
    assert list(rows[0]) == HEADER
    assert [r["phase"] for r in rows] == ["heating"] * 6 + ["cooling"] * 5
    frames = read(tmp_path / "NaCl" / "md.traj", index=":")
    assert len(frames) == 11
    assert np.isfinite(frames[-1].get_forces()).all()
    assert np.isfinite(frames[-1].get_potential_energy())
    assert not (tmp_path / "NaCl" / "restart.npz").exists()  # removed when complete


@pytest.mark.parametrize("variable_cell", [False, True], ids=["nvt", "npt"])
def test_async_output_writes_the_same_files(tmp_path, variable_cell):
    """Queued snapshots written one step late give byte-identical files to synchronous writing."""
    sync_dir, async_dir = tmp_path / "sync", tmp_path / "async"
    _, a = run(sync_dir, asynchronous=False, variable_cell=variable_cell)
    _, b = run(async_dir, asynchronous=True, variable_cell=variable_cell)
    assert_states_equal(a, b)
    assert file_bytes(async_dir) == file_bytes(sync_dir)
    frames = read(async_dir / "md.traj", index=":")
    assert len(frames) == 11
    if variable_cell:
        assert frames[-1].get_stress().shape == (6,)
        assert not np.allclose(frames[-1].cell.array, frames[0].cell.array)


def test_frames_hold_the_values_of_their_step(tmp_path):
    """A snapshot copies the values; later steps do not change a queued frame."""
    atoms, state = make_state()
    integrator = FakeIntegrator(state)
    fake_step(state, 100.0, integrator)
    expected = {key: value.clone() for key, value in output_values(state).items()}
    with trajectory_io.OutputWriter(tmp_path, atoms, header=HEADER) as writer:
        queue = trajectory_io.AsyncOutput(writer)
        queue.submit(snapshot(state), sample=prefix(1), frame=True)
        fake_step(state, 100.0, integrator)
        fake_step(state, 100.0, integrator)
        assert len(queue) == 1
        written = queue.drain()
        assert len(queue) == 0
    assert [step for step, _ in written] == [1]
    frame = read(tmp_path / "md.traj")
    np.testing.assert_array_equal(frame.positions, expected["positions"].numpy())
    np.testing.assert_array_equal(frame.get_momenta(), expected["momenta"].numpy())
    np.testing.assert_array_equal(frame.get_forces(), expected["forces"].numpy())
    assert frame.get_potential_energy() == float(expected["energy"][0])


def test_observables_match_the_research_engine():
    """Temperature, potential and kinetic energy per atom; NPT adds volume per atom and pressure."""
    from ase import units
    atoms, state = make_state(variable_cell=True)
    fake_step(state, 300.0, FakeIntegrator(state))
    values = sync(state)
    n = len(atoms)
    kinetic = 0.5 * float((state.masses * state.velocities ** 2).sum())
    energy = float(state.energy[0])
    expected = (2 * kinetic / (3 * n * units.kB), energy / n, kinetic / n)
    assert trajectory_io.observables(values, n, variable_cell=False) == expected
    volume = abs(np.linalg.det(state.cell.numpy()))
    stress = state.stress.numpy()
    pressure = (2.0 * kinetic - volume * float(sum(stress[i][i] for i in range(3)))) / (3.0 * volume)
    got = trajectory_io.observables(values, n, variable_cell=True)
    assert got[:3] == expected
    assert got[3:] == pytest.approx((volume / n, pressure / units.GPa), rel=1e-14)
    assert all(type(value) is float for value in got)


def test_nonfinite_observables_stop_the_run(tmp_path):
    atoms, state = make_state()
    values = sync(state)
    values["energy"] = torch.tensor([float("nan")], dtype=torch.float64)
    with trajectory_io.OutputWriter(tmp_path, atoms, header=HEADER) as writer:
        with pytest.raises(RuntimeError, match="Nonfinite MD observables at step 0"):
            writer.write(values, sample=prefix(0))


def test_output_steps():
    """0, every interval, and the ends of the phases (workflow.output_steps)."""
    assert trajectory_io.output_steps(20, 2, 10) == list(range(0, 21, 2))
    assert trajectory_io.output_steps(20, 4, 10) == [0, 4, 8, 10, 12, 16, 20]
    assert trajectory_io.output_steps(30, 4, 10, hold_steps=5) == [0, 4, 8, 10, 12, 15, 16, 20, 24, 28, 30]


# -- truncation at resume --

def test_truncate_outputs_drops_rows_and_frames_after_the_restart_step(tmp_path):
    """Rows with step > restart step and frames beyond the frames written up to it are removed."""
    run(tmp_path, asynchronous=False)
    full_rows = (tmp_path / "temperature.csv").read_text().splitlines()
    full_frames = read(tmp_path / "md.traj", index=":")
    rows = trajectory_io.truncate_outputs(tmp_path, 8, frame_steps())
    kept = (tmp_path / "temperature.csv").read_text().splitlines()
    assert kept == full_rows[:6]  # header and steps 0, 2, 4, 6, 8
    assert [r[0] for r in rows] == [0, 2, 4, 6, 8]
    assert rows[1] == (2, 0.001, "heating", 60.0, *map(float, full_rows[2].split(",")[4:]))
    frames = read(tmp_path / "md.traj", index=":")
    assert len(frames) == 5
    for new, old in zip(frames, full_frames[:5]):
        np.testing.assert_array_equal(new.positions, old.positions)
        np.testing.assert_array_equal(new.get_forces(), old.get_forces())


def test_truncate_outputs_counts_phase_end_frames(tmp_path):
    """With interval 4 the heating end (step 10) is an extra frame: step 12 keeps 0, 4, 8, 10, 12."""
    atoms, state = make_state()
    steps = trajectory_io.output_steps(TOTAL_STEPS, 4, HEAT_STEPS)
    with trajectory_io.OutputWriter(tmp_path, atoms, header=HEADER) as writer:
        for step in steps:
            state.positions.add_(0.01)
            writer.write(sync(state), sample=prefix(step), frame=True)
    trajectory_io.truncate_outputs(tmp_path, 12, steps)
    frames = read(tmp_path / "md.traj", index=":")
    assert len(frames) == 5
    with (tmp_path / "temperature.csv").open() as handle:
        assert [int(r["step"]) for r in csv.DictReader(handle)] == [0, 4, 8, 10, 12]


# -- the restart file --

def test_restart_round_trip_of_the_1a_state(tmp_path):
    """write → read → load gives the state bit for bit, including a partly used noise batch."""
    _, state = make_state(variable_cell=True)
    integrator = FakeIntegrator(state)
    for step in range(1, 6):
        fake_step(state, 200.0, integrator)
    assert 0 < state.rng.cursor < NOISE_BATCH
    path = tmp_path / "restart.npz"
    restart_io.write_restart(path, state, engine="gpu", variable_cell=True, integrator_state=integrator.state_dict())
    stored = restart_io.read_restart(path)
    _, fresh = make_state(variable_cell=True)
    fresh_integrator = FakeIntegrator(fresh)
    restart_io.load_state(stored, fresh, fresh_integrator)
    assert_states_equal(state, fresh)
    assert torch.equal(fresh_integrator.p_eps, integrator.p_eps)
    assert restart_io.integrator_state(stored).keys() == {"p_eps", "force_eps"}
    # The next steps continue identically (the generator and the drawn batch both came across).
    for _ in range(4):
        fake_step(state, 200.0, integrator)
        fake_step(fresh, 200.0, fresh_integrator)
    assert_states_equal(state, fresh)
    # save → load → save gives identical contents (SPEC §10.6 test; np.savez stamps the zip
    # entries with the time of writing, so the bytes are compared per entry).
    again = tmp_path / "again.npz"
    restart_io.write_restart(path, fresh, engine="gpu", variable_cell=True,
                             integrator_state=fresh_integrator.state_dict())
    restart_io.load_state(restart_io.read_restart(path), state, integrator)
    restart_io.write_restart(again, state, engine="gpu", variable_cell=True, integrator_state=integrator.state_dict())
    first, second = restart_io.read_restart(path), restart_io.read_restart(again)
    assert first.keys() == second.keys()
    for key in first:
        assert first[key].dtype == second[key].dtype and first[key].tobytes() == second[key].tobytes(), key


def test_restart_keys_are_the_research_engine_keys(tmp_path):
    """Today's keys and meanings (workflow.write_restart, GPUEngine.state), so either side reads the file."""
    _, state = make_state(variable_cell=True)
    integrator = FakeIntegrator(state)
    fake_step(state, 300.0, integrator)
    path = tmp_path / "restart.npz"
    restart_io.write_restart(path, state, engine="gpu", variable_cell=True, integrator_state=integrator.state_dict())
    with np.load(path, allow_pickle=False) as stored:
        keys = set(stored.files)
        assert {"step", "positions", "momenta", "cell", "engine", "rng_state", "noise", "noise_next",
                "forces", "energy", "cell_state", "p_eps", "force_eps", "stress"} <= keys
        assert int(stored["step"]) == 1 and str(stored["engine"]) == "gpu"
        np.testing.assert_array_equal(stored["momenta"], (state.velocities * state.masses).numpy())
        np.testing.assert_array_equal(stored["noise"], state.rng.batch.numpy())
        assert int(stored["noise_next"]) == state.rng.cursor
        assert json.loads(str(stored["rng_state"])) == state.rng.generator.bit_generator.state
        np.testing.assert_array_equal(stored["cell_state"], state.cell.numpy())
        assert stored["energy"].shape == (1,)
    _, nvt = make_state()
    restart_io.write_restart(path, nvt, engine="gpu")
    with np.load(path, allow_pickle=False) as stored:
        assert not {"cell_state", "stress", "p_eps"} & set(stored.files)


def test_a_research_engine_restart_file_loads(tmp_path):
    """A file without velocities (written by the research engine) loads them as momenta / masses."""
    _, state = make_state()
    fake_step(state, 300.0, FakeIntegrator(state))
    path = tmp_path / "restart.npz"
    momenta = (state.velocities * state.masses).numpy()
    np.savez(path, step=1, positions=state.positions.numpy(), momenta=momenta, cell=state.cell.numpy(),
             engine="gpu", rng_state=json.dumps(state.rng.generator.bit_generator.state),
             noise=state.rng.batch.numpy(), noise_next=state.rng.cursor,
             forces=state.forces.numpy(), energy=state.energy.numpy())
    _, fresh = make_state()
    restart_io.load_state(restart_io.read_restart(path, engine="gpu"), fresh)
    masses = fresh.masses.numpy().reshape(-1)
    np.testing.assert_array_equal(fresh.velocities.numpy(), momenta / masses[:, np.newaxis])
    for name in ("positions", "forces", "energy", "cell"):
        assert torch.equal(getattr(fresh, name), getattr(state, name)), name
    assert fresh.step == 1 and fresh.rng.cursor == state.rng.cursor
    assert fresh.rng.generator.bit_generator.state == state.rng.generator.bit_generator.state


def test_restart_refuses_another_engine_and_another_structure(tmp_path):
    _, state = make_state()
    path = tmp_path / "restart.npz"
    restart_io.write_restart(path, state, engine="ase")
    with pytest.raises(ValueError, match="was written by the ase engine"):
        restart_io.read_restart(path, engine="gpu")
    stored = restart_io.read_restart(path)
    _, other = make_state()
    other.species[0] = 1
    with pytest.raises(ValueError, match="species"):
        restart_io.load_state(stored, other)


def test_restart_write_is_atomic(tmp_path):
    """Temporary file, then rename: a failing write leaves the previous restart file intact."""
    _, state = make_state()
    path = tmp_path / "restart.npz"
    restart_io.write_restart(path, state, engine="gpu")
    before = path.read_bytes()
    fake_step(state, 300.0, FakeIntegrator(state))
    with patch.object(np, "savez", side_effect=OSError("disk full")), pytest.raises(OSError):
        restart_io.write_restart(path, state, engine="gpu")
    assert path.read_bytes() == before
    restart_io.write_restart(path, state, engine="gpu")
    assert path.read_bytes() != before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["restart.npz"]


# -- resume (workflow.run_crystal) --

@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("variable_cell", [False, True], ids=["nvt", "npt"])
def test_resume_continues_the_uninterrupted_trajectory(tmp_path, asynchronous, variable_cell):
    """Crash in step 10 after the step-8 restart file; resuming gives the uninterrupted files."""
    straight, crashed = tmp_path / "straight", tmp_path / "crashed"
    _, reference = run(straight, asynchronous=asynchronous, variable_cell=variable_cell)
    with pytest.raises(Crash):
        run(crashed, asynchronous=asynchronous, crash_in=10, variable_cell=variable_cell)
    with np.load(crashed / "restart.npz") as stored:
        assert int(stored["step"]) == 8
        assert int(stored["noise_next"]) == 2  # inside the batch drawn at step 7
    real_load, opened = np.load, []

    def tracking_load(*args, **kwargs):
        handle = real_load(*args, **kwargs)
        opened.append(handle)
        return handle

    with patch.object(np, "load", tracking_load):
        _, resumed = run(crashed, asynchronous=asynchronous, resume=True, variable_cell=variable_cell)
    # The restart file must not stay open while the run continues (Windows drives in WSL refuse
    # to replace an open file).
    assert opened and all(getattr(h, "fid", None) is None for h in opened)
    assert_states_equal(resumed, reference)
    assert file_bytes(crashed) == file_bytes(straight)
    assert len(read(crashed / "md.traj", index=":")) == 11


def test_resume_drops_outputs_written_after_the_restart_step(tmp_path):
    """A crash in step 11 leaves the row and frame of step 10 past the step-8 restart: they go."""
    straight, crashed = tmp_path / "straight", tmp_path / "crashed"
    run(straight, asynchronous=False)
    with pytest.raises(Crash):
        run(crashed, asynchronous=False, crash_in=11)
    with (crashed / "temperature.csv").open() as handle:
        assert [int(r["step"]) for r in csv.DictReader(handle)][-1] == 10
    assert len(read(crashed / "md.traj", index=":")) == 6
    with np.load(crashed / "restart.npz") as stored:
        assert int(stored["step"]) == 8
    run(crashed, asynchronous=False, resume=True)
    with (crashed / "temperature.csv").open() as a, (straight / "temperature.csv").open() as b:
        assert [r["step"] for r in csv.DictReader(a)] == [r["step"] for r in csv.DictReader(b)]  # no duplicates
    assert file_bytes(crashed) == file_bytes(straight)


def test_engine_snapshot_shape(tmp_path):
    """copy_to_host on the CPU: plain host copies, no event (the research engine's CPU path)."""
    _, state = make_state()
    host, event = trajectory_io.copy_to_host(output_values(state))
    assert event is None
    assert set(host) == {"positions", "momenta", "forces", "energy", "kinetic"}
    assert host["positions"].data_ptr() != state.positions.data_ptr()
    assert torch.equal(host["positions"], state.positions)
    assert host["energy"].shape == (1,) and host["kinetic"].shape == (1,)
    values = trajectory_io.wait(Snapshot(step=0, values=host, event=event))
    assert values is host

