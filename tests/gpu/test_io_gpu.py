"""Async output and restart on CUDA: real pinned copies and events (plan task 1a-T14).

Reuses the fake dynamics of ``tests/cpu/test_io.py`` with the state on the GPU.
"""

import sys
from pathlib import Path

import numpy as np
import torch

from fastmace.engine.base import Snapshot
from fastmace.io import restart as restart_io
from fastmace.io import trajectory as trajectory_io

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpu"))
import test_io as cpu  # noqa: E402

DEVICE = "cuda"


def test_copy_to_host_is_pinned_and_ordered_behind_the_step():
    """Pinned host tensors, an event on the current stream, values of the step they were taken at."""
    _, state = cpu.make_state(DEVICE)
    integrator = cpu.FakeIntegrator(state)
    cpu.fake_step(state, 300.0, integrator)
    expected = {key: value.cpu() for key, value in cpu.output_values(state).items()}
    host, event = trajectory_io.copy_to_host(cpu.output_values(state))
    assert isinstance(event, torch.cuda.Event)
    assert all(value.is_pinned() and value.device.type == "cpu" for value in host.values())
    # Later GPU work on the same stream does not reach the queued copies.
    for _ in range(5):
        cpu.fake_step(state, 300.0, integrator)
    values = trajectory_io.wait(Snapshot(step=1, values=host, event=event))
    assert event.query()
    for key, value in expected.items():
        assert torch.equal(values[key], value), key


def test_async_output_writes_the_same_files(tmp_path):
    for variable_cell in (False, True):
        sync_dir, async_dir = tmp_path / f"sync-{variable_cell}", tmp_path / f"async-{variable_cell}"
        _, a = cpu.run(sync_dir, asynchronous=False, variable_cell=variable_cell, device=DEVICE)
        _, b = cpu.run(async_dir, asynchronous=True, variable_cell=variable_cell, device=DEVICE)
        cpu.assert_states_equal(a, b)
        assert cpu.file_bytes(async_dir) == cpu.file_bytes(sync_dir), variable_cell


def test_restart_round_trip_on_the_gpu(tmp_path):
    _, state = cpu.make_state(DEVICE, variable_cell=True)
    integrator = cpu.FakeIntegrator(state)
    for _ in range(5):
        cpu.fake_step(state, 200.0, integrator)
    path = tmp_path / "restart.npz"
    restart_io.write_restart(path, state, engine="gpu", variable_cell=True, integrator_state=integrator.state_dict())
    _, fresh = cpu.make_state(DEVICE, variable_cell=True)
    fresh_integrator = cpu.FakeIntegrator(fresh)
    addresses = [fresh.positions.data_ptr(), fresh.velocities.data_ptr(), fresh.rng.batch.data_ptr()]
    restart_io.load_state(restart_io.read_restart(path, engine="gpu"), fresh, fresh_integrator)
    assert [fresh.positions.data_ptr(), fresh.velocities.data_ptr(), fresh.rng.batch.data_ptr()] == addresses
    assert fresh.positions.is_cuda and fresh.rng.batch.is_cuda
    cpu.assert_states_equal(state, fresh)
    for _ in range(4):
        cpu.fake_step(state, 200.0, integrator)
        cpu.fake_step(fresh, 200.0, fresh_integrator)
    cpu.assert_states_equal(state, fresh)
    np.testing.assert_array_equal(fresh_integrator.p_eps.cpu().numpy(), integrator.p_eps.cpu().numpy())


def test_resume_continues_the_uninterrupted_trajectory(tmp_path):
    straight, crashed = tmp_path / "straight", tmp_path / "crashed"
    _, reference = cpu.run(straight, asynchronous=True, device=DEVICE)
    try:
        cpu.run(crashed, asynchronous=True, crash_in=11, device=DEVICE)
    except cpu.Crash:
        pass
    _, resumed = cpu.run(crashed, asynchronous=True, resume=True, device=DEVICE)
    cpu.assert_states_equal(resumed, reference)
    assert cpu.file_bytes(crashed) == cpu.file_bytes(straight)
