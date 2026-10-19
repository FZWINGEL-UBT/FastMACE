"""``Simulation.add_sink``: asynchronous per-step sinks for ``fastmace.ase``'s files (plan task 1b-T21).

The hook goes through the chunked output's queue: no sink, no extra work; sinks drained at restart
points and at the end of ``run``, oldest first.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace import integrators
from fastmace.config import EngineConfig
from fastmace.engine import constraints
from fastmace.io import output as output_io
from fastmace.io.output import Output
from fastmace.planner import minimal

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

LANGEVIN = integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01)


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    pytest.importorskip("cuequivariance_torch")
    from fastmace.model import loading

    model = tiny_mace_model([11, 17], **FAMILIES["mh1"])
    path = tmp_path_factory.mktemp("models") / "tiny-mh1-11-17.model"
    torch.save(model, path)
    loading._LICENCES_SHOWN.add(str(path))
    return loading.load_pristine(str(path), trusted=True)


@pytest.fixture(autouse=True)
def cpu_planner(monkeypatch):
    planner = minimal.MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ={},
                                     process=constraints.ProcessRecord())
    monkeypatch.setattr(minimal, "default_planner", lambda force_field: planner)


def _simulation(loaded, output=None):
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    return Simulation(atoms, ForceField(loaded, atoms, precision="float64"), integrator=LANGEVIN, seed=4,
                      output=output)


def _output(path, restart_every=None):
    return Output(str(path), trajectory_every=3, thermo_every=2, restart_every=restart_every)


def _count_snapshots(sim):
    engine = sim._engine
    original = engine.snapshot
    calls = []

    def snapshot():
        calls.append(engine.state.step)
        return original()

    engine.snapshot = snapshot
    return calls


def test_no_sink_takes_no_extra_snapshots(loaded, tmp_path):
    sim = _simulation(loaded, output=_output(tmp_path))
    calls = _count_snapshots(sim)
    sim.run(steps=6)
    assert calls == [0, 2, 3, 4, 6]  # the chunked output's thermo rows and frames only


def test_a_sink_leaves_the_trajectory_and_the_output_unchanged(loaded, tmp_path):
    plain = _simulation(loaded, output=_output(tmp_path / "plain"))
    plain.run(steps=7)
    plain.close()
    sunk = _simulation(loaded, output=_output(tmp_path / "sunk"))
    sunk.add_sink(lambda step, values: None, 1)
    sunk.run(steps=7)
    sunk.close()
    np.testing.assert_array_equal(sunk.committed().positions, plain.committed().positions)
    np.testing.assert_array_equal(sunk.committed().momenta, plain.committed().momenta)
    assert (tmp_path / "sunk" / "thermo.0.csv").read_text() == (tmp_path / "plain" / "thermo.0.csv").read_text()
    for a, b in zip(output_io.read_frames(tmp_path / "sunk"), output_io.read_frames(tmp_path / "plain"), strict=True):
        np.testing.assert_array_equal(a.positions, b.positions)
        np.testing.assert_array_equal(a.get_forces(), b.get_forces())


def test_a_sink_sees_every_multiple_of_its_interval_with_the_committed_values(loaded):
    sim = _simulation(loaded)
    seen = []
    sim.add_sink(lambda step, values: seen.append((step, values["positions"].copy())), 2)
    sim.run(steps=3)
    assert [step for step, _ in seen] == [2]  # drained at the end of run
    sim.run(steps=3)
    assert [step for step, _ in seen] == [2, 4, 6]
    np.testing.assert_array_equal(seen[-1][1], sim.committed().positions)


def test_sinks_are_drained_at_restart_points_in_queue_order(loaded, tmp_path):
    sim = _simulation(loaded, output=_output(tmp_path, restart_every=4))
    order = []
    sim.add_sink(lambda step, values: order.append(("sink", step)), 2)
    original_write = sim._write

    def write(step, values, sample, frame):
        order.append(("output", step))
        original_write(step, values, sample, frame)

    sim._write = write
    original_save = sim.save_restart

    def save_restart(path):
        original_save(path)
        order.append(("restart", sim.committed().step))

    sim.save_restart = save_restart
    sim.run(steps=4)
    assert order == [("output", 0), ("output", 2), ("sink", 2), ("output", 3), ("output", 4), ("sink", 4),
                     ("restart", 4)]


@pytest.mark.parametrize("sink, interval, error", [(None, 1, TypeError), (print, 0, ValueError),
                                                   (print, True, ValueError), (print, 1.5, ValueError)])
def test_add_sink_checks_its_arguments(loaded, sink, interval, error):
    with pytest.raises(error):
        _simulation(loaded).add_sink(sink, interval)
