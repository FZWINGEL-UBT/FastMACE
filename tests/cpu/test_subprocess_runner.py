"""The subprocess runner (plan task 2-T07; SPEC §8.7, §10.6; phase 2 acceptance item 4).

A plan that a process constraint forbids here (``Plan.runner == "subprocess"``) runs in a child
process, handed over through restart writes. On the CPU the constraint cannot arise (nothing is
compiled), so the tests mark the parent's plans ``"subprocess"`` by monkeypatching the planner in
this process; the child is a fresh process and plans for itself.

- A plan run through the runner, with two hand-offs, continues bit for bit like an in-process run
  with restart writes at the same steps (CPU float64): state, thermo rows, frames, restart points.
- Sinks, ``submit_frame`` and host writes (owner decision D4) go through the hand-offs and equal
  the in-process run; ``close()`` removes the hand-off directory.
- A child that raises, or is killed, gives ``SubprocessRunnerError`` with its message, never a hang;
  the hand-off directory is kept. A stalled or too-long child is killed (the wait, on a sleeping
  child).
- The default path never reaches the runner.

The fault injections live here: a wrapper interpreter given as ``RunnerConfig.python`` patches the
child before it runs (no hook in the library).
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace import integrators
from fastmace.config import EngineConfig, RunnerConfig
from fastmace.engine import constraints
from fastmace.engine import subprocess as runner_module
from fastmace.errors import SubprocessRunnerError
from fastmace.io import output as output_io
from fastmace.io.output import Output
from fastmace.planner import minimal
from fastmace.protocols import ramp

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

NA_CL = (11, 17)
LANGEVIN = integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01)
CPU_ENGINE = EngineConfig(device="cpu", compile_mode="none")  # the child plans with the same options
POSIX = pytest.mark.skipif(os.name != "posix", reason="the wrapper interpreter is a shell script")


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    pytest.importorskip("cuequivariance_torch")
    from fastmace.model import loading

    model = tiny_mace_model(list(NA_CL), **FAMILIES["mh1"])
    path = tmp_path_factory.mktemp("models") / "tiny-mh1-11-17.model"
    torch.save(model, path)
    loading._LICENCES_SHOWN.add(str(path))
    return loading.load_pristine(str(path), trusted=True)


@pytest.fixture(autouse=True)
def cpu_planner(monkeypatch, tmp_path):
    record = constraints.ProcessRecord()
    planner = minimal.MinimalPlanner(engine=CPU_ENGINE, environ={}, process=record)
    monkeypatch.setattr(minimal, "default_planner", lambda force_field: planner)
    monkeypatch.setenv("FASTMACE_CACHE", str(tmp_path / "cache"))
    return planner


@pytest.fixture
def runner_plans_later(monkeypatch):
    """Call it to mark the plans made from then on (``runner_plans``, after a reference run)."""
    from fastmace.errors import Reason

    def start():
        def claim(self, plan, owner, *, offer_runner=False):
            if not offer_runner:
                return plan
            reason = Reason("marked by the test (SPEC §8.7)", code="runner-subprocess-family")
            return dataclasses.replace(plan, runner="subprocess",
                                       decisions=(*plan.decisions, ("runner", "subprocess", reason)))

        monkeypatch.setattr(minimal.MinimalPlanner, "_claim_family", claim)

    return start


@pytest.fixture
def runner_plans(runner_plans_later):
    """Every MD plan made in this process from now on is marked to run in a child."""
    runner_plans_later()


def _simulation(loaded, *, seed=11, output=None, runner=None, atoms=None, **options):
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    atoms = nacl(repeat=1, rattle=0.05, seed=3) if atoms is None else atoms
    return Simulation(atoms, ForceField(loaded, atoms, precision="float64"), integrator=LANGEVIN, output=output,
                      seed=seed, engine=CPU_ENGINE, runner=runner, **options)


def _output(directory):
    return Output(directory, thermo_every=1, trajectory_every=1, restart_every=3)


def _state(sim):
    committed = sim.committed()
    return committed.step, committed.positions.copy(), committed.momenta.copy(), committed.energy, \
        committed.forces.copy()


def _assert_same_state(got, want):
    assert got[0] == want[0]
    for a, b in zip(got[1:], want[1:]):
        np.testing.assert_array_equal(a, b)


def _metadata(directory):
    return json.loads((directory / "metadata.json").read_text())


# --------------------------------------------------------------------------- the default path


def test_default_plans_never_reach_the_runner(loaded, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the subprocess runner was reached on the default path")

    monkeypatch.setattr(runner_module.SubprocessRunner, "__init__", refuse)
    sim = _simulation(loaded)
    assert sim.plan.runner == "in-process"
    sim.run(steps=2)
    assert sim.committed().step == 2


# --------------------------------------------------------------------------- bitwise hand-offs


def test_two_hand_offs_continue_bitwise_against_an_in_process_run(loaded, tmp_path, runner_plans_later):
    schedule = ramp(300.0, 250.0, ps=0.007)  # 7 steps at 1 fs
    points = tmp_path / "points"

    # In process, with restart writes where the runner hands over: at setup and after each run().
    whole = _simulation(loaded, output=_output(tmp_path / "whole"))
    start = _state(whole)
    whole.save_restart(points)
    whole.run(schedule=schedule, steps=2)
    whole.save_restart(points)
    whole.run(steps=3)
    whole.save_restart(points)
    whole.run()
    whole.save_restart(points)
    want = _state(whole)
    whole.close()

    runner_plans_later()
    sim = _simulation(loaded, output=_output(tmp_path / "child"))
    assert sim.plan.runner == "subprocess" and sim.plan.decision("runner")[1] == "subprocess"
    _assert_same_state(_state(sim), start)  # the setup child's step 0
    sim.run(schedule=schedule, steps=2)
    assert sim.schedule.step == 2 and sim.committed().step == 2
    sim.run(steps=3)
    sim.run()
    assert sim.schedule.finished
    _assert_same_state(_state(sim), want)
    assert sim.plan.runner == "subprocess" and sim.plan.engine == whole.plan.engine
    assert "subprocess" in sim.report.text()
    sim.close()

    assert output_io.read_thermo(tmp_path / "child") == output_io.read_thermo(tmp_path / "whole")
    frames, wanted = output_io.read_frames(tmp_path / "child"), output_io.read_frames(tmp_path / "whole")
    assert len(frames) == len(wanted) == 8  # no duplicate or missing frame
    for got, ref in zip(frames, wanted):
        np.testing.assert_array_equal(got.positions, ref.positions)
    metadata, reference = _metadata(tmp_path / "child"), _metadata(tmp_path / "whole")
    assert metadata["restart_points"] == reference["restart_points"]
    assert [point["step"] for point in metadata["restart_points"]] == [0, 2, 3, 5, 6, 7]
    assert metadata["subprocess"]["calls"] == 4  # setup and three runs
    assert metadata["subprocess"]["children"] == 1 and metadata["subprocess"]["persistent"]  # one worker (OPT-T07)
    assert metadata["timing"]["status"] == "complete"
    assert not (tmp_path / "child" / runner_module.HANDOFF_NAME).exists()  # removed by a clean close()


def _drive(sim, points=None):
    """Sinks, a frame at step 0 and a host write across hand-offs; returns what the sinks saw."""
    seen = []
    sim.submit_frame(lambda step, values: seen.append(("frame", step, values["positions"].copy())))
    sim.add_sink(lambda step, values: seen.append(("sink", step, values["energy"].copy())), 2)
    if points is not None:
        sim.save_restart(points)
    sim.run(steps=3)
    if points is not None:
        sim.save_restart(points)
    sim.set_momenta(sim.committed().momenta * 0.5)
    sim.run(steps=2)
    if points is not None:
        sim.save_restart(points)
    sim.flush_output()
    return seen


def test_sinks_frames_and_host_writes_through_the_runner(loaded, tmp_path, runner_plans_later):
    whole = _simulation(loaded)
    want_seen = _drive(whole, tmp_path / "points")
    want = _state(whole)

    runner_plans_later()
    sim = _simulation(loaded)
    assert sim.plan.runner == "subprocess"
    seen = _drive(sim)
    assert [(kind, step) for kind, step, _ in seen] == [(kind, step) for kind, step, _ in want_seen]
    assert [(kind, step) for kind, step, _ in seen] == [("frame", 0), ("sink", 2), ("sink", 4)]
    for (_, _, got), (_, _, ref) in zip(seen, want_seen):
        np.testing.assert_array_equal(got, ref)
    _assert_same_state(_state(sim), want)
    handoffs = tmp_path / "cache" / "handoff"
    assert any(handoffs.iterdir())
    sim.close()
    assert not any(handoffs.iterdir())


def test_resume_through_the_runner_continues_bitwise(loaded, tmp_path, runner_plans_later):
    from fastmace.simulation import Simulation

    whole = _simulation(loaded, output=_output(tmp_path / "whole"))
    whole.run(steps=5)
    want = _state(whole)
    whole.close()

    first = _simulation(loaded, output=_output(tmp_path / "parts"))
    first.run(steps=4)  # the restart file is from step 3
    del first
    runner_plans_later()
    resumed = Simulation.resume(tmp_path / "parts", allow_plan_change=True)
    assert resumed.plan.runner == "subprocess" and resumed.committed().step == 3
    resumed.run(steps=2)
    _assert_same_state(_state(resumed), want)
    resumed.close()
    assert output_io.read_thermo(tmp_path / "parts") == output_io.read_thermo(tmp_path / "whole")


def _swapped(atoms):
    """The structure after an accepted Na <-> Cl swap of atoms 0 and 1, with the new species' masses."""
    from ase.data import atomic_masses

    atoms = atoms.copy()
    numbers = atoms.numbers.copy()
    numbers[[0, 1]] = numbers[[1, 0]]
    atoms.numbers = numbers
    atoms.set_masses(atomic_masses[numbers])
    return atoms


def test_a_child_resumes_the_composition_of_the_restart(loaded, tmp_path, runner_plans_later):
    """A fresh child takes species and masses from the hand-off restart, not the initial structure (3-T08)."""
    from fastmace.simulation import SIMULATION_NAME, Simulation

    initial = nacl(repeat=1, rattle=0.05, seed=3)
    current = _swapped(initial)
    whole = _simulation(loaded, output=_output(tmp_path / "whole"), atoms=current)
    whole.run(steps=5)
    want = _state(whole)
    whole.close()

    first = _simulation(loaded, output=_output(tmp_path / "parts"), atoms=current)
    first.run(steps=4)  # the restart file is from step 3
    del first
    record_path = tmp_path / "parts" / SIMULATION_NAME
    record = json.loads(record_path.read_text())
    record["structure"]["numbers"] = [int(z) for z in initial.numbers]
    record["structure"]["masses"] = [float(m) for m in initial.get_masses()]
    record_path.write_text(json.dumps(record))
    runner_plans_later()
    resumed = Simulation.resume(tmp_path / "parts", allow_plan_change=True)
    assert resumed.plan.runner == "subprocess" and resumed.committed().step == 3
    resumed.run(steps=2)
    _assert_same_state(_state(resumed), want)
    resumed.close()


def test_a_species_host_write_reaches_the_next_child(loaded, tmp_path, runner_plans):
    """A composition change through the runner: the next fresh child applies it after its resume, and
    its hand-off restart holds the new species, which the child after it resumes (3-T08)."""
    from fastmace.simulation import Simulation

    initial = nacl(repeat=1, rattle=0.05, seed=3)
    new_species = _swapped(initial).numbers
    sim = _simulation(loaded, output=_output(tmp_path / "parts"), runner=RunnerConfig(persistent=False))
    assert sim.plan.runner == "subprocess"
    sim.run(steps=2)
    try:
        sim.set_species(new_species)
        sim.run(steps=2)  # the child applies it
    except NotImplementedError:
        pytest.skip("Simulation.set_species: plan task 3-T03 (merged before 3-T08)")
    except SubprocessRunnerError as error:  # the child's species event refused without 3-T02's check
        if "check_species" not in str(error):
            raise
        pytest.skip("the child's species event needs 3-T02's MACEForceField.check_species")
    sim.save_restart(tmp_path / "saved")
    resumed = Simulation.resume(tmp_path / "saved", allow_plan_change=True)
    resumed.run(steps=1)
    sim.run(steps=1)
    _assert_same_state(_state(resumed), _state(sim))
    sim.close()
    resumed.close()


def test_the_ase_shim_through_the_runner(loaded, tmp_path, monkeypatch, runner_plans_later):
    """``fastmace.ase``'s observers and host writes go through the hand-offs (owner decision D4)."""
    from ase import units
    from ase.io import read

    from fastmace.ase import Langevin
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # the child's default planner: the CPU, as here

    def atoms():
        structure = nacl(repeat=1, rattle=0.05, seed=3)
        structure.set_momenta(np.zeros((len(structure), 3)))
        return structure

    reference = Simulation(atoms(), ForceField(loaded, atoms(), precision="float32"), integrator=integrators.Langevin(
        timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01), seed=1)
    reference.save_restart(tmp_path / "points")
    reference.run(steps=3)
    reference.save_restart(tmp_path / "points")
    reference.set_positions(reference.committed().positions + 0.01)
    reference.run(steps=2)

    runner_plans_later()
    moved = atoms()
    dyn = Langevin(moved, 1.0 * units.fs, temperature_K=300, friction=0.01 / units.fs, model=loaded, seed=1,
                   trajectory="md.traj", loginterval=1)
    seen = []

    def shift():
        seen.append(dyn.nsteps)
        if dyn.nsteps == 3:
            moved.positions += 0.01

    dyn.attach(shift, interval=3)
    dyn.run(5)
    assert dyn.simulation.plan.runner == "subprocess" and seen == [0, 3]
    _assert_same_state(_state(dyn.simulation), _state(reference))
    np.testing.assert_array_equal(moved.positions, reference.committed().positions)
    dyn.close()
    frames = read("md.traj", index=":")
    assert [len(frames)] == [6]  # steps 0-5, none twice
    assert not (tmp_path / "fastmace-run" / runner_module.HANDOFF_NAME).exists()


# --------------------------------------------------------------------------- errors


def _wrapper(tmp_path, body, **config):
    """A wrapper interpreter (``RunnerConfig.python``) whose child runs ``body`` in place of
    ``Simulation.run``. ``body`` may call ``ORIGINAL(self, *args, **kwargs)`` and count the calls in
    this child process in ``CALLS[0]``; ``config`` goes to ``RunnerConfig``."""
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent("""\
        import os
        import signal
        import sys

        import fastmace.simulation as simulation

        ORIGINAL = simulation.Simulation.run
        CALLS = [0]


        def run(self, *args, **kwargs):
            CALLS[0] += 1
        {body}


        simulation.Simulation.run = run
        from fastmace.engine import subprocess as runner

        sys.exit(runner.main(sys.argv[3:]))  # after "-m fastmace.engine.subprocess"
        """).format(body=textwrap.indent(textwrap.dedent(body), "    ")))
    python = tmp_path / "python"
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    python.chmod(0o755)
    return RunnerConfig(python=str(python), **config)


def _drive_update(sim):
    sim.run(steps=2)
    masses = sim.committed().masses * np.where(np.arange(8) % 2 == 0, 1.5, 0.8)
    sim.update(masses=masses, friction_per_fs=0.02)
    sim.run(steps=2)
    return masses


@POSIX
@pytest.mark.parametrize("persistent", [True, False], ids=["worker", "fresh-children"])
def test_mutations_through_the_runner_equal_an_in_process_run(loaded, tmp_path, runner_plans_later, persistent):
    """``Simulation.update`` reaches the child as one host write per field (``HOST_WRITE_KINDS``, 3-T03), and
    a fresh child resumes the hand-off restart with the new masses and friction (restart v3, 3-T08);
    ``committed()`` carries the composition."""
    whole = _simulation(loaded)
    masses = _drive_update(whole)
    want = _state(whole)

    runner_plans_later()
    sim = _simulation(loaded, runner=RunnerConfig(persistent=persistent))
    assert sim.plan.runner == "subprocess"
    _drive_update(sim)
    _assert_same_state(_state(sim), want)
    committed = sim.committed()
    np.testing.assert_array_equal(committed.masses, masses)
    np.testing.assert_array_equal(committed.species, whole.committed().species)
    assert committed.charges is None
    assert sim.integrator.friction_per_fs == 0.02
    sim.close()


@POSIX
def test_a_failing_child_raises_with_its_message(loaded, tmp_path, runner_plans):
    runner = _wrapper(tmp_path, 'raise RuntimeError("injected failure in the child")')
    sim = _simulation(loaded, runner=runner)
    before = _state(sim)
    with pytest.raises(SubprocessRunnerError) as caught:
        sim.run(steps=2)
    error = caught.value
    assert "injected failure in the child" in error.child_message and "injected failure" in str(error)
    assert error.exit_code == 1
    assert error.directory is not None and os.path.isdir(error.directory)  # kept after an error
    _assert_same_state(_state(sim), before)  # the parent stays at the last hand-off
    sim.close()
    assert os.path.isdir(error.directory)


@POSIX
def test_a_killed_child_raises_with_its_message(loaded, tmp_path, runner_plans):
    runner = _wrapper(tmp_path, """\
        print("injected: the child kills itself", file=sys.stderr, flush=True)
        os.kill(os.getpid(), signal.SIGKILL)
        """)
    sim = _simulation(loaded, runner=runner)
    with pytest.raises(SubprocessRunnerError) as caught:
        sim.run(steps=2)
    assert caught.value.exit_code == -9
    assert "injected: the child kills itself" in caught.value.child_message


def _sleeping_child(tmp_path):
    heartbeat = tmp_path / "heartbeat"
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                               start_new_session=os.name == "posix")
    return process, heartbeat


def test_a_stalled_child_is_killed(tmp_path):
    process, heartbeat = _sleeping_child(tmp_path)
    code, reason = runner_module._wait(process, heartbeat, timeout_s=None, stall_timeout_s=0.5, poll_s=0.05)
    assert code is not None and code != 0
    assert "stall_timeout_s=0.5" in reason


def test_a_child_over_its_time_limit_is_killed(tmp_path):
    process, heartbeat = _sleeping_child(tmp_path)
    code, reason = runner_module._wait(process, heartbeat, timeout_s=0.5, stall_timeout_s=60.0, poll_s=0.05)
    assert code is not None and code != 0
    assert "timeout_s=0.5" in reason


def test_progress_resets_the_stall_clock(tmp_path):
    heartbeat = tmp_path / "heartbeat"
    code = ("import sys, time\n"
            "for step in range(8):\n"
            "    open(sys.argv[1], 'w').write(f'running {step}')\n"
            "    time.sleep(0.15)\n")
    process = subprocess.Popen([sys.executable, "-c", code, str(heartbeat)])
    code, reason = runner_module._wait(process, heartbeat, timeout_s=None, stall_timeout_s=0.6, poll_s=0.05)
    assert (code, reason) == (0, None)  # 1.2 s in all, never 0.6 s without progress


# --------------------------------------------------------------------------- durable copies (2-T07b)


class _Queue:
    def drain(self):
        pass


class _Parent:
    """What ``SubprocessRunner.save_restart`` reads from its ``Simulation``."""

    _queue = _Queue()

    def _write_metadata(self):
        pass


def _hand_off(tmp_path, marker):
    """A runner whose current hand-off holds a restart and record marked ``marker``."""
    stage = tmp_path / "stage"
    stage.mkdir(exist_ok=True)
    np.savez(stage / "restart.npz", marker=np.array([marker]))
    (stage / "simulation.json").write_text(json.dumps({"marker": marker, "output": None}))
    runner = runner_module.SubprocessRunner()
    runner.current = stage
    return runner


def test_an_interrupted_save_restart_keeps_the_previous_checkpoint(tmp_path, monkeypatch):
    target = tmp_path / "run"
    _hand_off(tmp_path, 1).save_restart(_Parent(), target / "restart.npz")
    before = (target / "restart.npz").read_bytes()
    runner = _hand_off(tmp_path, 2)

    def interrupted(source, destination, *args, **kwargs):
        destination.write(b"NEW_")
        raise OSError("injected: the copy was interrupted")

    monkeypatch.setattr(runner_module.shutil, "copyfileobj", interrupted)
    with pytest.raises(OSError, match="injected"):
        runner.save_restart(_Parent(), target / "restart.npz")
    assert (target / "restart.npz").read_bytes() == before
    with np.load(target / "restart.npz") as stored:
        assert stored["marker"].tolist() == [1]
    assert json.loads((target / "simulation.json").read_text())["marker"] == 1
    assert sorted(path.name for path in target.iterdir()) == ["restart.npz", "simulation.json"]  # no temporary


def test_save_restart_fsyncs_replaces_then_fsyncs_the_directory(tmp_path, monkeypatch):
    from test_durable import Recorder, directory_fsync

    target = tmp_path / "run"
    target.mkdir()
    runner = _hand_off(tmp_path, 1)
    recorder = Recorder(monkeypatch, target)
    runner.save_restart(_Parent(), target / "restart.npz")
    events = []
    for name in ("restart.npz", "simulation.json"):
        temporary = f".{name}.{os.getpid()}.tmp"
        events += [("fsync", temporary), ("replace", temporary, name), *directory_fsync()]
    assert recorder.events == events
    with np.load(target / "restart.npz") as stored:
        assert stored["marker"].tolist() == [1]


# --------------------------------------------------------------------------- the heartbeat (2-T07c)


def test_concurrent_heartbeat_writes_never_raise(tmp_path):
    """The heartbeat thread and the main thread both write the file: the writes must not race on one
    temporary file (seen as FileNotFoundError on the rename under load, killing a call)."""
    import threading

    path = tmp_path / "child-0001.heartbeat"
    beat = runner_module._Heartbeat(path)
    beat.start(0.0005)  # the thread rewrites as fast as it can
    errors = []

    def hammer(name):
        try:
            for index in range(400):
                beat.mark(f"{name}-{index}")
        except Exception as error:  # noqa: BLE001  (any exception is the failure)
            errors.append(error)

    threads = [threading.Thread(target=hammer, args=(f"writer{number}",)) for number in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    beat.stop()
    assert errors == []
    assert path.read_text(encoding="utf-8").startswith("writer")
    assert [entry.name for entry in tmp_path.iterdir()] == [path.name]  # no temporary file left


def test_a_failing_heartbeat_write_never_raises(tmp_path, monkeypatch):
    beat = runner_module._Heartbeat(tmp_path / "child-0001.heartbeat")

    def broken(*args, **kwargs):
        raise OSError("injected: the heartbeat's directory is gone")

    monkeypatch.setattr(runner_module.os, "replace", broken)
    beat.mark("call")  # the call goes on; the parent sees a stale heartbeat at worst
    assert [entry.name for entry in tmp_path.iterdir()] == ["child-0001.heartbeat"]


# --------------------------------------------------------------------------- mc= through the runner (3-T15)


@pytest.mark.parametrize("persistent", [True, False], ids=["worker", "fresh-children"])
def test_mc_attaches_keeps_and_detaches_through_the_runner(loaded, tmp_path, runner_plans, persistent):
    from fastmace.errors import PlanRefused
    from fastmace.mc import Displace, Swap
    from fastmace.planner.workload import Workload

    workload = Workload(max_temperature_K=300.0, species_changes=True, mc_moves=("swap",),
                        min_mc_temperature_K=250.0)
    sim = _simulation(loaded, workload=workload, output=_output(tmp_path / "run"),
                      runner=RunnerConfig(persistent=persistent))
    assert sim.plan.runner == "subprocess"
    with pytest.raises(PlanRefused):  # checked in the parent, before any child runs
        sim.run(steps=2, mc=Displace(delta_A=0.1, every=2, attempts=1))
    sim.run(steps=4, mc=Swap(pairs=[("Na", "Cl")], every=2, attempts=5))  # blocks at 2 and 4
    sim.run(steps=2)  # kept: the block at 6 (a fresh child resumes it from the hand-off restart)
    sim.run(steps=2, mc=None)  # detached
    sim.run(steps=2)
    counters = sim.metadata()["counters"]["mc"]
    assert counters["swap"]["attempted"] == 15 and counters["swap"]["impossible"] == 0
    assert "swap Na-Cl every 2, 5 attempts" in sim.report.text()
    header, rows = output_io.read_thermo(tmp_path / "run")
    assert header[-2:] == ["mc_swap_attempted", "mc_swap_accepted"]
    assert [int(row[-2]) for row in rows][-1] == 15
    sim.close()
