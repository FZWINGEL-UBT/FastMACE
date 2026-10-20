"""The persistent subprocess worker (plan task OPT-T07; docs/design/opt-T07-persistent-worker.md).

One child serves every ``run()`` of a ``Plan.runner == "subprocess"`` simulation and keeps its
engine. Each call still ends with the canonical restart write of 2-T07, so the worker is the
"uninterrupted run with the same restart points" of SPEC §10.6, and today's fresh child per call
(``RunnerConfig(persistent=False)``) is the "interrupted" one: bit for bit equal on CPU float64.

- Repeated short calls in one worker equal an in-process run with restart writes after every call.
- The worker equals the fresh-child path: state, rows, frames, restart points; host writes and the
  ``fastmace.ase`` shim's observer mutations included.
- A failed call (raise, SIGKILL) ends the worker; the next ``run()`` starts a fresh child from the
  last hand-off and continues exactly.
- The worker ends on ``close()``, on garbage collection, at the parent's exit, and when the parent is
  killed, idle or in a call.

Plans are marked ``runner="subprocess"`` in the parent by monkeypatching its planner, as in 2-T07.
POSIX only: on Windows the runner keeps a fresh child per call.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import os
import signal
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest
import test_subprocess_runner as base

from _support.structures import nacl
from fastmace.config import RunnerConfig
from fastmace.engine import subprocess as runner_module
from fastmace.errors import SubprocessRunnerError
from fastmace.io import output as output_io
from fastmace.protocols import ramp

# 2-T07's fixtures and helpers, bound here so that pytest finds the fixtures in this module.
loaded, cpu_planner, runner_plans, runner_plans_later = (base.loaded, base.cpu_planner, base.runner_plans,
                                                         base.runner_plans_later)
_assert_same_state, _metadata, _output, _simulation, _state, _wrapper = (
    base._assert_same_state, base._metadata, base._output, base._simulation, base._state, base._wrapper)

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning"),
              pytest.mark.skipif(os.name != "posix", reason="the persistent worker is POSIX only (reply pipe)")]

FRESH = RunnerConfig(persistent=False)


def _block(sim):
    return sim.metadata()["subprocess"]


def _gone(pid, within=20.0):
    """Whether process ``pid`` is gone (or a zombie nobody waits for) within ``within`` seconds."""
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        try:
            with open(f"/proc/{pid}/stat") as handle:
                if handle.read().split()[2] == "Z":
                    return True
        except OSError:
            return True
        time.sleep(0.1)
    return False


# --------------------------------------------------------------------------- one worker, many calls


def test_repeated_short_calls_in_one_worker(loaded, tmp_path, runner_plans_later):
    points = tmp_path / "points"
    whole = _simulation(loaded, output=_output(tmp_path / "whole"))
    whole.save_restart(points)
    for steps in [1] * 10 + [3]:
        whole.run(steps=steps)
        whole.save_restart(points)
    want = _state(whole)
    whole.close()

    runner_plans_later()
    sim = _simulation(loaded, output=_output(tmp_path / "worker"))
    seconds = []
    for steps in [1] * 10 + [3]:
        started = time.perf_counter()
        sim.run(steps=steps)
        seconds.append(time.perf_counter() - started)
    _assert_same_state(_state(sim), want)
    block = _block(sim)
    assert (block["children"], block["calls"], block["persistent"]) == (1, 12, True)
    assert block["last"]["exit_code"] is None  # the worker is still running
    pid = block["last"]["pid"]
    sim.close()
    assert _gone(pid)
    assert output_io.read_thermo(tmp_path / "worker") == output_io.read_thermo(tmp_path / "whole")
    assert len(output_io.read_frames(tmp_path / "worker")) == len(output_io.read_frames(tmp_path / "whole")) == 14
    assert _metadata(tmp_path / "worker")["restart_points"] == _metadata(tmp_path / "whole")["restart_points"]
    print(f"calls after the first: {max(seconds[1:]):.2f} s at most (a fresh child is about 5 s)")


def _sequence(sim):
    """Four calls with a schedule, a host write, ``committed()`` after it, and a frame at step 0."""
    seen = []
    sim.submit_frame(lambda step, values: seen.append((step, values["positions"].copy())))
    sim.add_sink(lambda step, values: seen.append((step, values["energy"].copy())), 2)
    sim.run(schedule=ramp(300.0, 250.0, ps=0.009), steps=2)
    sim.run(steps=3)
    before = sim.committed()
    sim.set_positions(before.positions + 0.01)
    after = sim.committed()  # re-evaluated at the host write's canonical point
    assert not np.array_equal(after.forces, before.forces)
    sim.set_momenta(after.momenta * 0.5)
    sim.run(steps=2)
    sim.run()
    sim.flush_output()
    return seen, after


def test_the_worker_equals_a_fresh_child_per_call(loaded, tmp_path, runner_plans):
    fresh = _simulation(loaded, output=_output(tmp_path / "fresh"), runner=FRESH)
    want_seen, want_after = _sequence(fresh)
    want = _state(fresh)
    assert _block(fresh)["children"] == _block(fresh)["calls"] and not _block(fresh)["persistent"]
    fresh.close()

    sim = _simulation(loaded, output=_output(tmp_path / "worker"))
    seen, after = _sequence(sim)
    _assert_same_state(_state(sim), want)
    np.testing.assert_array_equal(after.forces, want_after.forces)
    assert [step for step, _ in seen] == [step for step, _ in want_seen]
    for (_, got), (_, ref) in zip(seen, want_seen):
        np.testing.assert_array_equal(got, ref)
    assert _block(sim)["children"] == 1
    sim.close()
    assert output_io.read_thermo(tmp_path / "worker") == output_io.read_thermo(tmp_path / "fresh")
    for got, ref in zip(output_io.read_frames(tmp_path / "worker"), output_io.read_frames(tmp_path / "fresh")):
        np.testing.assert_array_equal(got.positions, ref.positions)
    assert _metadata(tmp_path / "worker")["restart_points"] == _metadata(tmp_path / "fresh")["restart_points"]


def _shim(loaded, tmp_path, name):
    """The ASE shim with an observer every 2 steps that moves atoms and scales momenta, 10 steps."""
    from ase import units

    from fastmace.ase import Langevin

    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    atoms.set_momenta(np.zeros((len(atoms), 3)))
    dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=300, friction=0.01 / units.fs, model=loaded, seed=1,
                   output=str(tmp_path / name), trajectory=str(tmp_path / f"{name}.traj"), loginterval=1)

    def mutate():
        if dyn.nsteps:
            atoms.positions += 0.002
            atoms.set_momenta(atoms.get_momenta() * 0.9)

    dyn.attach(mutate, interval=2)
    dyn.run(10)
    state = _state(dyn.simulation)
    block = _block(dyn.simulation)
    dyn.close()
    return state, block


def test_observer_mutations_through_the_worker_equal_fresh_children(loaded, tmp_path, monkeypatch,
                                                                    runner_plans):
    from ase.io import read

    from fastmace import simulation

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # the child's default planner: the CPU, as here

    @dataclasses.dataclass(frozen=True)
    class Fresh(RunnerConfig):  # the shim takes no runner option: its default, made fresh here
        persistent: bool = False

    worker, worker_block = _shim(loaded, tmp_path, "worker")
    monkeypatch.setattr(simulation, "RunnerConfig", Fresh)
    fresh, fresh_block = _shim(loaded, tmp_path, "fresh")
    _assert_same_state(worker, fresh)
    assert worker_block["children"] == 1 and fresh_block["children"] == fresh_block["calls"] > 5
    frames = read(tmp_path / "worker.traj", index=":")
    assert len(frames) == 11
    for got, ref in zip(frames, read(tmp_path / "fresh.traj", index=":")):
        np.testing.assert_array_equal(got.positions, ref.positions)


# --------------------------------------------------------------------------- recovery


@pytest.mark.parametrize("fault", ["raise", "kill"])
def test_a_failed_call_ends_the_worker_and_the_next_run_resumes_fresh(loaded, tmp_path, runner_plans_later,
                                                                     fault):
    points = tmp_path / "points"
    whole = _simulation(loaded, output=_output(tmp_path / "whole"))
    whole.save_restart(points)
    for _ in range(2):
        whole.run(steps=2)
        whole.save_restart(points)
    want = _state(whole)
    whole.close()

    runner_plans_later()
    body = {"raise": 'raise RuntimeError("injected failure in call 2")',
            "kill": 'print("injected: killing call 2", file=sys.stderr, flush=True)\n'
                    'os.kill(os.getpid(), signal.SIGKILL)'}[fault]
    runner = _wrapper(tmp_path, f"if CALLS[0] == 2:\n    {body.replace(chr(10), chr(10) + '    ')}\n"
                                "return ORIGINAL(self, *args, **kwargs)")
    sim = _simulation(loaded, output=_output(tmp_path / "worker"), runner=runner)
    sim.run(steps=2)
    first_pid = _block(sim)["last"]["pid"]
    with pytest.raises(SubprocessRunnerError) as caught:
        sim.run(steps=2)
    assert "injected" in caught.value.child_message
    assert caught.value.exit_code == (1 if fault == "raise" else -9)
    assert _gone(first_pid)
    assert sim.committed().step == 2  # the parent stays at the last hand-off
    sim.run(steps=2)  # a fresh child resumes it
    _assert_same_state(_state(sim), want)
    block = _block(sim)
    assert block["children"] == 2 and block["failed"] is not None
    sim.close()
    assert output_io.read_thermo(tmp_path / "worker") == output_io.read_thermo(tmp_path / "whole")


def _sleeping(tmp_path):
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], stdin=subprocess.PIPE,
                               start_new_session=True)
    replies = runner_module._Replies(None)
    return process, replies, tmp_path / "heartbeat"


def test_a_stalled_call_is_killed(tmp_path):
    process, replies, heartbeat = _sleeping(tmp_path)
    outcome = runner_module._wait_reply(process, replies, heartbeat, timeout_s=None, stall_timeout_s=0.5,
                                        poll_s=0.05)
    assert outcome[0] == "stopped" and "stall_timeout_s=0.5" in outcome[2]
    assert process.poll() is not None


def test_a_call_over_its_time_limit_is_killed(tmp_path):
    process, replies, heartbeat = _sleeping(tmp_path)
    outcome = runner_module._wait_reply(process, replies, heartbeat, timeout_s=0.5, stall_timeout_s=60.0,
                                        poll_s=0.05)
    assert outcome[0] == "stopped" and "timeout_s=0.5" in outcome[2]


# --------------------------------------------------------------------------- termination


def test_close_ends_the_worker(loaded, runner_plans):
    sim = _simulation(loaded)
    sim.run(steps=1)
    pid = _block(sim)["last"]["pid"]
    sim.close()
    assert _gone(pid, within=1.0)


def test_garbage_collection_ends_the_worker(loaded, runner_plans):
    sim = _simulation(loaded)
    sim.run(steps=1)
    pid = _block(sim)["last"]["pid"]
    del sim
    gc.collect()
    assert _gone(pid)


PARENT = """
import fastmace  # noqa: F401
import dataclasses, os, sys, time
import torch
from _support.structures import nacl
from fastmace import integrators
from fastmace.config import EngineConfig
from fastmace.errors import Reason
from fastmace.model import loading
from fastmace.model.forcefield import ForceField
from fastmace.planner import minimal
from fastmace.simulation import Simulation


def claim(self, plan, owner, *, offer_runner=False):
    if not offer_runner:
        return plan
    reason = Reason("marked by the test (SPEC §8.7)", code="runner-subprocess-family")
    return dataclasses.replace(plan, runner="subprocess", decisions=(*plan.decisions, ("runner", "subprocess",
                                                                                         reason)))


minimal.MinimalPlanner._claim_family = claim
loaded = loading.load_pristine(sys.argv[1], trusted=True)
atoms = nacl(repeat=1, rattle=0.05, seed=3)
sim = Simulation(atoms, ForceField(loaded, atoms, precision="float64"), seed=11,
                 integrator=integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01),
                 engine=EngineConfig(device="cpu", compile_mode="none"))
print(sim.metadata()["subprocess"]["last"]["pid"], flush=True)
mode = sys.argv[2]
if mode == "busy":
    sim.run(steps=10**7)
elif mode == "idle":
    time.sleep(600)
"""


def _parent(loaded, tmp_path, mode):
    script = tmp_path / "parent.py"
    script.write_text(textwrap.dedent(PARENT))
    tests = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.dirname(tests), tests]),
                       FASTMACE_CACHE=str(tmp_path / "cache"), CUDA_VISIBLE_DEVICES="")
    process = subprocess.Popen([sys.executable, str(script), str(loaded.path), mode], stdout=subprocess.PIPE,
                               env=environment, text=True)
    pid = int(process.stdout.readline())
    return process, pid


def test_the_parents_exit_ends_the_worker(loaded, tmp_path):
    process, pid = _parent(loaded, tmp_path, "exit")
    assert process.wait(timeout=120) == 0
    assert _gone(pid)


@pytest.mark.parametrize("mode", ["idle", "busy"])
def test_a_killed_parent_ends_the_worker(loaded, tmp_path, mode):
    process, pid = _parent(loaded, tmp_path, mode)
    if mode == "busy":
        time.sleep(3.0)  # the call is running
    process.send_signal(signal.SIGKILL)
    process.wait(timeout=30)
    assert _gone(pid)


def test_the_record_counts_calls_and_children(loaded, runner_plans):
    sim = _simulation(loaded, runner=FRESH)
    sim.run(steps=1)
    block = _block(sim)
    assert (block["children"], block["calls"], block["persistent"]) == (2, 2, False)
    assert block["last"]["exit_code"] == 0
    sim.close()
    assert json.dumps(block)  # JSON-safe
