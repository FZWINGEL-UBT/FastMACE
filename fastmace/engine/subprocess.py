"""The subprocess runner (SPEC §8.7, §10.6): a plan that a process constraint forbids here runs in a child.

Plan task 2-T01 fixed the contract; plan task 2-T07 implements it. Where §8.7 forbids a plan in this
process (a second backend family after a compiled model; ``reproducible=True`` after CUDA was
initialised), the planner marks it ``Plan.runner == "subprocess"``. ``Simulation`` then builds no
engine and runs no gate in this process; this runner does the work in child processes:

1. **setup** (at construction): the child builds the ``Simulation`` from the parent's record and
   ``Atoms`` (planning, gate and build happen there, so their refusals still come before the first
   step) and writes a canonical restart at step 0, the first hand-off;
2. **each** ``run()``: the child resumes the last hand-off with ``allow_plan_change=True``, applies
   the parent's pending host writes (owner decision D4), runs the steps and writes a canonical
   restart at the last committed step, the next hand-off. Its output chunks continue the
   parent's (§10.6): every hand-off is a restart point and so a chunk boundary;
3. the parent reads the child's restart, committed values and records back, so
   ``sim.committed()``, ``sim.plan``, ``sim.gate``, ``sim.report`` and ``metadata.json`` reflect
   the child.

State crosses only through files: the restart write, JSON records and NumPy arrays, never pickled
live objects. Each child works in its own numbered stage directory under the hand-off directory
(``<output>/.fastmace-handoff/<run>/``, or ``<FASTMACE_CACHE>/handoff/<run>/`` without output), and
the parent moves on to a stage only after the child succeeded, so a failed child leaves the last
good hand-off in place. Sink values (``Simulation.add_sink``) are written by the child at each due
step and replayed to the parent's sinks, oldest first, when the child returns.

A child that fails, is killed or stalls raises ``SubprocessRunnerError`` with its message, never a
hang: the parent waits with ``RunnerConfig.timeout_s`` (whole child) and ``stall_timeout_s`` (no
progress: the child's heartbeat file names its phase and committed step) and kills the child's
process group. The hand-off directory is deleted on a clean ``close()`` and kept, with its path in
the error and in ``metadata.json``, after an error.

**The persistent worker (OPT-T07, ``RunnerConfig.persistent``, the default on POSIX).** The setup
child does not exit: it stays as the worker for the ``Simulation``'s lifetime, keeping its compiled
engine, and serves every later ``run()`` as a call (docs/design/opt-T07-persistent-worker.md). The
calls are the same requests in the same stage directories, and each still ends with the canonical
restart write, so the worker is §10.6's uninterrupted run with the same restart points, and a fresh
child per call (``persistent=False``, and always on Windows) the interrupted one: bit for bit equal
where the plan is reproducible. The protocol: the parent writes one JSON line per call,
``{"stage": <path>}``, to the worker's stdin; the worker answers one JSON line,
``{"stage": <path>, "status": "ok" | "error"}``, on a reply pipe (``--reply <fd>``); EOF on stdin, or
``{"exit": true}``, ends it. A failed call ends the worker, and the next ``run()`` starts a fresh child
from the last good hand-off. The worker ends on ``close()`` (``CLOSE_GRACE_S``), on garbage
collection and at the parent's exit (a ``weakref.finalize``, ``FINALIZE_GRACE_S``), and when the
parent dies (EOF on stdin while idle; the heartbeat thread sees the parent change while busy).

The child imports the same FastMACE as the parent (its package directory goes first on
``PYTHONPATH``) and runs in the parent's working directory, so relative paths in the records hold.

This module imports neither torch nor MACE at import time: the parent may not be allowed to.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import sys
import threading
import time
import traceback
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmace.config import RunnerConfig

TASK = "plan task 2-T07"
HANDOFF_NAME = ".fastmace-handoff"  # the hand-off directory under the run's output directory
CACHE_NAME = "handoff"  # without output: <FASTMACE_CACHE>/handoff/<run>/
REQUEST_NAME = "request.json"
RESULT_NAME = "result.json"
START_NAME = "start.traj"  # the parent's Atoms for the setup child (ASE trajectory: exact float64)
COMMITTED_NAME = "committed.npz"  # the committed step's output values (Engine.sync)
COMPOSITION_PREFIX = "composition."  # beside them in that file: species, masses, charges (Committed, 3-T03)
HEARTBEAT_NAME = "heartbeat"  # a fresh child's default: in its stage
CHILD_NAME = "child-{:04d}"  # per child process under the hand-off directory: .log and .heartbeat
SINKS_NAME = "sinks"
# The host writes a child applies after its resume, by ``Simulation.set_<kind>``: positions and momenta
# (owner decision D4), and species, masses, charges, the total charge (``set_charges(None, total_charge=)``)
# and the friction (3-T03 events; the next hand-off restart holds them, 3-T08). ``Simulation.update`` queues
# one write per field, in the order of its transaction.
HOST_WRITE_KINDS = ("positions", "momenta", "species", "masses", "charges", "total_charge", "friction")
REQUEST_VERSION = 1
MC_KEEP = "keep"  # a run request's "mc": keep the attached move set (Simulation.run's mc=...)
POLL_S = 0.2  # how often the parent looks at the child
LOG_TAIL_LINES = 30  # the child's message when it left no result
CLOSE_GRACE_S = 30.0  # close(): the worker's time to close its files and exit before it is killed
FINALIZE_GRACE_S = 5.0  # the same at garbage collection and at the parent's exit
PARENT_CHECK_S = 1.0  # how often a child's heartbeat thread checks that its parent is alive
RUNNER_WARNING = ("the plan runs in a child process through the subprocess runner (SPEC §8.7): one worker child "
                  "serves every run() (RunnerConfig.persistent); with a fresh child per run() each run() pays the "
                  "child's setup")
_EMPTY = object()  # no reply line yet


# --------------------------------------------------------------------------- parent


def handoff_root(output_directory: Any = None, environ: Any = None) -> Path:
    """The hand-off directory of a new run: under ``output_directory``, else under ``FASTMACE_CACHE``."""
    import uuid

    from fastmace import _cache

    run = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
    if output_directory is not None:
        return Path(output_directory) / HANDOFF_NAME / run
    return _cache.cache_root(environ) / CACHE_NAME / run


class _Array:
    """A host array with ``numpy()``, what ``Simulation``'s output queue reads from a snapshot."""

    def __init__(self, value):
        self._value = value

    def numpy(self):
        import numpy as np

        return np.array(self._value, copy=True)


class _HostSnapshot:
    """A committed step's values from a child, queued like an engine snapshot (``submit_frame``)."""

    def __init__(self, step, values):
        self.step = step
        self._values = values

    def wait(self):
        return {key: _Array(value) for key, value in self._values.items()}


class SubprocessRunner:
    """Runs a ``Simulation``'s setup and each ``run()`` call in a child process (SPEC §8.7).

    ``config``: a ``RunnerConfig``; ``None`` takes its defaults (``sys.executable``, the parent's
    environment, no total limit, a 30-minute stall limit).
    """

    def __init__(self, config: RunnerConfig | None = None):
        from fastmace.config import RunnerConfig

        self.config = RunnerConfig() if config is None else config
        self.root: Path | None = None
        self.current: Path | None = None  # the stage holding the last good hand-off
        self.children = 0
        self.seconds = 0.0
        self.failed: str | None = None  # the stage directory of a failed child
        self.last: dict[str, Any] | None = None
        self.reason: str | None = None
        self._decision = None  # the parent plan's "runner" decision, kept on the child's plans
        self._stages = 0
        self._values: dict[str, Any] | None = None
        self._composition: dict[str, Any] | None = None  # species, masses, charges of the committed step
        self._step = 0
        self._metadata: dict[str, Any] | None = None
        self._host_writes: list[tuple[str, Any]] = []
        self._reproducible = False  # the children get the reproducible process environment (2-T16)
        self.calls = 0  # setup and run() calls served (OPT-T07)
        self._worker = None  # the persistent worker's Popen, while it runs
        self._replies = None
        self._owner = None  # the process that started the worker
        self._finalizer = None
        self._log = self._heartbeat = None  # the worker's log and heartbeat files

    # ------------------------------------------------------------------ Simulation's hooks

    def setup(self, simulation: Any) -> None:
        """Build ``simulation`` in a child and take its step-0 hand-off (plan, gate, report, state)."""
        from ase.io import write

        from fastmace.io import output as output_io

        sim = simulation
        if sim.output is not None:
            directory = Path(sim.output.directory)
            existing = output_io.chunk_files(directory) if directory.is_dir() else {}
            if existing:  # as ChunkedOutput refuses it in process, before any child starts
                names = ", ".join(path.name for paths in existing.values() for path in paths)
                raise FileExistsError(f"{directory} already holds output chunks ({names}); resume the run or use a "
                                      "new directory")
        self._start(sim)
        stage = self._stage()
        write(stage / START_NAME, sim._atoms)
        record = sim._simulation_record(stage)
        if record["output"] is not None:
            record["output"]["directory"] = str(Path(sim.output.directory).resolve())
        request = {"action": "setup", "simulation": record, "check": sim.check, "started": False}
        result = self._launch(stage, request)
        self._take(sim, stage, result, gate=True, setup=True)

    def resume(self, simulation: Any, source: Any, record: dict[str, Any], output: Any) -> None:
        """Take a restart directory written elsewhere (``Simulation.resume``) as the first hand-off; the
        child resumes it, so the plan, gate and committed state are the child's."""
        sim = simulation
        self._start(sim, output)
        stage = self._stage()
        _copy(Path(source) / "restart.npz", stage / "restart.npz")
        record = json.loads(json.dumps(record))
        if record.get("output") is not None:
            record["output"]["directory"] = str(Path(record["output"]["directory"]).resolve())
        _write_json(stage / "simulation.json", record)
        self.current = stage
        sim.output = output
        sim._started = True
        self._run_child(sim, steps=None, schedule=None, check=sim.check, gate=True)

    def run(self, simulation: Any, *, steps: int | None = None, schedule: Any = None, mc: Any = ...) -> None:
        """Hand ``simulation``'s last committed step to a child, run ``steps`` there (along the attached
        schedule, or ``schedule``, a fresh ``ScheduleState`` that replaces it), and take the child's
        state, plan and records back. ``mc`` is ``Simulation.run``'s, already checked: the child attaches,
        keeps (``...``) or detaches (``None``) the move set (3-T15). ``SubprocessRunnerError`` on any child
        failure."""
        simulation._queue.drain()  # frames queued at the committed step come before the steps
        self._run_child(simulation, steps=steps, schedule=schedule, check="skip", gate=False, mc=mc)

    def host_write(self, simulation: Any, kind: str, values: Any) -> None:
        """Queue a host write (``HOST_WRITE_KINDS``); the next child applies it after its resume."""
        from fastmace.simulation import HOST_WRITE_WARNING

        if kind not in HOST_WRITE_KINDS:
            raise ValueError(f"unknown host write {kind!r}; known: {HOST_WRITE_KINDS}")
        self._host_writes.append((kind, values))
        if HOST_WRITE_WARNING not in simulation._report.extra_warnings:
            simulation._report.add_warning(HOST_WRITE_WARNING)

    def committed(self, simulation: Any) -> Any:
        """The committed step of the last hand-off (after any pending host write)."""
        from fastmace.simulation import Committed, _frozen, committed_composition

        self._apply_host_writes(simulation)
        values = self._values
        stress = values.get("stress")
        return Committed(step=self._step, positions=_frozen(values["positions"]), momenta=_frozen(values["momenta"]),
                         cell=_frozen(values["cell"]), energy=float(values["energy"][0]),
                         forces=_frozen(values["forces"]), stress=None if stress is None else _frozen(stress),
                         **committed_composition(self._composition))

    def submit_frame(self, simulation: Any, sink: Any) -> None:
        simulation._queue.submit(_HostSnapshot(self._step, self._values), sink)

    def save_restart(self, simulation: Any, path: Path) -> None:
        """Copy the last hand-off (a canonical restart point at the committed step) to ``path``.

        Both files are written durably (``io.durable``: temporary file, fsync, rename, directory
        fsync; SPEC §10.6), so an interrupted copy leaves the previous restart as it was (2-T07b).
        """
        from fastmace.simulation import _write_json as write_record

        self._apply_host_writes(simulation)
        simulation._queue.drain()
        directory = path.parent
        directory.mkdir(parents=True, exist_ok=True)
        _copy(self.current / "restart.npz", path)
        record = json.loads((self.current / "simulation.json").read_text(encoding="utf-8"))
        output = record.get("output")
        if output is not None and Path(output["directory"]).resolve() == directory.resolve():
            output["directory"] = "."
        write_record(directory / "simulation.json", record)  # canonical JSON, as Simulation.save_restart
        simulation._write_metadata()

    def metadata(self, simulation: Any) -> dict[str, Any]:
        """The child's ``metadata.json`` record with the parent's report, status and a ``subprocess`` block."""
        record = json.loads(json.dumps(self._metadata))
        timing = record.get("timing")
        if isinstance(timing, dict):
            timing["status"] = simulation._status
        record["report"] = simulation._report.to_metadata()
        from fastmace import provenance

        record["subprocess"] = provenance.subprocess_block(self)
        return record

    def record(self) -> dict[str, Any]:
        """The ``subprocess`` block of ``metadata.json`` (its shape is given to 2-T17)."""
        return {"python": self.config.resolved_python(), "reason": self.reason,
                "directory": None if self.root is None else str(self.root), "children": self.children,
                "calls": self.calls, "persistent": self.persistent, "seconds": self.seconds, "failed": self.failed,
                "last": self.last}

    @property
    def persistent(self) -> bool:
        """Whether one worker serves every call: ``RunnerConfig.persistent``, on POSIX (the reply pipe)."""
        return bool(self.config.persistent) and os.name == "posix"

    def restart_worker(self) -> None:
        """End the worker; the next call starts a fresh child from the last hand-off (a plan revision
        decided in the parent, docs/design/opt-T07-persistent-worker.md)."""
        self._stop_worker(CLOSE_GRACE_S)

    def close(self) -> None:
        """End the worker, then remove the hand-off directory, unless a child failed (then it is kept
        for the record)."""
        self._stop_worker(CLOSE_GRACE_S)
        if self.root is None or self.failed is not None:
            return
        shutil.rmtree(self.root, ignore_errors=True)
        parent = self.root.parent
        if parent.name == HANDOFF_NAME:
            try:
                parent.rmdir()  # only if empty: another run may use it
            except OSError:
                pass

    # ------------------------------------------------------------------ steps

    def _start(self, sim, output=...):
        output = sim.output if output is ... else output
        self.root = handoff_root(None if output is None else Path(output.directory).resolve())
        self.root.mkdir(parents=True, exist_ok=True)
        self._decision = next((entry for entry in sim.plan.decisions if entry[0] == "runner"), None)
        workload = getattr(sim.plan, "workload", None)
        self._reproducible = bool(workload is not None and workload.reproducible)
        self.reason = None if self._decision is None else str(self._decision[2])

    def _stage(self):
        self._stages += 1
        stage = self.root / f"{self._stages:04d}"
        stage.mkdir(parents=True)
        return stage

    def _apply_host_writes(self, sim):
        if self._host_writes:
            self._run_child(sim, steps=None, schedule=None, check="skip", gate=False)

    def _run_child(self, sim, *, steps, schedule, check, gate, mc=...):
        import numpy as np

        from fastmace.simulation import mc_moves, move_record

        stage = self._stage()
        writes = []
        for index, (kind, values) in enumerate(self._host_writes):
            name = f"host-{index}-{kind}.npy"
            np.save(stage / name, np.asarray(values, dtype=np.float64))
            writes.append([kind, name])
        request = {"action": "run", "source": str(self.current), "steps": steps,
                   "schedule": None if schedule is None else schedule.to_restart(), "started": sim._started,
                   "check": check, "host_writes": writes, "sinks": sorted({interval for interval, _ in sim._sinks}),
                   "mc": MC_KEEP if mc is ... else None if mc is None else [move_record(m) for m in mc_moves(mc)]}
        result = self._launch(stage, request)
        self._host_writes = []
        self._take(sim, stage, result, gate=gate)
        if steps:
            sim._started = True
        self._replay_sinks(sim, stage)
        if self.current != stage:  # nothing to hand on: the previous hand-off stays the current one
            shutil.rmtree(stage, ignore_errors=True)

    def _command(self, stage, *arguments):
        python = self.config.resolved_python()
        environment = dict(os.environ) if self.config.inherit_environment else {}
        environment.update(dict(self.config.environment))
        if self._reproducible:  # set before the child's CUDA starts (SPEC §8.7, owner decision 14)
            from fastmace.planner.reproducible import PROCESS_ENVIRONMENT

            environment.update(PROCESS_ENVIRONMENT)
        package_root = str(Path(__file__).resolve().parents[2])  # the directory holding fastmace/
        paths = [part for part in environment.get("PYTHONPATH", "").split(os.pathsep) if part]
        environment["PYTHONPATH"] = os.pathsep.join([package_root, *[p for p in paths if p != package_root]])
        return [python, "-m", "fastmace.engine.subprocess", str(stage), *arguments], environment

    def _child_files(self):
        """The next child's log and heartbeat files, under the hand-off directory (kept after an error)."""
        name = CHILD_NAME.format(self.children + 1)
        return self.root / f"{name}.log", self.root / f"{name}.heartbeat"

    def _popen(self, command, environment, log, **options):
        import subprocess

        from fastmace.errors import SubprocessRunnerError

        try:
            return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                    start_new_session=os.name == "posix", **options)
        except OSError as error:
            raise SubprocessRunnerError(
                f"the subprocess runner could not start its child ({command[0]}): {error}",
                why="a plan that a process constraint forbids here runs in a child (SPEC §8.7)",
                next_step="Check RunnerConfig.python and RunnerConfig.environment.", exit_code=None,
                child_message=str(error), directory=self.root) from error

    def _launch(self, stage, request):
        """Serve one call on ``stage``: by the worker, or by a fresh child; its result, or
        ``SubprocessRunnerError``."""
        config = self.config
        request = {"request_version": REQUEST_VERSION, **request,
                   "heartbeat_s": max(0.05, min(10.0, config.stall_timeout_s / 10))}
        _write_json(stage / REQUEST_NAME, request)
        self.calls += 1
        started = time.perf_counter()
        try:
            if self.persistent:
                code, stopped, pid, log = self._call_worker(stage)
            else:
                code, stopped, pid, log = self._fresh_child(stage)
        except Exception:
            self.failed = str(stage)
            raise
        seconds = time.perf_counter() - started
        self.seconds += seconds
        self.last = {"action": request["action"], "stage": stage.name, "pid": pid, "exit_code": code,
                     "seconds": seconds, "steps": request.get("steps")}
        result = _read_json(stage / RESULT_NAME)
        if code in (0, None) and stopped is None and result is not None and result.get("status") == "ok":
            return result
        self._fail(stage, request, code, stopped, result, log)

    def _fresh_child(self, stage):
        """A child for this call alone (2-T07's path): it exits when the call is done."""
        log_path, heartbeat = self._child_files()
        command, environment = self._command(stage, "--heartbeat", str(heartbeat))
        with open(log_path, "wb") as log:
            process = self._popen(command, environment, log)
            self.children += 1
            code, stopped = _wait(process, heartbeat, timeout_s=self.config.timeout_s,
                                  stall_timeout_s=self.config.stall_timeout_s)
        return code, stopped, process.pid, log_path

    def _call_worker(self, stage):
        """The call by the persistent worker: started on this stage if there is none, else sent to it.
        A worker that is gone, or belongs to another process (a fork), is replaced by a fresh one, which
        resumes the request's ``source`` like any fresh child. Returns (exit code, or None while the
        worker runs on; why it was stopped; pid; log)."""
        worker = self._worker
        if worker is not None and (self._owner != os.getpid() or worker.poll() is not None):
            self._forget_worker()
            worker = None
        if worker is not None:
            try:
                worker.stdin.write((json.dumps({"stage": str(stage)}) + "\n").encode("utf-8"))
                worker.stdin.flush()
            except OSError:  # it died between calls
                self._forget_worker()
                worker = None
        if worker is None:
            worker = self._start_worker(stage)
        outcome = _wait_reply(worker, self._replies, self._heartbeat, timeout_s=self.config.timeout_s,
                              stall_timeout_s=self.config.stall_timeout_s)
        log = self._log
        if outcome[0] == "reply" and outcome[1].get("status") == "ok":
            return None, None, worker.pid, log
        if outcome[0] == "reply":  # an error: the worker exits after replying
            try:
                code = worker.wait(timeout=CLOSE_GRACE_S)
            except Exception:
                _kill(worker)
                code = worker.wait()
            stopped = None
        else:
            code, stopped = outcome[1], outcome[2]
        self._forget_worker()
        return code, stopped, worker.pid, log

    def _start_worker(self, stage):
        log_path, heartbeat = self._child_files()
        read_end, write_end = os.pipe()
        command, environment = self._command(stage, "--heartbeat", str(heartbeat), "--worker", "--reply",
                                             str(write_end))
        import subprocess

        try:
            with open(log_path, "wb") as log:
                worker = self._popen(command, environment, log, stdin=subprocess.PIPE, pass_fds=(write_end,))
        except Exception:
            os.close(read_end)
            raise
        finally:
            os.close(write_end)
        self.children += 1
        self._worker, self._owner = worker, os.getpid()
        self._replies = _Replies(read_end)
        self._log, self._heartbeat = log_path, heartbeat
        self._finalizer = weakref.finalize(self, _terminate, worker, FINALIZE_GRACE_S, self._owner)
        return worker

    def _forget_worker(self):
        if self._finalizer is not None:
            self._finalizer.detach()
        self._worker = self._finalizer = self._replies = None

    def _stop_worker(self, grace):
        worker = self._worker
        if worker is None:
            return
        owner = self._owner
        self._forget_worker()
        _terminate(worker, grace, owner)

    def _fail(self, stage, request, code, stopped, result, log):
        from fastmace.errors import SubprocessRunnerError

        self.failed = str(stage)
        if result is not None and result.get("status") == "error":
            error = result["error"]
            message = f"{error['type']}: {error['message']}"
        else:
            message = _tail(log)
        if stopped is not None:
            what = f"the subprocess runner's child {stopped} and was killed"
        elif code is not None and code < 0:
            what = f"the subprocess runner's child was killed by signal {-code}"
        else:
            what = f"the subprocess runner's child failed (exit code {code})"
        first = message.strip().splitlines()[-1] if message.strip() else "no message"
        raise SubprocessRunnerError(
            f"{what}: {first}",
            why=f"the plan runs in a child process (SPEC §8.7); the child's log is {log}",
            exit_code=code, child_message=message, directory=self.root)

    def _take(self, sim, stage, result, *, gate, setup=False):
        """Make the parent reflect the child: committed values, plan, gate, schedule, report, records."""
        import dataclasses

        import numpy as np

        from fastmace.planner.plan import Plan
        from fastmace.planner.selfcheck import GateRecord
        from fastmace.protocols import ScheduleState

        with np.load(stage / COMMITTED_NAME, allow_pickle=False) as stored:
            values = {key: stored[key] for key in stored.files}
        self._composition = {key[len(COMPOSITION_PREFIX):]: values.pop(key) for key in list(values)
                             if key.startswith(COMPOSITION_PREFIX)}
        self._values = values  # the output values, as the sinks and frames get them
        self._step = int(result["step"])
        if result["saved"]:
            previous, self.current = self.current, stage
            if previous is not None and previous != stage:
                shutil.rmtree(previous, ignore_errors=True)
        plan = Plan.from_restart(result["plan"])
        decisions = plan.decisions
        if self._decision is not None and all(entry[0] != "runner" for entry in decisions):
            decisions = (*decisions, self._decision)
        sim._plan = dataclasses.replace(plan, runner="subprocess", decisions=decisions)
        if gate:
            fields = dict(result["gate"])
            for name in ("nonfinite", "notes"):
                fields[name] = tuple(fields.get(name) or ())
            sim._gate = GateRecord(**fields)
        sim._schedule = None if result["schedule"] is None else ScheduleState.from_restart(result["schedule"])
        if result.get("mc") is not None:  # the child's MC record for the report (3-T15)
            from fastmace.simulation import move_from_record

            sim._report.mc = {**result["mc"], "moves": tuple(move_from_record(m) for m in result["mc"]["moves"])}
        sim._restart_points = [dict(point) for point in result["restart_points"]]
        sim._status = result["simulation_status"]
        sim._steps_run = result["steps_run"]
        sim._loop_s = result["loop_s"]
        self._metadata = result["metadata"]
        report = sim._report
        report.plan = sim._plan
        report.gate = None if sim._gate.status == "skipped" else sim._gate.to_metadata()
        if setup:
            report.setup_s = {"plan": sim._setup_s.get("plan", 0.0), **{
                f"child {stage_name}": float(seconds) for stage_name, seconds in result["setup_s"].items()},
                "child process": max(0.0, self.seconds - sum(result["setup_s"].values()))}
            sim._setup_s = dict(report.setup_s)
        child = result["report"]
        report.speed = dict(child["speed"])
        report.measured = None if child["measured_window"] is None else dict(child["measured_window"])
        window = child["speed_window"]
        report.start_step = int(window["start_step"])
        first = report.start_step + int(window["warmup_steps"])
        report.window = (first, first + int(window["window_steps"]))
        for text in result["extra_warnings"]:
            if text not in report.extra_warnings:
                report.add_warning(text)
        sim._write_metadata()

    def _replay_sinks(self, sim, stage):
        import numpy as np

        directory = stage / SINKS_NAME
        if not directory.is_dir():
            return
        for path in sorted(directory.glob("*.npz")):
            step = int(path.stem)
            for interval, sink in sim._sinks:
                if step % interval == 0:
                    with np.load(path, allow_pickle=False) as stored:
                        values = {key: stored[key] for key in stored.files}
                    sink(step, values)
        shutil.rmtree(directory, ignore_errors=True)


def _wait(process, heartbeat, *, timeout_s, stall_timeout_s, poll_s=POLL_S, clock=time.monotonic):
    """Wait for ``process``; kill its process group if it runs longer than ``timeout_s`` or its
    ``heartbeat`` file does not change for ``stall_timeout_s``. Returns (exit code, why it was stopped
    or None)."""
    import subprocess

    started = last = clock()
    seen = None
    while True:
        try:
            return process.wait(timeout=poll_s), None
        except subprocess.TimeoutExpired:
            pass
        now = clock()
        beat = _read_text(heartbeat)
        if beat != seen:
            seen, last = beat, now
        if timeout_s is not None and now - started > timeout_s:
            why = f"ran longer than RunnerConfig.timeout_s={timeout_s:g} s"
        elif now - last > stall_timeout_s:
            why = (f"made no progress for RunnerConfig.stall_timeout_s={stall_timeout_s:g} s (last heartbeat: "
                   f"{seen or 'none'})")
        else:
            continue
        _kill(process)
        return process.wait(), why


class _Replies:
    """The worker's reply lines, read by a thread into a queue; ``None`` marks the pipe's end."""

    def __init__(self, fd):
        self._queue = queue.Queue()
        self.ended = False
        if fd is not None:
            handle = os.fdopen(fd, "r", encoding="utf-8")
            threading.Thread(target=self._read, args=(handle,), name="fastmace-replies", daemon=True).start()

    def _read(self, handle):
        with handle:
            for line in handle:
                self._queue.put(line)
        self._queue.put(None)

    def get(self, timeout):
        """The next line, ``None`` at the end, or ``_EMPTY`` after ``timeout`` seconds."""
        if self.ended:
            time.sleep(timeout)
            return _EMPTY
        try:
            line = self._queue.get(timeout=timeout)
        except queue.Empty:
            return _EMPTY
        if line is None:
            self.ended = True
        return line


def _wait_reply(process, replies, heartbeat, *, timeout_s, stall_timeout_s, poll_s=POLL_S, clock=time.monotonic):
    """Wait for the worker's reply to a call. Kill its process group if the call runs longer than
    ``timeout_s`` or its ``heartbeat`` does not change for ``stall_timeout_s``. Returns
    ``("reply", record)``, ``("exit", exit code, None)`` or ``("stopped", exit code, why)``."""
    started = last = clock()
    seen = None
    while True:
        line = replies.get(poll_s)
        if line is not _EMPTY and line is not None:
            try:
                return "reply", json.loads(line)
            except ValueError:
                return "reply", {"status": "error", "line": line}
        code = process.poll()
        if code is not None:
            return "exit", code, None
        now = clock()
        beat = _read_text(heartbeat)
        if beat != seen:
            seen, last = beat, now
        if timeout_s is not None and now - started > timeout_s:
            why = f"ran longer than RunnerConfig.timeout_s={timeout_s:g} s"
        elif now - last > stall_timeout_s:
            why = (f"made no progress for RunnerConfig.stall_timeout_s={stall_timeout_s:g} s (last heartbeat: "
                   f"{seen or 'none'})")
        else:
            continue
        _kill(process)
        return "stopped", process.wait(), why


def _terminate(process, grace, owner):
    """End a worker: close its stdin (it closes its files and exits), wait ``grace`` seconds, then kill
    its process group. Only the process that started it does this (not a fork of it)."""
    import subprocess

    if os.getpid() != owner or process.poll() is not None:
        return
    try:
        process.stdin.close()
    except OSError:
        pass
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        _kill(process)
        process.wait()


def _kill(process):
    import signal

    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)  # the child's own session: its workers too, nobody else
        else:
            process.kill()
    except (ProcessLookupError, PermissionError):
        pass


def _read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _read_json(path):
    text = _read_text(path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _tail(path, lines=LOG_TAIL_LINES):
    text = _read_text(path) or ""
    return "\n".join(text.splitlines()[-lines:])


def _write_json(path, record):
    """JSON written durably and atomically (``io.durable.atomic_write``): the records are read while
    they change, and a hand-off's records must survive a crash as its restart does."""
    from fastmace.io import durable

    durable.atomic_write(path, json.dumps(record, indent=1, allow_nan=False) + "\n")


def _copy(source, target):
    """Copy ``source`` to ``target`` durably and atomically: an interrupted copy leaves ``target`` as it
    was (``io.durable.write_atomically``)."""
    from fastmace.io import durable

    def write(handle):
        with open(source, "rb") as original:
            shutil.copyfileobj(original, handle)

    durable.write_atomically(target, write)


# --------------------------------------------------------------------------- child


class _Heartbeat:
    """The child's progress file: its phase and the engine's step, rewritten when they change.

    A thread looks every ``interval`` seconds, so the step loop is never touched; a main thread that
    hangs (in Python or in a kernel) leaves the file unchanged, which the parent counts as a stall.

    Both that thread and the main thread (``mark``) write the file. One lock makes them take turns,
    and each write has its own temporary name, so no rename can find another writer's file gone
    (2-T07c: ``FileNotFoundError`` under load). A failed write never raises: a heartbeat that does not
    change is at worst read as a stall by the parent; it must not end the call.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.parent = os.getppid()  # a child whose parent died ends itself (OPT-T07)
        self.phase = "started"
        self.simulation = None
        self.interval = 1.0
        self._written = None
        self._lock = threading.Lock()
        self._writes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="fastmace-heartbeat", daemon=True)
        self.write()

    def text(self):
        sim = self.simulation
        engine = None if sim is None else getattr(sim, "_engine", None)
        step = None if engine is None else int(engine.state.step)
        return f"{self.phase} {step}" if step is not None else self.phase

    def write(self):
        # An atomic rename without fsync: the heartbeat only shows liveness to the waiting parent,
        # is rewritten every few seconds, and means nothing after a crash of the machine.
        with self._lock:
            try:
                text = self.text()
            except Exception:  # an engine half built: the next write has it
                return
            if text == self._written:
                return
            self._writes += 1
            temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.{self._writes}.tmp")
            try:
                temporary.write_text(text, encoding="utf-8")
                os.replace(temporary, self.path)
                self._written = text
            except Exception:  # never let the heartbeat end the child
                try:
                    temporary.unlink()
                except OSError:
                    pass

    def mark(self, phase):
        self.phase = phase
        self.write()

    def start(self, interval):
        self.interval = float(interval)
        self._thread.start()

    def stop(self):
        # Wait for the thread: a write calls torch (the engine's step), and a daemon thread still inside torch
        # when the interpreter exits is ended by pthread_exit, whose unwind through torch's noexcept GIL
        # re-acquire calls std::terminate: SIGABRT after a written result (4-T13). One write at most.
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=PARENT_CHECK_S)

    def _loop(self):
        # Every PARENT_CHECK_S at least, for the parent check; the file is rewritten only on a change.
        while not self._stop.wait(min(self.interval, PARENT_CHECK_S)):
            if os.name == "posix" and os.getppid() != self.parent:
                os._exit(3)  # the parent is gone: nobody will read this child's results; free the GPU
            self.write()  # never raises


def main(argv: list[str] | None = None) -> int:
    """The child's entry point: ``python -m fastmace.engine.subprocess <directory> [--heartbeat <path>]
    [--worker --reply <fd>]``.

    Reads ``request.json`` in ``directory``, does it, and writes ``result.json`` (status ``ok``, or
    ``error`` with the exception's type and message, also on stderr). A fresh child then exits (code 0
    or 1). A worker (``--worker``) answers on the reply pipe and serves the stages its stdin names until
    EOF or ``{"exit": true}``; a failed call ends it with code 1.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="python -m fastmace.engine.subprocess")
    parser.add_argument("directory")
    parser.add_argument("--heartbeat")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--reply", type=int)
    try:
        arguments = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    except SystemExit as exit:
        return int(exit.code or 2)
    directory = Path(arguments.directory)
    heartbeat = _Heartbeat(arguments.heartbeat or directory / HEARTBEAT_NAME)
    reply = None if arguments.reply is None else os.fdopen(arguments.reply, "w", encoding="utf-8")
    worker = _Worker(persistent=arguments.worker)
    try:
        while True:
            ok = _serve(directory, heartbeat, worker)
            if reply is not None:
                reply.write(json.dumps({"stage": str(directory), "status": "ok" if ok else "error"}) + "\n")
                reply.flush()
            if not ok:
                return 1
            if not arguments.worker:
                return 0
            heartbeat.mark("idle")
            line = sys.stdin.readline()
            if not line:
                return 0  # EOF: the parent closed the worker, or died
            command = json.loads(line)
            if command.get("exit"):
                return 0
            directory = Path(command["stage"])
    finally:
        heartbeat.simulation = None  # no last write reads the engine
        heartbeat.stop()
        worker.close()


class _Worker:
    """What a child keeps between calls: its ``Simulation`` and the stage of its last hand-off."""

    def __init__(self, persistent):
        self.persistent = persistent
        self.simulation = None
        self.stage = None

    def close(self):
        """Close the output files; nothing else (the parent writes the final records)."""
        sim = self.simulation
        if sim is not None and sim._writer is not None:
            sim._writer.close()
            sim._writer = None


def _serve(directory, heartbeat, worker):
    """One call: its result in ``directory``; False if it failed."""
    try:
        request = json.loads((directory / REQUEST_NAME).read_text(encoding="utf-8"))
        if not heartbeat._thread.is_alive():
            heartbeat.start(request.get("heartbeat_s", 1.0))
        result = _child(directory, request, heartbeat, worker)
    except Exception as error:  # the parent raises SubprocessRunnerError with this message
        traceback.print_exc()
        sys.stderr.flush()
        _write_json(directory / RESULT_NAME, {"status": "error", "error": {
            "type": type(error).__name__, "module": type(error).__module__, "message": str(error),
            "traceback": traceback.format_exc()}})
        return False
    _write_json(directory / RESULT_NAME, result)
    return True


def _child(directory, request, heartbeat, worker=None):
    import fastmace  # noqa: F401  (before torch: file locks, SPEC §14)

    # isort: split
    import numpy as np

    from fastmace import simulation as simulations
    from fastmace.protocols import ScheduleState

    if request.get("request_version") != REQUEST_VERSION:
        raise ValueError(f"hand-off request version {request.get('request_version')!r} is not {REQUEST_VERSION}")
    worker = _Worker(persistent=False) if worker is None else worker
    action = request["action"]
    if action not in ("setup", "run"):
        raise ValueError(f"unknown hand-off action {action!r}")
    if worker.simulation is not None:  # the worker's next call: the engine it has (OPT-T07)
        if action != "run" or str(request["source"]) != str(worker.stage):
            raise ValueError(f"the worker holds the hand-off of {worker.stage}, the request continues "
                             f"{request.get('source')!r} ({action})")
        sim = worker.simulation
        heartbeat.mark("call")
    else:
        heartbeat.mark("setup" if action == "setup" else "resume")
        if action == "setup":
            sim = _construct(directory, request)
        else:
            sim = simulations.Simulation.resume(request["source"], allow_plan_change=True, check=request["check"])
            worker.stage = request["source"]
        worker.simulation = sim
    sim._sinks = []  # this call's sinks only
    heartbeat.simulation = sim
    heartbeat.mark("running")
    if not request["started"]:
        sim._started = False  # the step-0 output belongs to the first run(), as in process
    for kind, name in request.get("host_writes", ()):
        values = np.load(directory / name, allow_pickle=False)
        if kind not in HOST_WRITE_KINDS:
            raise ValueError(f"unknown host write {kind!r}")
        if kind == "species":  # stored as float64 like every host write; atomic numbers are exact there
            values = values.astype(np.int64)
        elif values.ndim == 0:  # a scalar: the friction or the total charge
            values = values.item()
        if kind == "total_charge":
            sim.set_charges(None, total_charge=values)
        else:
            getattr(sim, f"set_{kind}")(values)  # a composition change ends at a canonical point (3-T03)
    sinks = directory / SINKS_NAME
    written = set()

    def save(step, values):
        if step not in written:
            written.add(step)
            np.savez(sinks / f"{step:012d}.npz", **values)

    intervals = request.get("sinks", ())
    if intervals:
        sinks.mkdir(exist_ok=True)
        for interval in intervals:
            sim.add_sink(save, int(interval))
    if request.get("schedule") is not None:
        sim._schedule = ScheduleState.from_restart(request["schedule"])
        sim._engine.state.protocol = sim._schedule
    steps = request.get("steps")
    if steps:
        mc = request.get("mc", MC_KEEP)
        if mc == MC_KEEP:
            sim.run(steps=int(steps))
        else:  # attach or detach (3-T15); the parent checked the move set against the workload
            sim.run(steps=int(steps), mc=None if mc is None else [simulations.move_from_record(m) for m in mc])
    saved = action == "setup" or bool(steps) or bool(request.get("host_writes"))
    if saved:
        sim.save_restart(directory)  # the next hand-off: a canonical restart point (§10.6)
        worker.stage = str(directory)
    else:
        sim.flush_output()
    if not worker.persistent:  # a fresh child: the chunk opened at the restart point waits for the next child
        worker.close()
    engine = sim._engine
    values = dict(engine.sync())
    if "cell" not in values:
        values["cell"] = engine.state.cell.detach().cpu().numpy()
    values.update({COMPOSITION_PREFIX + key: value
                   for key, value in simulations.composition_values(engine.state).items()})
    np.savez(directory / COMMITTED_NAME, **values)
    report = sim.report.to_metadata()
    return {
        "status": "ok",
        "simulation_status": sim._status,
        "saved": saved,
        "step": int(engine.state.step),
        "plan": sim.plan.to_restart(**engine.restart_buckets()),
        "gate": sim.gate.to_metadata(),
        "schedule": None if sim.schedule is None else sim.schedule.to_restart(),
        "restart_points": list(sim._restart_points),
        "steps_run": sim._steps_run,
        "loop_s": sim._loop_s,
        "setup_s": {name: float(seconds) for name, seconds in sim._setup_s.items()},
        "report": report,
        "extra_warnings": list(sim.report.extra_warnings),
        "metadata": sim.metadata(),
        "mc": _mc_summary(sim),
    }


def _mc_summary(sim):
    """The child's MC record for the parent's report (``Simulation._mc_summary``), JSON-safe, or ``None``."""
    from fastmace.simulation import move_record

    summary = sim._mc_summary()
    if summary is None:
        return None
    return {**summary, "moves": [move_record(move) for move in summary["moves"]]}


def _construct(directory, request):
    """The setup child's ``Simulation``, from the parent's record and ``Atoms`` (as ``Simulation.resume``
    builds one from ``simulation.json``)."""
    from ase.io import read

    from fastmace import simulation as simulations
    from fastmace.io.output import Output

    record = request["simulation"]
    options = simulations.PlannerOptions.from_record(record.get("planner_options"))
    output = record["output"]
    return simulations.Simulation(
        read(directory / START_NAME), simulations._record_force_field(record),
        integrator=simulations._integrator(record["integrator"]), workload=simulations._workload(record["workload"]),
        output=None if output is None else Output(**output), seed=record["seed"], check=request["check"],
        allow_naive_cueq=record["allow_naive_cueq"], mode=options.mode, allow_reference=options.allow_reference,
        fallback=options.fallback, gate=options.gate, runner=options.runner, engine=options.engine,
        neighbours=options.neighbours)


if __name__ == "__main__":
    raise SystemExit(main())
