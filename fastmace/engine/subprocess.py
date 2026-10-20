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

The child imports the same FastMACE as the parent (its package directory goes first on
``PYTHONPATH``) and runs in the parent's working directory, so relative paths in the records hold.

This module imports neither torch nor MACE at import time: the parent may not be allowed to.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
import traceback
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
HEARTBEAT_NAME = "heartbeat"
LOG_NAME = "child.log"
SINKS_NAME = "sinks"
REQUEST_VERSION = 1
POLL_S = 0.2  # how often the parent looks at the child
LOG_TAIL_LINES = 30  # the child's message when it left no result
RUNNER_WARNING = ("the plan runs in a child process through the subprocess runner (SPEC §8.7): each run() "
                  "starts a child that resumes the last restart write, so the setup cost is paid per run()")


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
        self._step = 0
        self._metadata: dict[str, Any] | None = None
        self._host_writes: list[tuple[str, Any]] = []
        self._reproducible = False  # the children get the reproducible process environment (2-T16)

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
        shutil.copyfile(Path(source) / "restart.npz", stage / "restart.npz")
        record = json.loads(json.dumps(record))
        if record.get("output") is not None:
            record["output"]["directory"] = str(Path(record["output"]["directory"]).resolve())
        _write_json(stage / "simulation.json", record)
        self.current = stage
        sim.output = output
        sim._started = True
        self._run_child(sim, steps=None, schedule=None, check=sim.check, gate=True)

    def run(self, simulation: Any, *, steps: int | None = None, schedule: Any = None) -> None:
        """Hand ``simulation``'s last committed step to a child, run ``steps`` there (along the attached
        schedule, or ``schedule``, a fresh ``ScheduleState`` that replaces it), and take the child's
        state, plan and records back. ``SubprocessRunnerError`` on any child failure."""
        simulation._queue.drain()  # frames queued at the committed step come before the steps
        self._run_child(simulation, steps=steps, schedule=schedule, check="skip", gate=False)

    def host_write(self, simulation: Any, kind: str, values: Any) -> None:
        """Queue a host write of positions or momenta; the next child applies it after its resume."""
        from fastmace.simulation import HOST_WRITE_WARNING

        self._host_writes.append((kind, values))
        if HOST_WRITE_WARNING not in simulation._report.extra_warnings:
            simulation._report.add_warning(HOST_WRITE_WARNING)

    def committed(self, simulation: Any) -> Any:
        """The committed step of the last hand-off (after any pending host write)."""
        from fastmace.simulation import Committed, _frozen

        self._apply_host_writes(simulation)
        values = self._values
        stress = values.get("stress")
        return Committed(step=self._step, positions=_frozen(values["positions"]), momenta=_frozen(values["momenta"]),
                         cell=_frozen(values["cell"]), energy=float(values["energy"][0]),
                         forces=_frozen(values["forces"]), stress=None if stress is None else _frozen(stress))

    def submit_frame(self, simulation: Any, sink: Any) -> None:
        simulation._queue.submit(_HostSnapshot(self._step, self._values), sink)

    def save_restart(self, simulation: Any, path: Path) -> None:
        """Copy the last hand-off (a canonical restart point at the committed step) to ``path``."""
        self._apply_host_writes(simulation)
        simulation._queue.drain()
        directory = path.parent
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.current / "restart.npz", path)
        record = json.loads((self.current / "simulation.json").read_text(encoding="utf-8"))
        output = record.get("output")
        if output is not None and Path(output["directory"]).resolve() == directory.resolve():
            output["directory"] = "."
        _write_json(directory / "simulation.json", record)
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
                "seconds": self.seconds, "failed": self.failed, "last": self.last}

    def close(self) -> None:
        """Remove the hand-off directory, unless a child failed (then it is kept for the record)."""
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

    def _run_child(self, sim, *, steps, schedule, check, gate):
        import numpy as np

        stage = self._stage()
        writes = []
        for index, (kind, values) in enumerate(self._host_writes):
            name = f"host-{index}-{kind}.npy"
            np.save(stage / name, np.asarray(values, dtype=np.float64))
            writes.append([kind, name])
        request = {"action": "run", "source": str(self.current), "steps": steps,
                   "schedule": None if schedule is None else schedule.to_restart(), "started": sim._started,
                   "check": check, "host_writes": writes, "sinks": sorted({interval for interval, _ in sim._sinks})}
        result = self._launch(stage, request)
        self._host_writes = []
        self._take(sim, stage, result, gate=gate)
        if steps:
            sim._started = True
        self._replay_sinks(sim, stage)
        if self.current != stage:  # nothing to hand on: the previous hand-off stays the current one
            shutil.rmtree(stage, ignore_errors=True)

    def _command(self, stage):
        python = self.config.resolved_python()
        environment = dict(os.environ) if self.config.inherit_environment else {}
        environment.update(dict(self.config.environment))
        if self._reproducible:  # set before the child's CUDA starts (SPEC §8.7, owner decision 14)
            from fastmace.planner.reproducible import PROCESS_ENVIRONMENT

            environment.update(PROCESS_ENVIRONMENT)
        package_root = str(Path(__file__).resolve().parents[2])  # the directory holding fastmace/
        paths = [part for part in environment.get("PYTHONPATH", "").split(os.pathsep) if part]
        environment["PYTHONPATH"] = os.pathsep.join([package_root, *[p for p in paths if p != package_root]])
        return [python, "-m", "fastmace.engine.subprocess", str(stage)], environment

    def _launch(self, stage, request):
        """Run one child on ``stage``; its result, or ``SubprocessRunnerError``."""
        import subprocess

        from fastmace.errors import SubprocessRunnerError

        config = self.config
        request = {"request_version": REQUEST_VERSION, **request,
                   "heartbeat_s": max(0.05, min(10.0, config.stall_timeout_s / 10))}
        _write_json(stage / REQUEST_NAME, request)
        command, environment = self._command(stage)
        started = time.perf_counter()
        with open(stage / LOG_NAME, "wb") as log:
            try:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                           start_new_session=os.name == "posix")
            except OSError as error:
                self.failed = str(stage)
                raise SubprocessRunnerError(
                    f"the subprocess runner could not start its child ({command[0]}): {error}",
                    why="a plan that a process constraint forbids here runs in a child (SPEC §8.7)",
                    next_step="Check RunnerConfig.python and RunnerConfig.environment.", exit_code=None,
                    child_message=str(error), directory=self.root) from error
            code, stopped = _wait(process, stage / HEARTBEAT_NAME, timeout_s=config.timeout_s,
                                  stall_timeout_s=config.stall_timeout_s)
        seconds = time.perf_counter() - started
        self.children += 1
        self.seconds += seconds
        self.last = {"action": request["action"], "stage": stage.name, "pid": process.pid, "exit_code": code,
                     "seconds": seconds, "steps": request.get("steps")}
        result = _read_json(stage / RESULT_NAME)
        if code == 0 and stopped is None and result is not None and result.get("status") == "ok":
            return result
        self.failed = str(stage)
        if result is not None and result.get("status") == "error":
            error = result["error"]
            message = f"{error['type']}: {error['message']}"
        else:
            message = _tail(stage / LOG_NAME)
        if stopped is not None:
            what = f"the subprocess runner's child {stopped} and was killed"
        elif code is not None and code < 0:
            what = f"the subprocess runner's child was killed by signal {-code}"
        else:
            what = f"the subprocess runner's child failed (exit code {code})"
        first = message.strip().splitlines()[-1] if message.strip() else "no message"
        raise SubprocessRunnerError(
            f"{what}: {first}",
            why=f"the plan runs in a child process (SPEC §8.7); the child's log is {stage / LOG_NAME}",
            exit_code=code, child_message=message, directory=self.root)

    def _take(self, sim, stage, result, *, gate, setup=False):
        """Make the parent reflect the child: committed values, plan, gate, schedule, report, records."""
        import dataclasses

        import numpy as np

        from fastmace.planner.plan import Plan
        from fastmace.planner.selfcheck import GateRecord
        from fastmace.protocols import ScheduleState

        with np.load(stage / COMMITTED_NAME, allow_pickle=False) as stored:
            self._values = {key: stored[key] for key in stored.files}
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
    """JSON written atomically: temporary file, then rename (the heartbeat and records are read while
    they change)."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, indent=1, allow_nan=False) + "\n")
    os.replace(temporary, path)


# --------------------------------------------------------------------------- child


class _Heartbeat:
    """The child's progress file: its phase and the engine's step, rewritten when they change.

    A thread looks every ``interval`` seconds, so the step loop is never touched; a main thread that
    hangs (in Python or in a kernel) leaves the file unchanged, which the parent counts as a stall.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.phase = "started"
        self.simulation = None
        self.interval = 1.0
        self._written = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="fastmace-heartbeat", daemon=True)
        self.write()

    def text(self):
        sim = self.simulation
        engine = None if sim is None else getattr(sim, "_engine", None)
        step = None if engine is None else int(engine.state.step)
        return f"{self.phase} {step}" if step is not None else self.phase

    def write(self):
        text = self.text()
        if text != self._written:
            temporary = self.path.with_name(f".{self.path.name}.tmp")
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, self.path)
            self._written = text

    def mark(self, phase):
        self.phase = phase
        self.write()

    def start(self, interval):
        self.interval = float(interval)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                self.write()
            except Exception:  # never let the heartbeat end the child
                pass


def main(argv: list[str] | None = None) -> int:
    """The child's entry point: ``python -m fastmace.engine.subprocess <directory>``.

    Reads ``request.json`` in ``directory``, does it, and writes ``result.json`` (status ``ok``, or
    ``error`` with the exception's type and message, also on stderr). Exit code 0 or 1.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) != 1:
        print("usage: python -m fastmace.engine.subprocess <hand-off stage directory>", file=sys.stderr)
        return 2
    directory = Path(argv[0])
    heartbeat = _Heartbeat(directory / HEARTBEAT_NAME)
    try:
        request = json.loads((directory / REQUEST_NAME).read_text(encoding="utf-8"))
        heartbeat.start(request.get("heartbeat_s", 1.0))
        result = _child(directory, request, heartbeat)
    except Exception as error:  # the parent raises SubprocessRunnerError with this message
        traceback.print_exc()
        sys.stderr.flush()
        _write_json(directory / RESULT_NAME, {"status": "error", "error": {
            "type": type(error).__name__, "module": type(error).__module__, "message": str(error),
            "traceback": traceback.format_exc()}})
        return 1
    finally:
        heartbeat.stop()
    _write_json(directory / RESULT_NAME, result)
    return 0


def _child(directory, request, heartbeat):
    import fastmace  # noqa: F401  (before torch: file locks, SPEC §14)

    # isort: split
    import numpy as np

    from fastmace import simulation as simulations
    from fastmace.protocols import ScheduleState

    if request.get("request_version") != REQUEST_VERSION:
        raise ValueError(f"hand-off request version {request.get('request_version')!r} is not {REQUEST_VERSION}")
    action = request["action"]
    heartbeat.mark("setup" if action == "setup" else "resume")
    if action == "setup":
        sim = _construct(directory, request)
    elif action == "run":
        sim = simulations.Simulation.resume(request["source"], allow_plan_change=True, check=request["check"])
    else:
        raise ValueError(f"unknown hand-off action {action!r}")
    heartbeat.simulation = sim
    heartbeat.mark("running")
    if not request["started"]:
        sim._started = False  # the step-0 output belongs to the first run(), as in process
    for kind, name in request.get("host_writes", ()):
        values = np.load(directory / name, allow_pickle=False)
        if kind == "positions":
            sim.set_positions(values)
        elif kind == "momenta":
            sim.set_momenta(values)
        else:
            raise ValueError(f"unknown host write {kind!r}")
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
        sim.run(steps=int(steps))
    saved = action == "setup" or bool(steps) or bool(request.get("host_writes"))
    if saved:
        sim.save_restart(directory)  # the next hand-off: a canonical restart point (§10.6)
    else:
        sim.flush_output()
    if sim._writer is not None:  # the chunk opened at the restart point stays empty until the next child
        sim._writer.close()
        sim._writer = None
    engine = sim._engine
    values = dict(engine.sync())
    if "cell" not in values:
        values["cell"] = engine.state.cell.detach().cpu().numpy()
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
    }


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
