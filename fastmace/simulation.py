"""Simulation: one structure, one force field, one integrator, run in segments (SPEC §7.2, §7.6).

Plan task 1b-T01 fixed the signature; plan task 1b-T17 implements it. SPEC §5 lists no module for
``Simulation``; it lives here, above ``planner`` and ``engine``, and gets its engine from the
minimal planner (1b-T12), never by building one (SPEC §5).

What the constructor does, in this order (``docs/design/contracts-1b.md``):

1. with a model instead of a ``ForceField``, build ``ForceField(model, atoms, precision="float32",
   species=species)`` (§7.6 "Simulation inputs");
2. ``workload=None`` becomes ``Workload(max_temperature_K=integrator.temperature_K)`` (owner
   decision D7), recorded in the plan;
3. plan through the minimal planner; ``PlanRefused`` here carries every reason;
4. with ``check="gate"`` or ``"allow_unavailable"``: the gate's references (§12.3, 1b-T16),
   before any production engine exists in the process;
5. build the engine through the planner;
6. the gate's production half through that engine (captured replay, registry snapshot and
   restore), then ``GateFailed`` or ``GateUnavailable`` if it says so; the record is ``sim.gate``.
   Without a requested gate its status is ``skipped`` (owner decision D1).

So a constructed ``Simulation`` is ready to step, and every refusal comes before the first step.

The run loop is 1a's (``langevin/workflow.py:run_crystal`` as driven through
``langevin/fastmace_engine.py``): the schedule's target is filled before each step where it
changes (``protocols.TargetFill``), and output is asynchronous: a thermo or frame step queues a
snapshot that is written after the next step has been launched. ``Simulation`` owns the step
timing that the run report (1b-T15) shows: the steady-state window after warm-up, and the
whole-run mean at the end of each ``run``.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np

    from fastmace.io.output import Output
    from fastmace.planner.workload import Workload

Check = Literal["skip", "gate", "allow_unavailable"]
# check="skip": the gate does not run, status "skipped" (the default, owner decision D1).
# check="gate": run the gate before the first step; failed -> GateFailed, unavailable -> GateUnavailable.
# check="allow_unavailable": as "gate", but an unavailable gate is recorded and the run goes on.
CHECK_MODES = ("skip", "gate", "allow_unavailable")

GATE_SKIPPED = {"status": "skipped", "reason": 'opt-in: check="gate"'}
THERMO_HEADER = ("step", "time_ps", "phase", "target_K", "temperature_K", "potential_eV_atom", "kinetic_eV_atom",
                 "total_eV_atom")
NPT_COLUMNS = ("volume_A3_atom", "pressure_GPa")
CONSTANT_PHASE = "constant"  # the thermo phase column without a schedule
ENGINE_NAME = "fastmace"  # the restart file's engine entry
RESTART_NAME = "restart.npz"  # io.restart.RESTART_NAME (that module imports torch)
SIMULATION_NAME = "simulation.json"  # next to the restart: what the restart format does not hold
SIMULATION_RECORD_VERSION = 1
HOST_WRITE_WARNING = ("host writes of positions or momenta run at a canonical point (quiesce, write, rebuild, "
                      "re-evaluate; owner decision D4): correct, and slower than a step")


def check_mode(check: Any) -> str:
    """``check`` if it is one of ``CHECK_MODES``; else ``ValueError`` naming them."""
    if check not in CHECK_MODES:
        raise ValueError(f"check={check!r} is not one of {CHECK_MODES}; the gate is opt-in: check='gate' (SPEC §12.3)")
    return check


def _selfcheck():
    """The gate module (1b-T16), or ``None`` while it is not part of this FastMACE."""
    try:
        from fastmace.planner import selfcheck
    except ImportError:
        return None
    return selfcheck


def _steps(steps: Any) -> int:
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError(f"steps must be an int, got {steps!r}")
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    return steps


@dataclass(frozen=True)
class Committed:
    """The committed step as read-only host arrays: what an observer sees (``Simulation.committed``).

    ``stress`` (3×3, eV/Å³) is ``None`` for fixed-cell integrators; ``energy`` in eV.
    """

    step: int
    positions: np.ndarray
    momenta: np.ndarray
    cell: np.ndarray
    energy: float
    forces: np.ndarray
    stress: np.ndarray | None


def _frozen(array):
    import numpy as np

    out = np.array(array, copy=True)
    out.setflags(write=False)
    return out


class _Queue:
    """Snapshots waiting to be written, oldest first, each with its writer ``callback(step, values)``."""

    def __init__(self):
        self._pending = deque()

    def submit(self, snapshot, callback) -> None:
        self._pending.append((snapshot, callback))

    def drain(self) -> None:
        while self._pending:
            snapshot, callback = self._pending.popleft()
            values = {key: value.numpy() for key, value in snapshot.wait().items()}
            callback(snapshot.step, values)

    def __len__(self) -> int:
        return len(self._pending)


class Simulation:
    """MD on the graph engine with a plan, an optional gate, schedules and complete restarts.

    ``Simulation(atoms, force_field, *, species=None, integrator, workload=None, output=None, seed,
    check="skip", allow_naive_cueq=False)``:

    - ``atoms``: ASE ``Atoms``, full 3D PBC (§4.4); copied, the caller's object is never changed;
    - ``force_field``: a ``fm.ForceField``, an ``fm.load`` result or a label; a model is wrapped in
      a float32 ``ForceField`` with ``species=`` (default: the structure's species). A path raises
      ``TypeError`` (load it with ``fm.load``, the trust decision needs one); ``species`` with a
      ``ForceField`` raises ``ValueError`` (the ``ForceField`` owns the superset);
    - ``integrator``: ``fm.integrators.Langevin`` or ``fm.integrators.LangevinBAOAB``;
    - ``workload``: ``fm.Workload``; default ``Workload(max_temperature_K=integrator.temperature_K)``;
    - ``output``: ``fm.Output``; thermo rows, trajectory frames and ``metadata.json`` go to its
      directory, in chunks (§10.6). ``None`` writes no file;
    - ``seed``: the integrator's noise seed, recorded;
    - ``check``: one of ``CHECK_MODES``;
    - ``allow_naive_cueq``: accept cuEquivariance's naive kernels (§9.2), recorded.
    """

    def __init__(self, atoms: Any, force_field: Any, *, species: Any = None, integrator: Any,
                 workload: Workload | None = None, output: Output | None = None, seed: int,
                 check: Check = "skip", allow_naive_cueq: bool = False):
        check_mode(check)
        from fastmace import integrators as records
        from fastmace.errors import PlanRefused, Reason
        from fastmace.io.output import Output
        from fastmace.model.forcefield import ForceField
        from fastmace.planner import minimal
        from fastmace.planner.workload import Workload

        if not isinstance(integrator, (records.Langevin, records.LangevinBAOAB)):
            raise TypeError(f"integrator must be fm.integrators.Langevin or fm.integrators.LangevinBAOAB, got "
                            f"{type(integrator).__name__}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"seed must be an int, got {seed!r}")
        if output is not None and not isinstance(output, Output):
            raise TypeError(f"output must be an fm.Output or None, got {type(output).__name__}")
        if isinstance(force_field, ForceField):
            if species is not None:
                raise ValueError("species= with a ForceField: the ForceField owns the species superset; pass "
                                 "fm.ForceField(model, species=...) or the model itself (SPEC §7.6)")
            ff = force_field
        else:
            ff = ForceField(force_field, atoms, precision="float32", species=species)
        if workload is None:
            workload = Workload(max_temperature_K=float(integrator.temperature_K))  # owner decision D7
        elif not isinstance(workload, Workload):
            raise TypeError(f"workload must be an fm.Workload, got {type(workload).__name__}")
        if integrator.temperature_K > workload.max_temperature_K:
            raise PlanRefused.from_reasons([Reason(
                f"the integrator's temperature_K={integrator.temperature_K:g} K is above "
                f"Workload.max_temperature_K={workload.max_temperature_K:g} K, which sized the skin and the buckets "
                "(rule 11, SPEC §7.6)", code="target-above-max-temperature")])

        self.force_field = ff
        self.integrator = integrator
        self.workload = workload
        self.output = output
        self.seed = seed
        self.check = check
        self.allow_naive_cueq = bool(allow_naive_cueq)
        self._atoms = atoms.copy()
        self._config = integrator.to_config(seed=seed)
        self._npt = self._config.ensemble == "npt"
        self._schedule = None
        self._mc = None  # the attached move set; always None until phase 3
        self._status = "setup"
        self._queue = _Queue()
        self._writer = None
        self._started = False
        self._steps_run = 0
        self._loop_s = 0.0
        self._run_started = None
        self._window_clock = None
        self._restart_points = []
        self._host_writes = 0
        setup = {}

        planner = minimal.default_planner(ff)
        started = time.perf_counter()
        self._plan = planner.plan(ff, atoms, self._config, workload, allow_naive_cueq=self.allow_naive_cueq)
        setup["plan"] = time.perf_counter() - started

        gate = _selfcheck() if check != "skip" else None
        references = None
        if check != "skip" and gate is not None:
            started = time.perf_counter()
            references = gate.references(self._plan, ff, atoms, self._config, check=check, seed=seed)
            setup["gate"] = time.perf_counter() - started

        started = time.perf_counter()
        self._engine = planner.build(self._plan, ff, atoms)
        setup["build"] = time.perf_counter() - started
        state = self._engine.state
        state.force_field_identity = ff.identity
        state.plan_revision = self._plan.revision
        from fastmace.protocols import TargetFill

        self._fill = TargetFill(self._engine.integrator)

        if check == "skip":
            self._gate = dict(GATE_SKIPPED)
        elif gate is None:
            self._gate = self._gate_unavailable("the gate (plan task 1b-T16) is not part of this FastMACE")
        else:
            started = time.perf_counter()
            self._gate = self._gate_verdict(gate.production(self._engine, references, check=check))
            setup["gate"] = setup.get("gate", 0.0) + time.perf_counter() - started
        self._setup_s = setup
        self._report = self._new_report()
        if output is not None:
            self._open_output()
        self._write_metadata()

    # ------------------------------------------------------------------ gate

    def _gate_unavailable(self, reason):
        record = {"status": "unavailable", "reason": reason}
        if self.check == "gate":
            from fastmace.errors import GateUnavailable

            raise GateUnavailable(f"the requested gate is unavailable: {reason}",
                                  why="check='gate' runs only with a gate verdict (owner decision D1, SPEC §12.3)",
                                  next_step='pass check="allow_unavailable" to run with the gate recorded as '
                                            'unavailable, or check="skip"', gate_record=record)
        return record

    def _gate_verdict(self, record):
        from fastmace.errors import GateFailed

        mapping = record.to_metadata() if hasattr(record, "to_metadata") else dict(record)
        status = mapping.get("status")
        if status == "failed":
            raise GateFailed(f"the gate failed on this structure: {mapping.get('reason') or 'see the record'}",
                             why="production and the corrected references disagree beyond the ratio rule "
                                 "(SPEC §12.3); a failing gate never runs silently (§9.3 rule 14)",
                             gate_record=record)
        if status == "unavailable":
            return self._gate_unavailable(mapping.get("reason") or "no reference ran")
        return mapping

    # ------------------------------------------------------------------ public state

    @property
    def plan(self) -> Any:
        """The active ``Plan`` revision (§9.5)."""
        return self._plan

    @property
    def gate(self) -> Any:
        """The gate record (§12.3): status ``passed``, ``failed``, ``unavailable`` or ``skipped``."""
        return self._gate

    @property
    def report(self) -> Any:
        """The run report (``fastmace.report.Report``, §7.7)."""
        return self._report

    @property
    def schedule(self) -> Any:
        """The attached schedule position (``protocols.ScheduleState``), or ``None``."""
        return self._schedule

    def committed(self) -> Committed:
        """The committed step as read-only host arrays (flushes any step in flight)."""
        engine = self._engine
        values = engine.sync()
        cell = values["cell"] if "cell" in values else engine.state.cell.detach().cpu().numpy()
        stress = values.get("stress")
        return Committed(step=int(engine.state.step), positions=_frozen(values["positions"]),
                         momenta=_frozen(values["momenta"]), cell=_frozen(cell), energy=float(values["energy"][0]),
                         forces=_frozen(values["forces"]), stress=None if stress is None else _frozen(stress))

    def submit_frame(self, sink: Callable[[int, Mapping[str, Any]], Any]) -> None:
        """Queue the committed step's output values for ``sink(step, values)`` (host arrays: positions,
        momenta, forces, energy, kinetic; cell and stress for NPT). The copies run behind the GPU;
        ``sink`` is called after the next step has been launched, or by ``flush_output``."""
        self._queue.submit(self._engine.snapshot(), sink)

    def flush_output(self) -> None:
        """Write everything queued and flush the output files."""
        self._queue.drain()
        if self._writer is not None:
            self._writer.flush()

    def metadata(self) -> dict[str, Any]:
        """The ``metadata.json`` record of the run as it stands (``provenance.metadata_record``)."""
        return self._metadata_record()

    def close(self) -> None:
        """Write what is queued, close the output files and write the final ``metadata.json``."""
        self.flush_output()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._status != "non-finite":
            self._status = "complete"
        self._write_metadata()

    def __enter__(self) -> Simulation:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ run

    def run(self, steps: int | None = None, schedule: Any = None, mc: Any = ...) -> None:
        """Advance ``steps`` steps, or along ``schedule`` to its end (§7.2 "Schedules and step counts").

        ``schedule=None`` continues the attached schedule, or keeps the target without one.
        ``mc=...`` (the default) keeps the attached move set, ``mc=None`` detaches it, a move set
        attaches it (§7.6). Every check runs before the first step: a schedule above
        ``Workload.max_temperature_K`` raises ``PlanRefused``, ``steps`` beyond the schedule
        ``ValueError``, and in 1b any ``mc`` is checked against the workload, then refused ("MC:
        phase 3").
        """
        from fastmace.errors import PlanRefused, Reason
        from fastmace.protocols import Schedule, ScheduleState

        if steps is not None:
            steps = _steps(steps)
        if schedule is not None:
            if not isinstance(schedule, Schedule):
                raise TypeError(f"schedule must be an fm.protocols Schedule (ramp, hold, +), got "
                                f"{type(schedule).__name__}")
            position = ScheduleState(schedule, self._config.timestep_fs)
        else:
            position = self._schedule
        if position is not None:
            count = position.steps_to_run(steps)
            highest = position.max_target(count)
            if highest > self.workload.max_temperature_K:
                raise PlanRefused.from_reasons([Reason(
                    f"the schedule reaches {highest:g} K in the steps this run would take, above "
                    f"Workload.max_temperature_K={self.workload.max_temperature_K:g} K, which sized the skin and "
                    "the buckets (rule 11); plan a new Simulation with a higher max_temperature_K (SPEC §7.6)",
                    code="schedule-above-max-temperature")])
        elif steps is None:
            raise ValueError("run() without a schedule needs steps=: without a schedule the target stays and "
                             "nothing fixes the run's length (SPEC §7.2)")
        else:
            count = steps
        if mc is not ... and mc is not None:
            self._refuse_mc(mc, position, count)  # raises: nothing is attached before phase 3

        self._schedule = position
        self._engine.state.protocol = position
        self._advance(count)

    def _refuse_mc(self, mc, position, count):
        """Check an MC block against the workload (§7.6 "Simulation inputs"), then refuse it: MC is phase 3."""
        from fastmace.errors import PlanRefused, Reason
        from fastmace.planner.minimal import _symbols, species_numbers

        workload = self.workload
        reasons = []
        blocks = tuple(mc) if isinstance(mc, (list, tuple)) else (mc,)
        moves, named = [], set()
        for block in blocks:
            move = getattr(block, "move", None) or getattr(block, "kind", None) or type(block).__name__.lower()
            moves.append(str(move))
            for attribute in ("pairs", "species", "transmutable", "elements"):
                value = getattr(block, attribute, None)
                if value is None:
                    continue
                for item in value:
                    named.update(species_numbers(item if isinstance(item, (list, tuple, set, frozenset))
                                                 else [item]))
        undeclared = [move for move in moves if move not in workload.mc_moves]
        if undeclared:
            reasons.append(Reason(
                f"MC move {', '.join(undeclared)} is not in Workload.mc_moves={workload.mc_moves!r}; the plan and "
                "the gate never covered it (SPEC §7.6)", code="mc-move-not-declared"))
        if workload.min_mc_temperature_K is not None:
            lowest = (position.min_target(count) if position is not None
                      else float(self._engine.integrator.temperature_K))
            if lowest < workload.min_mc_temperature_K:
                reasons.append(Reason(
                    f"the MC block's target temperature reaches {lowest:g} K in the steps this run would take, below "
                    f"Workload.min_mc_temperature_K={workload.min_mc_temperature_K:g} K (SPEC §7.6, §12.3)",
                    code="mc-below-min-temperature"))
        outside = sorted(named - set(self._plan.superset))
        if outside:
            reasons.append(Reason(
                f"the MC block names {_symbols(outside)}, outside the ForceField's superset "
                f"({_symbols(self._plan.superset)}) (SPEC §7.6, §8.5)", rule=2, code="species-not-in-superset"))
        reasons.append(Reason("MC: phase 3; Simulation.run in 1b runs MD only (SPEC §16)", code="mc-phase-3"))
        raise PlanRefused.from_reasons(reasons)

    def _advance(self, count):
        """``count`` MD steps along the attached schedule (if any), with output and step timing."""
        import torch

        from fastmace.errors import NonFiniteError

        engine = self._engine
        position = self._schedule
        output = self.output
        thermo_every = output.thermo_every if output is not None else None
        frame_every = output.trajectory_every if output is not None else None
        restart_every = output.restart_every if output is not None else None
        window_first, window_last = self._report.window
        cuda = engine.device.type == "cuda"

        def synchronised_clock():
            engine.flush()
            if cuda:
                torch.cuda.synchronize(engine.device)
            return self._loop_s + (time.perf_counter() - self._run_started)

        if not self._started:
            self._started = True
            if output is not None:
                self._submit_output(0, row=True, frame=frame_every is not None)
        self._status = "running"
        self._run_started = time.perf_counter()
        resolved = position.resolved if position is not None else None
        try:
            for _ in range(count):
                if position is not None:
                    self._fill(resolved.target(position.step + 1))
                engine.step()
                if position is not None:
                    position.step += 1
                step = engine.state.step
                self._queue.drain()
                row = thermo_every is not None and step % thermo_every == 0
                frame = frame_every is not None and step % frame_every == 0
                if row or frame:
                    self._submit_output(step, row=row, frame=frame)
                if restart_every is not None and step % restart_every == 0:
                    self.save_restart(Path(output.directory) / RESTART_NAME)
                if step == window_first:
                    self._window_clock = synchronised_clock()
                elif step == window_last and self._window_clock is not None and self._report.measured is None:
                    elapsed = synchronised_clock() - self._window_clock
                    self._report.record_speed(elapsed / (window_last - window_first), window_first, window_last)
            self._queue.drain()
            if output is None:
                energy = engine.potential_energy()
                if not math.isfinite(energy):
                    raise NonFiniteError(f"non-finite potential energy ({energy}) at step {engine.state.step}",
                                         why="the run stops at a non-finite energy; it is never fixed (SPEC §9.6)")
        except NonFiniteError:
            self._status = "non-finite"
            self._stop_clock(synchronised_clock, count_steps=engine.state.step)
            if self._writer is not None:
                self._writer.flush()
            self._write_metadata()
            raise
        self._stop_clock(synchronised_clock, count_steps=engine.state.step)
        self._status = "paused"
        self._report.finish(self._loop_s / self._steps_run, steps=self._steps_run)
        if self._writer is not None:
            self._writer.flush()
        self._write_metadata()

    def _stop_clock(self, clock, count_steps):
        self._loop_s = clock()
        self._run_started = None
        self._steps_run = count_steps - self._report.start_step

    # ------------------------------------------------------------------ output

    def _open_output(self, resume_from=None):
        from fastmace.io.output import ChunkedOutput

        directory = Path(self.output.directory)
        directory.mkdir(parents=True, exist_ok=True)
        header = THERMO_HEADER + (NPT_COLUMNS if self._npt else ())
        self._writer = ChunkedOutput(directory, self._atoms.copy(), header, resume_from=resume_from)

    def _sample(self, step):
        position = self._schedule
        if position is not None:
            schedule_step = position.step - (self._engine.state.step - step)
            target, phase = position.resolved.target(schedule_step), position.resolved.phase(schedule_step)
        else:
            target, phase = float(self._engine.integrator.temperature_K), CONSTANT_PHASE
        return (step, step * self._config.timestep_fs / 1000, phase, target)

    def _submit_output(self, step, *, row, frame):
        sample = self._sample(step) if row else None
        self._queue.submit(self._engine.snapshot(), lambda step, values: self._write(step, values, sample, frame))

    def _write(self, step, values, sample, frame):
        from fastmace.errors import NonFiniteError

        if sample is not None:
            for name in ("energy", "kinetic"):
                value = float(values[name][0])
                if not math.isfinite(value):
                    raise NonFiniteError(f"non-finite {name} ({value}) at step {step}",
                                         why="the run stops at a non-finite energy; it is never fixed (SPEC §9.6)")
        self._writer.write(values, sample, frame)

    # ------------------------------------------------------------------ report and metadata

    def _new_report(self):
        import numpy as np

        from fastmace import report as reports
        from fastmace.planner.minimal import _symbols

        model = self.force_field.model
        atoms = self._atoms
        reads_charge = model.charge is not None or model.spin_multiplicity is not None
        heights = 1.0 / np.linalg.norm(atoms.cell.reciprocal(), axis=1)
        outputs = () if self.output is None else (str(Path(self.output.directory)),)
        return reports.Report(
            model=model.label or Path(str(model.path)).name, head=model.head, sha256=model.sha256,
            structure=atoms.get_chemical_formula(), n_atoms=len(atoms), pbc=tuple(bool(p) for p in atoms.pbc),
            species=_symbols(self._plan.superset).split(), plan=self._plan,
            gate=None if self._gate.get("status") == "skipped" else self._gate, setup_s=self._setup_s,
            outputs=outputs, charge_spin=(model.charge, model.spin_multiplicity) if reads_charge else None,
            cell_heights_A=tuple(float(h) for h in heights), cutoff_A=self.force_field.traits.r_max,
            warmup_steps=reports.DEFAULT_WARMUP_STEPS, window_steps=reports.DEFAULT_WINDOW_STEPS,
            start_step=int(self._engine.state.step))

    def _metadata_record(self):
        from fastmace import provenance

        state = self._engine.state
        timing = {"status": self._status, "steps": self._steps_run, "loop_s": self._loop_s}
        return provenance.metadata_record(
            checkpoint=self.force_field.model, identity=self.force_field.identity, plan=self._plan, seed=self.seed,
            generators={"integrator": state.rng.generator}, cursors={"integrator": int(state.rng.cursor)},
            schedule=self._schedule, gate=self._gate, engine_info=self._engine.info(),
            restart_points=self._restart_points, setup=self._setup_s, timing=timing, report=self._report,
            device=str(self._engine.device))

    def _write_metadata(self):
        if self.output is None:
            return
        from fastmace import provenance

        provenance.write_metadata(Path(self.output.directory) / "metadata.json", self._metadata_record())

    # ------------------------------------------------------------------ restart (§10.6)

    def save_restart(self, path: Any) -> None:
        """Write the complete §10.6 state at the current committed step, a canonical restart point.

        ``path`` is ``<directory>/restart.npz`` (or the directory): ``Simulation.resume(directory)``
        reads it there. The output queue is drained and the output chunks end here (§10.6), the
        engine is canonicalised and written by ``io.restart.write_engine_restart`` (T05's format),
        and ``simulation.json`` next to it holds what that format does not: the model source, the
        integrator record, workload, ``Output``, seed, ``allow_naive_cueq``, species, masses and PBC.
        """
        from fastmace.io import restart

        path = Path(path)
        if path.suffix != ".npz":
            path = path / RESTART_NAME
        if path.name != RESTART_NAME:
            raise ValueError(f"save_restart writes {RESTART_NAME!r} (Simulation.resume reads <directory>/"
                             f"{RESTART_NAME}), got {path.name!r}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._queue.drain()
        outputs = self._writer.restart_point() if self._writer is not None else None
        restart.write_engine_restart(path, self._engine, name=ENGINE_NAME, plan=self._plan, outputs=outputs)
        if outputs is not None:
            point = {"step": int(self._engine.state.step), **outputs}
            if not self._restart_points or self._restart_points[-1] != point:
                self._restart_points.append(point)
        _write_json(path.parent / SIMULATION_NAME, self._simulation_record(path.parent))
        self._write_metadata()

    def _simulation_record(self, directory):
        import dataclasses

        model = self.force_field.model
        atoms = self._atoms
        output = None
        if self.output is not None:
            output = dataclasses.asdict(self.output)
            if Path(self.output.directory).resolve() == Path(directory).resolve():
                output["directory"] = "."  # the restart's own directory, wherever it is moved
        workload = dataclasses.asdict(self.workload)
        workload["needs_per_atom"] = sorted(workload["needs_per_atom"])
        workload["mc_moves"] = list(workload["mc_moves"])
        integrator = dataclasses.asdict(self.integrator)
        if isinstance(integrator.get("external_stress_GPa"), tuple):
            integrator["external_stress_GPa"] = list(integrator["external_stress_GPa"])
        return {
            "simulation_record_version": SIMULATION_RECORD_VERSION,
            "model": {"label": model.label, "path": str(model.path), "head": model.head, "sha256": model.sha256,
                      "trust": model.trust, "charge": model.charge, "spin_multiplicity": model.spin_multiplicity},
            "force_field": {"precision": self.force_field.precision, "species": list(self.force_field.species),
                            "rewrites": "auto" if self.force_field.rewrites == "auto"
                            else dataclasses.asdict(self.force_field.rewrites)},
            "integrator": {"class": type(self.integrator).__name__, "fields": integrator},
            "workload": workload,
            "output": output,
            "seed": self.seed,
            "allow_naive_cueq": self.allow_naive_cueq,
            "structure": {"numbers": [int(z) for z in atoms.numbers],
                          "masses": [float(m) for m in atoms.get_masses()],
                          "pbc": [bool(p) for p in atoms.pbc]},
            "plan_revision": int(self._plan.revision),
            "restart_points": list(self._restart_points),
        }

    @classmethod
    def resume(cls, directory: Any, *, allow_plan_change: bool = False, check: Check = "skip") -> Simulation:
        """Continue from the restart in ``directory``. A different force-field identity raises
        ``RestartMismatch``; a different plan needs ``allow_plan_change=True`` and makes a new
        revision (§9.5, §10.6). ``check`` as in the constructor.

        The model is loaded again from ``simulation.json``'s source, checked against its SHA-256; the
        engine is built at the restart's positions and cell through the planner, then takes the
        restart's state, edge bucket and neighbour capacity (``io.restart.resume_engine``). Output
        chunks after the restart point are deleted and new ones opened; nothing is rewritten.
        """
        import json

        from fastmace.io import restart
        from fastmace.model.forcefield import ForceField

        check_mode(check)
        directory = Path(directory)
        file = directory / RESTART_NAME
        if not file.is_file():
            raise FileNotFoundError(f"no restart file {file}; Simulation.save_restart writes it")
        record_path = directory / SIMULATION_NAME
        if not record_path.is_file():
            raise FileNotFoundError(f"no {SIMULATION_NAME} next to the restart file {file}; Simulation.resume "
                                    "continues runs that Simulation.save_restart wrote")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("simulation_record_version") != SIMULATION_RECORD_VERSION:
            raise ValueError(f"{record_path}: record version {record.get('simulation_record_version')!r} is not "
                             f"{SIMULATION_RECORD_VERSION}")
        stored = restart.read_restart(file, engine=ENGINE_NAME)
        loaded = _load_model(record["model"])
        rewrites = record["force_field"]["rewrites"]
        if rewrites != "auto":
            from fastmace.config import RewriteConfig

            rewrites = RewriteConfig(**rewrites)
        ff = ForceField(loaded, precision=record["force_field"]["precision"], rewrites=rewrites,
                        species=record["force_field"]["species"])
        restart.check_identity(stored, ff.identity)
        output = record["output"]
        if output is not None:
            from fastmace.io.output import Output

            if output["directory"] == ".":
                output["directory"] = str(directory)
            output = Output(**output)
        sim = cls(_restart_atoms(record["structure"], stored), ff, integrator=_integrator(record["integrator"]),
                  workload=_workload(record["workload"]), output=None, seed=record["seed"], check=check,
                  allow_naive_cueq=record["allow_naive_cueq"])
        sim._resume(stored, record, output, allow_plan_change)
        return sim

    def _resume(self, stored, record, output, allow_plan_change):
        from fastmace.errors import RestartMismatch
        from fastmace.io import restart
        from fastmace.planner.plan import Plan
        from fastmace.protocols import ScheduleState

        engine = self._engine
        restart.resume_engine(stored, engine)
        step = int(engine.state.step)
        previous = Plan.from_restart(restart.plan_record(stored))
        changed = [name for name in _PLAN_FIELDS if getattr(previous, name) != getattr(self._plan, name)]
        if changed:
            if not allow_plan_change:
                raise RestartMismatch(
                    f"the plan differs from the restart's in {', '.join(changed)}",
                    why="a different plan (another GPU, software or rewrite set) is the same physics but no "
                        "bitwise continuation, so it is never taken silently (SPEC §10.6)",
                    next_step="pass allow_plan_change=True to resume on the new plan, recorded as a new revision")
            fresh = self._plan
            self._plan = previous.revise(
                f"resumed with a different plan ({', '.join(changed)}; allow_plan_change=True)", step,
                **{name: getattr(fresh, name) for name in (*_PLAN_FIELDS, *_PLAN_RECORDS)})
        else:
            self._plan = previous
        engine.state.plan_revision = self._plan.revision
        protocol = restart.protocol_record(stored)
        self._schedule = None if protocol is None else ScheduleState.from_restart(protocol)
        engine.state.protocol = self._schedule
        target = (restart.integrator_record(stored) or {}).get("temperature_K")
        if target is not None:
            engine.integrator.set_temperature(target)
        self._fill.reset()
        self._started = True
        self._restart_points = [dict(point) for point in record.get("restart_points", ())]
        self._report = self._new_report()
        if output is not None:
            self.output = output
            self._open_output(resume_from=restart.output_indices(stored))
        self._write_metadata()

    # ------------------------------------------------------------------ host writes (owner decision D4)

    def _host_values(self, values, name):
        import numpy as np

        array = np.asarray(values, dtype=np.float64)
        expected = (len(self._atoms), 3)
        if array.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
        return array

    def _host_write(self, write):
        import torch

        self._queue.drain()
        engine = self._engine
        engine.flush()
        with torch.no_grad():
            write(engine.state)
        engine.canonicalise(reevaluate=True)
        if not self._host_writes:
            self._report.add_warning(HOST_WRITE_WARNING)
        self._host_writes += 1

    def set_positions(self, positions: Any) -> None:
        """Host write of positions at a canonical point: quiesce, write, rebuild, re-evaluate (owner
        decision D4). ``positions`` in Å, shape ``(N, 3)``; the momenta stay."""
        import torch

        array = self._host_values(positions, "positions")

        def write(state):
            state.positions.copy_(torch.as_tensor(array, dtype=state.positions.dtype))

        self._host_write(write)

    def set_momenta(self, momenta: Any) -> None:
        """Host write of momenta at a canonical point (owner decision D4). ``momenta`` in ASE units,
        shape ``(N, 3)``; the engine integrates velocities, ``momenta / masses``."""
        import torch

        array = self._host_values(momenta, "momenta")

        def write(state):
            momenta = torch.as_tensor(array, dtype=state.velocities.dtype, device=state.velocities.device)
            state.velocities.copy_(momenta / state.masses)

        self._host_write(write)


# What a resume compares to decide whether the plan changed (§10.6: "another GPU or rewrite set"), and
# the rest of the record a revision carries over from the new plan.
_PLAN_FIELDS = ("tier", "rewrites", "engine", "precision", "neighbours", "polar", "integrator", "superset",
                "workload")
_PLAN_RECORDS = ("decisions", "requested", "auto_options_off", "software", "allowances")


def _write_json(path, record):
    """Canonical JSON (sorted keys), written atomically: temporary file, then rename."""
    import json
    import os

    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_model(source):
    """The pristine model of a ``simulation.json`` source: a label as it is, a path with its SHA-256."""
    from fastmace.model import loading

    options = dict(charge=source.get("charge"), spin_multiplicity=source.get("spin_multiplicity"))
    if source.get("label"):
        return loading.load_pristine(source["label"], **options)
    return loading.load_pristine(source["path"], source.get("head"), sha256=source["sha256"], **options)


def _restart_atoms(structure, stored):
    """The run's ``Atoms`` at the restart's positions and cell (species, masses and PBC from the record)."""
    from ase import Atoms

    cell = stored["cell_state"] if "cell_state" in stored else stored["cell"]
    atoms = Atoms(numbers=structure["numbers"], positions=stored["positions"], cell=cell, pbc=structure["pbc"])
    atoms.set_masses(structure["masses"])
    return atoms


def _integrator(record):
    from fastmace import integrators

    classes = {"Langevin": integrators.Langevin, "LangevinBAOAB": integrators.LangevinBAOAB}
    if record["class"] not in classes:
        raise ValueError(f"unknown integrator record {record['class']!r}; this FastMACE has {sorted(classes)}")
    fields = dict(record["fields"])
    if isinstance(fields.get("external_stress_GPa"), list):
        fields["external_stress_GPa"] = tuple(fields["external_stress_GPa"])
    return classes[record["class"]](**fields)


def _workload(record):
    from fastmace.planner.workload import Workload

    fields = dict(record)
    fields["needs_per_atom"] = frozenset(fields["needs_per_atom"])
    fields["mc_moves"] = tuple(fields["mc_moves"])
    return Workload(**fields)
