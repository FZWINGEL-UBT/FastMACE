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

    def _open_output(self):
        from fastmace.io.output import ChunkedOutput

        directory = Path(self.output.directory)
        directory.mkdir(parents=True, exist_ok=True)
        header = THERMO_HEADER + (NPT_COLUMNS if self._npt else ())
        self._writer = ChunkedOutput(directory, self._atoms.copy(), header)

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
            schedule=self._schedule, gate=self._gate, engine_info=self._engine.info(), setup=self._setup_s,
            timing=timing, report=self._report, device=str(self._engine.device))

    def _write_metadata(self):
        if self.output is None:
            return
        from fastmace import provenance

        provenance.write_metadata(Path(self.output.directory) / "metadata.json", self._metadata_record())

    # ------------------------------------------------------------------ restart (after 1b-T14)

    def save_restart(self, path: Any) -> None:
        """Write the complete §10.6 state at the current committed step (a canonical point)."""
        raise NotImplementedError("Simulation.save_restart comes with plan task 1b-T14's canonical restart points")

    @classmethod
    def resume(cls, directory: Any, *, allow_plan_change: bool = False, check: Check = "skip") -> Simulation:
        """Continue from the restart in ``directory``. A different force-field identity raises
        ``RestartMismatch``; a different plan needs ``allow_plan_change=True`` and makes a new
        revision (§9.5, §10.6). ``check`` as in the constructor."""
        check_mode(check)
        raise NotImplementedError("Simulation.resume comes with plan task 1b-T14's canonical restart points")

    def set_positions(self, positions: Any) -> None:
        """Host write of positions at a canonical point: quiesce, write, rebuild, re-evaluate (owner
        decision D4)."""
        raise NotImplementedError("Simulation.set_positions comes with plan task 1b-T14's canonicalise()")

    def set_momenta(self, momenta: Any) -> None:
        """Host write of momenta at a canonical point (owner decision D4)."""
        raise NotImplementedError("Simulation.set_momenta comes with plan task 1b-T14's canonicalise()")
