"""Simulation: one structure, one force field, one integrator, run in segments (SPEC §7.2, §7.6).

Plan task 1b-T01 fixed the signature; plan task 1b-T17 implements it. SPEC §5 lists no module for
``Simulation``; it lives here, above ``planner`` and ``engine``, and gets its engine from the
minimal planner (1b-T12), never by building one (SPEC §5).

What the constructor does, in this order (``docs/design/contracts-1b.md``):

1. with a model instead of a ``ForceField``, build ``ForceField(model, atoms, precision="float32",
   species=species)`` (§7.6 "Simulation inputs");
2. ``workload=None`` becomes ``Workload(max_temperature_K=integrator.temperature_K)`` (owner
   decision D7), recorded in the plan;
3. plan through the minimal planner, with the caller's planner inputs (``PlannerOptions``: mode,
   ``allow_reference``, fallback, gate, runner, engine and neighbour options; plan task 2-T01);
   ``PlanRefused`` here carries every reason. A plan that a process constraint forbids here
   (``Plan.runner == "subprocess"``, §8.7) stops at this point: the subprocess runner
   (``engine/subprocess.py``, plan task 2-T07) does the rest in a child, and each ``run()`` in
   another, handed over through restart writes;
4. with ``check="gate"`` or ``"allow_unavailable"``: the gate's references (§12.3, 1b-T16),
   before any production engine exists in the process;
5. build the engine through the planner;
6. the gate's production half through that engine (captured replay, registry snapshot and
   restore), then ``GateFailed`` or ``GateUnavailable`` if it says so; the record is ``sim.gate``.
   Without a requested gate its status is ``skipped`` (owner decision D1);
7. a failure of steps 5-6 (build, compile, capture, self-check or a failed gate, classified by the
   step that raised it; the first step's CUDA graphs are captured after the build, before the gate,
   ``GraphEngine.prepare``, so a capture failure comes here too) goes to the fallback (§9.4, plan task 2-T15) under
   ``FallbackConfig.fallback == "auto"`` (decision 29): the active plan becomes the validated
   revision it finds, or ``FallbackExhausted`` is raised with the record. Under ``"refuse"`` the
   failure raises as in 1b. A resume never falls back: it rebuilds the restart's revision.

So a constructed ``Simulation`` is ready to step, and every refusal comes before the first step.

The run loop is 1a's (``langevin/workflow.py:run_crystal`` as driven through
``langevin/fastmace_engine.py``): the schedule's target is filled before each step where it
changes (``protocols.TargetFill``), and output is asynchronous: a thermo or frame step queues a
snapshot that is written after the next step has been launched. ``Simulation`` owns the step
timing that the run report (1b-T15) shows: the steady-state window after warm-up, and the
whole-run mean at the end of each ``run``.

MC (plan task 3-T15, SPEC §7.6): ``run(mc=…)`` attaches a move set to the simulation through T09's
``fastmace.mc.driver.MCDriver``, and it stays attached until a later ``run`` passes another
``mc=`` or ``mc=None``. A block runs right after the MD step it is due at (``MCDriver.due``), before
that step's output and restart, with ``_mc_block_active`` set for the restart guard. The streams,
counters and cycle position carry across a detach and re-attach, and across a restart (attached:
the restart's ``mc`` record; detached: ``simulation.json``).
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fastmace.config import EngineConfig, FallbackConfig, GateConfig, Mode, NeighbourConfig, RunnerConfig

if TYPE_CHECKING:
    import numpy as np

    from fastmace.io.output import Output
    from fastmace.planner.workload import Workload

Check = Literal["skip", "gate", "allow_unavailable"]
# check="skip": the gate does not run, status "skipped" (the default, owner decision D1).
# check="gate": run the gate before the first step; failed -> GateFailed, unavailable -> GateUnavailable.
# check="allow_unavailable": as "gate", but an unavailable gate is recorded and the run goes on.
CHECK_MODES = ("skip", "gate", "allow_unavailable")

THERMO_HEADER = ("step", "time_ps", "phase", "target_K", "temperature_K", "potential_eV_atom", "kinetic_eV_atom",
                 "total_eV_atom")
NPT_COLUMNS = ("volume_A3_atom", "pressure_GPa")
# The MC thermo columns, per declared move type (Workload.mc_moves): cumulative, as of the last block.
MC_THERMO_COUNTERS = ("attempted", "accepted")
CONSTANT_PHASE = "constant"  # the thermo phase column without a schedule
ENGINE_NAME = "fastmace"  # the restart file's engine entry
RESTART_NAME = "restart.npz"  # io.restart.RESTART_NAME (that module imports torch)
SIMULATION_NAME = "simulation.json"  # next to the restart: what the restart format does not hold
SIMULATION_RECORD_VERSION = 2  # 2: the planner options (2-T01); version 1 records resume with the defaults
HOST_WRITE_WARNING = ("host writes of positions or momenta run at a canonical point (quiesce, write, rebuild, "
                      "re-evaluate; owner decision D4): correct, and slower than a step")


def check_mode(check: Any) -> str:
    """``check`` if it is one of ``CHECK_MODES``; else ``ValueError`` naming them."""
    if check not in CHECK_MODES:
        raise ValueError(f"check={check!r} is not one of {CHECK_MODES}; the gate is opt-in: check='gate' (SPEC §12.3)")
    return check


@dataclass(frozen=True)
class PlannerOptions:
    """The planner inputs of ``Simulation`` and ``fm.md`` (plan task 2-T01), validated, recorded in
    ``simulation.json`` and used again by ``Simulation.resume``.

    - ``mode``: the execution mode (``config.Mode``, §9.3), or ``None`` (the default): the planner picks the
      graph tier, or the compiled tier by rule 4 or rule 9 (3-T06); ``allow_reference``: rule 1's flag;
    - ``fallback``, ``gate``, ``runner``: the safety records of phase 2 (§9.4, §12.3, §8.7);
    - ``engine``, ``neighbours``: the planner-level options; ``None`` keeps the default planner's
      (``planner.minimal.default_planner``: CUDA if available, else the CPU without compilation).
    """

    mode: Mode | None = None
    allow_reference: bool = False
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    engine: EngineConfig | None = None
    neighbours: NeighbourConfig | None = None

    def to_record(self) -> dict[str, Any]:
        """A JSON-safe mapping (``simulation.json``'s ``planner_options``)."""
        import dataclasses

        from fastmace.planner.plan import _json_safe

        def part(value):
            return None if value is None else _json_safe(dataclasses.asdict(value))

        return {"mode": self.mode, "allow_reference": self.allow_reference, "fallback": part(self.fallback),
                "gate": part(self.gate), "runner": part(self.runner), "engine": part(self.engine),
                "neighbours": part(self.neighbours)}

    @classmethod
    def from_record(cls, record: Mapping[str, Any] | None) -> PlannerOptions:
        """The options of a ``to_record`` mapping; ``None`` (a version 1 record) gives the defaults."""
        if record is None:
            return cls()
        from fastmace.planner.plan import _from_json

        def part(name, kind):
            values = record.get(name)
            if values is None:
                return None
            return kind(**{key: _from_json(value) for key, value in values.items()})

        return planner_options(mode=record.get("mode"), allow_reference=record.get("allow_reference", False),
                               fallback=part("fallback", FallbackConfig), gate=part("gate", GateConfig),
                               runner=part("runner", RunnerConfig), engine=part("engine", EngineConfig),
                               neighbours=part("neighbours", NeighbourConfig))


def planner_options(mode: Any = None, allow_reference: Any = False, fallback: Any = None, gate: Any = None,
                    runner: Any = None, engine: Any = None, neighbours: Any = None) -> PlannerOptions:
    """The validated ``PlannerOptions``: ``ValueError`` or ``TypeError`` before anything is loaded."""
    from fastmace.config import MODES

    if mode is not None and (not isinstance(mode, str) or mode not in MODES):
        raise ValueError(f"mode={mode!r} is not None or one of {MODES} (SPEC §9.3)")
    if not isinstance(allow_reference, bool):
        raise TypeError(f"allow_reference must be a bool, got {allow_reference!r}")
    parts = {}
    for name, value, kind, default in (("fallback", fallback, FallbackConfig, FallbackConfig),
                                       ("gate", gate, GateConfig, GateConfig),
                                       ("runner", runner, RunnerConfig, RunnerConfig),
                                       ("engine", engine, EngineConfig, None),
                                       ("neighbours", neighbours, NeighbourConfig, None)):
        if value is None:
            parts[name] = None if default is None else default()
        elif isinstance(value, kind):
            parts[name] = value
        else:
            raise TypeError(f"{name} must be a fastmace.config.{kind.__name__} or None, got {type(value).__name__}")
    return PlannerOptions(mode=mode, allow_reference=allow_reference, **parts)


def mc_columns(workload: Any) -> tuple[str, ...]:
    """The thermo columns of a workload's MC moves: ``mc_<move>_attempted``, ``mc_<move>_accepted``."""
    return tuple(f"mc_{move}_{counter}" for move in workload.mc_moves for counter in MC_THERMO_COUNTERS)


def mc_moves(mc: Any) -> tuple:
    """The move records of an ``mc=`` argument (one ``fm.mc`` record or a sequence of them), validated."""
    from fastmace import mc as records

    blocks = tuple(mc) if isinstance(mc, (list, tuple)) else (mc,)
    if not blocks:
        raise ValueError("mc= is an empty move set; pass mc=None to detach (SPEC §7.6)")
    kinds = (records.Swap, records.Transmute, records.Displace)
    for block in blocks:
        if not isinstance(block, kinds):
            raise TypeError(f"mc= takes fm.mc records (Swap, Transmute, Displace), got {type(block).__name__}")
    return blocks


def move_record(move: Any) -> dict[str, Any]:
    """A move record as JSON-safe data (T09's ``fastmace.mc.driver.move_record``: ``{"move": ..., **fields}``)."""
    from fastmace.mc.driver import move_record as record

    return record(move)


def move_from_record(record: Mapping[str, Any]) -> Any:
    """The ``fm.mc`` record of ``move_record``'s data (a restart's ``mc.moves`` entry or a CLI ``[[mc]]`` table),
    through T09's ``fastmace.mc.driver.move_from_record``. ``ValueError`` names an unknown move type or field."""
    from fastmace import mc as records
    from fastmace.mc.driver import move_from_record as from_record

    classes = {cls.move: cls for cls in (records.Swap, records.Transmute, records.Displace)}
    move = record.get("move")
    if move not in classes:
        raise ValueError(f"an MC move needs move = one of {tuple(classes)}, got {move!r}")
    names = {item.name for item in dataclasses.fields(classes[move])}
    unknown = sorted(set(record) - names - {"move"})
    if unknown:
        raise ValueError(f"unknown field{'s' if len(unknown) > 1 else ''} {', '.join(unknown)} for move {move!r}; "
                         f"fm.mc.{classes[move].__name__} takes {', '.join(sorted(names))}")
    # A TOML table gives mu and formal_charges as mappings; T09's records take them as (species, value) pairs.
    return from_record({key: list(value.items()) if isinstance(value, Mapping) else value
                        for key, value in record.items()})


def _steps(steps: Any) -> int:
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError(f"steps must be an int, got {steps!r}")
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    return steps


@dataclass(frozen=True)
class Committed:
    """The committed step as read-only host arrays: what an observer sees (``Simulation.committed``).

    ``stress`` (3×3, eV/Å³) is ``None`` for fixed-cell integrators; ``energy`` in eV. The composition
    (3-T03): ``species`` atomic numbers ``(N,)``, ``masses`` amu ``(N,)``, ``charges`` ``(N,)`` or
    ``None`` for models that take none.
    """

    step: int
    positions: np.ndarray
    momenta: np.ndarray
    cell: np.ndarray
    energy: float
    forces: np.ndarray
    stress: np.ndarray | None
    species: np.ndarray
    masses: np.ndarray
    charges: np.ndarray | None


def _frozen(array):
    import numpy as np

    out = np.array(array, copy=True)
    out.setflags(write=False)
    return out


def composition_values(state) -> dict[str, Any]:
    """The state's species, masses ``(N,)`` and charges as host arrays (charges omitted when ``None``)."""
    out = {"species": state.species.detach().cpu().numpy(), "masses": state.masses[:, 0].detach().cpu().numpy()}
    if state.charges is not None:
        out["charges"] = state.charges.detach().cpu().numpy()
    return out


def committed_composition(values: Mapping[str, Any]) -> dict[str, Any]:
    """``Committed``'s composition fields from ``composition_values`` (or the runner's copy of them)."""
    charges = values.get("charges")
    return {"species": _frozen(values["species"]), "masses": _frozen(values["masses"]),
            "charges": None if charges is None else _frozen(charges)}


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
    check="skip", allow_naive_cueq=False, mode=None, allow_reference=False, fallback=None, gate=None,
    runner=None, engine=None, neighbours=None)``:

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
    - ``allow_naive_cueq``: accept cuEquivariance's naive kernels (§9.2), recorded;
    - the planner inputs (``PlannerOptions``, recorded in ``simulation.json``): ``mode`` (§9.3; ``None``
      lets the planner choose),
      ``allow_reference`` (rule 1), ``fallback`` (``FallbackConfig``, §9.4), ``gate``
      (``GateConfig``, §12.3 "Resources"; the record is ``sim.gate_config``, the result
      ``sim.gate``), ``runner`` (``RunnerConfig``, §8.7), ``engine`` (``EngineConfig``) and
      ``neighbours`` (``NeighbourConfig``). ``None`` takes the defaults, and for ``engine`` and
      ``neighbours`` the default planner's.
    """

    # True while an MC block runs (T15's block hook sets it): restart writes are refused then (§10.6).
    _mc_block_active: bool = False

    def __init__(self, atoms: Any, force_field: Any, *, species: Any = None, integrator: Any,
                 workload: Workload | None = None, output: Output | None = None, seed: int,
                 check: Check = "skip", allow_naive_cueq: bool = False, mode: Mode | None = None,
                 allow_reference: bool = False, fallback: FallbackConfig | None = None,
                 gate: GateConfig | None = None, runner: RunnerConfig | None = None,
                 engine: EngineConfig | None = None, neighbours: NeighbourConfig | None = None):
        check_mode(check)
        options = planner_options(mode=mode, allow_reference=allow_reference, fallback=fallback, gate=gate,
                                  runner=runner, engine=engine, neighbours=neighbours)
        from fastmace import integrators as records
        from fastmace.errors import PlanRefused, Reason
        from fastmace.io.output import Output
        from fastmace.model.forcefield import ForceField
        from fastmace.planner import fallback, minimal, selfcheck
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
        self.options = options
        self._atoms = atoms.copy()
        self._config = integrator.to_config(seed=seed)
        self._npt = self._config.ensemble == "npt"
        self._schedule = None
        self._mc = None  # the attached move set's MCDriver (3-T15), or None
        self._mc_moves = None  # its move records, also while a child runs the plan
        self._mc_record = None  # the last MC state (MCDriver.state_dict), kept across a detach
        self._mc_ranges = []  # [first, last] step ranges run with a move set attached
        self._mc_blocks = 0
        self._mc_device_ms = None  # device time of the timed blocks: each driver's first block holds its captures
        self._mc_timed_attempts = 0
        self._mc_block_s = 0.0  # wall time of this process's MC blocks: left out of the speed line
        self._mc_block_s_before = 0.0  # the blocks' time before a resume (simulation.json)
        self._mc_untimed = None  # the driver whose first block has not run yet
        self._frame_composition = None  # host species and masses for the frames, after a composition change
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
        self._sinks = []  # (interval, sink): add_sink
        self._runner = None  # the subprocess runner of a plan that runs in a child (2-T07)
        setup = {}

        planner = _planner(minimal.default_planner(ff), options)
        started = time.perf_counter()
        stored = getattr(self, "_resume_plan", None)  # a resume of a fallback revision (2-T15)
        self._plan = planner.plan(ff, atoms, self._config, workload, allow_naive_cueq=self.allow_naive_cueq,
                                  allow_reference=options.allow_reference,
                                  mode=options.mode if stored is None else stored.tier)
        if stored is not None:
            from fastmace.planner.fallback import replay_off

            self._plan = replay_off(stored, self._plan)
        setup["plan"] = time.perf_counter() - started
        if self._plan.runner == "subprocess":
            self._start_runner(setup)
            return

        # The gate's references before any production engine exists (§12.3); under check="gate" an
        # unavailable reference raises GateUnavailable here, before the build. "skip" computes nothing.
        started = time.perf_counter()
        references = selfcheck.references(self._plan, ff, atoms, integrator, check=check, seed=seed,
                                         gate=options.gate)
        gate_s = time.perf_counter() - started

        started = time.perf_counter()
        failure = None
        try:
            self._engine = planner.build(self._plan, ff, atoms)
        except Exception as error:
            failure = self._failure("build", error, stored)
        else:
            self._attach_engine()
        setup["build"] = time.perf_counter() - started
        # The first step's CUDA graphs, captured now, inside the fallback window, not at the first
        # run() (2-T15b). A resume keeps capturing at its first step: it never falls back.
        if failure is None and stored is None and not getattr(self, "_resuming", False):
            started = time.perf_counter()
            try:
                self._engine.prepare()
            except Exception as error:
                failure = self._failure("capture", error, stored)
            if self._engine is not None and self._engine.capture:
                setup["capture"] = time.perf_counter() - started

        # The production half through this engine; GateFailed (and the unavailable record under
        # check="allow_unavailable") come from selfcheck. The record is final: sim.gate.
        started = time.perf_counter()
        if failure is None:
            try:
                self._gate = selfcheck.production(self._engine, references, check=check)
            except Exception as error:
                failure = self._failure("gate", error, stored)
        if check != "skip":
            setup["gate"] = gate_s + time.perf_counter() - started
        if failure is not None:  # the fallback (§9.4; plan task 2-T15)
            self._engine = None
            started = time.perf_counter()
            result = fallback.Fallback(planner, ff, atoms, self._config, references, options.fallback, seed=seed,
                                       gate=options.gate).run(failure)
            setup["fallback"] = time.perf_counter() - started
            self._plan = result.plan
            if result.runner:  # the candidate runs in a child (§9.4 step 3, 2-T07)
                self.options = dataclasses.replace(options, mode=result.plan.tier, engine=dataclasses.replace(
                    options.engine, backend=result.plan.engine.backend))
                self._start_runner(setup)
                return
            self._engine = result.engine
            self._attach_engine()
            # The user's check="skip" stays "skipped"; the validation is in plan.fallbacks.
            self._gate = result.gate if check != "skip" else selfcheck.skipped_record()
        # The software block of metadata.json, built once per process here, in the setup time,
        # not at the first run() (provenance.fastmace_block, 1b-T28).
        from fastmace import provenance

        started = time.perf_counter()
        provenance.fastmace_block(str(self._engine.device))
        setup["provenance"] = time.perf_counter() - started
        self._setup_s = setup
        self._report = self._new_report()
        if output is not None:
            self._open_output()
        self._write_metadata()

    def _attach_engine(self):
        """The engine's state carries the force field's identity and the plan revision; the target fill."""
        from fastmace.protocols import TargetFill

        state = self._engine.state
        state.force_field_identity = self.force_field.identity
        state.plan_revision = self._plan.revision
        self._fill = TargetFill(self._engine.integrator)

    def _failure(self, step, error, stored):
        """``error`` of the build or the gate as a ``fallback.Failure``, or raised again.

        Classified by the step that raised it (conductor answers, T15 Q7). Refusals, process
        constraints, out-of-memory and non-finite errors are not fallback triggers; neither is a gate
        that did not fail (``GateUnavailable``), nor any failure while resuming (a resume rebuilds the
        restart's revision and never falls back).
        """
        from fastmace.errors import CheckpointTrustError, GateFailed, ModelElementError, PlanRefused
        from fastmace.planner import fallback

        if stored is not None or getattr(self, "_resuming", False):
            raise error
        if fallback._propagates(error) or isinstance(error, (PlanRefused, ModelElementError, CheckpointTrustError)):
            raise error
        # A capture error is a capture failure wherever it is raised before the first step: in
        # prepare() or in the gate's replay (2-T15b).
        stage = fallback.stage_of(error, "capture" if step == "capture" else "build")
        if step == "gate" and stage != "capture":
            if not isinstance(error, GateFailed):
                raise error
            self._engine = self._fill = None
            return fallback.Failure(plan=self._plan, stage="gate", error=error, gate=error.gate_record)
        self._engine = self._fill = None  # nothing here keeps the failed engine alive (2-T15c)
        return fallback.Failure(plan=self._plan, stage=stage, error=error)

    def _start_runner(self, setup):
        """The plan runs in a child (``Plan.runner == "subprocess"``, SPEC §8.7; 2-T07): no gate and no
        engine in this process. The setup child plans, gates and builds there, and its restart at step 0
        is the first hand-off; ``Simulation.resume`` hands over its restart instead."""
        from fastmace.engine.subprocess import RUNNER_WARNING, SubprocessRunner
        from fastmace.planner import selfcheck

        self._engine = None
        self._fill = None
        self._gate = selfcheck.skipped_record()
        self._setup_s = setup
        self._runner = SubprocessRunner(self.options.runner)
        self._report = self._new_report(start_step=0)
        self._report.add_warning(RUNNER_WARNING)
        if not getattr(self, "_resuming", False):
            self._runner.setup(self)

    # ------------------------------------------------------------------ public state

    @property
    def plan(self) -> Any:
        """The active ``Plan`` revision (§9.5)."""
        return self._plan

    @property
    def fallback(self) -> FallbackConfig:
        """The ``FallbackConfig`` this simulation was made with (§9.4)."""
        return self.options.fallback

    @property
    def gate_config(self) -> GateConfig:
        """The ``GateConfig`` this simulation was made with (§12.3 "Resources"); ``gate`` is the result."""
        return self.options.gate

    @property
    def runner(self) -> RunnerConfig:
        """The ``RunnerConfig`` this simulation was made with (§8.7)."""
        return self.options.runner

    @property
    def gate(self) -> Any:
        """The gate record (``planner.selfcheck.GateRecord``, §12.3): status ``passed``, ``unavailable`` (with
        ``check="allow_unavailable"``) or ``skipped``; a failed gate raised ``GateFailed`` at construction."""
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
        if self._runner is not None:
            return self._runner.committed(self)
        engine = self._engine
        values = engine.sync()
        cell = values["cell"] if "cell" in values else engine.state.cell.detach().cpu().numpy()
        stress = values.get("stress")
        composition = composition_values(engine.state)
        return Committed(step=int(engine.state.step), positions=_frozen(values["positions"]),
                         momenta=_frozen(values["momenta"]), cell=_frozen(cell), energy=float(values["energy"][0]),
                         forces=_frozen(values["forces"]), stress=None if stress is None else _frozen(stress),
                         **committed_composition(composition))

    def submit_frame(self, sink: Callable[[int, Mapping[str, Any]], Any]) -> None:
        """Queue the committed step's output values for ``sink(step, values)`` (host arrays: positions,
        momenta, forces, energy, kinetic; cell and stress for NPT). The copies run behind the GPU;
        ``sink`` is called after the next step has been launched, or by ``flush_output``."""
        if self._runner is not None:
            self._runner.submit_frame(self, sink)
            return
        self._queue.submit(self._snapshot(), sink)

    def add_sink(self, sink: Callable[[int, Mapping[str, Any]], Any], interval: int) -> None:
        """Call ``sink(step, values)`` for every step of ``run`` that is a multiple of ``interval``, with
        the values of ``submit_frame``, through the same asynchronous queue as the chunked output (for
        ``fastmace.ase``'s trajectory and log files). ``interval`` is independent of the ``Output``
        cadences; the queue is drained, oldest first, at every restart point and at the end of each
        ``run``, so a sink has seen every due step up to there. Step 0 is the caller's
        (``submit_frame``)."""
        if not callable(sink):
            raise TypeError(f"sink must be callable, got {type(sink).__name__}")
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError(f"interval must be a positive int (steps), got {interval!r}")
        self._sinks.append((interval, sink))

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
        if self._runner is not None:
            self._runner.close()  # the hand-off directory, unless a child failed

    def __enter__(self) -> Simulation:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ run

    def run(self, steps: int | None = None, schedule: Any = None, mc: Any = ...) -> None:
        """Advance ``steps`` steps, or along ``schedule`` to its end (§7.2 "Schedules and step counts").

        ``schedule=None`` continues the attached schedule, or keeps the target without one.
        ``mc=...`` (the default) keeps the attached move set, ``mc=None`` detaches it, a move set
        attaches it in place of any attached one (§7.6): one ``fm.mc`` record (``Swap``, ``Transmute``,
        ``Displace``) or a sequence of them. Every check runs before the first step: a schedule above
        ``Workload.max_temperature_K`` raises ``PlanRefused``, ``steps`` beyond the schedule
        ``ValueError``, and the move set that will be attached (given or kept) is checked against the
        workload over every step of this call (``PlanRefused``; ``SpeciesNotInSuperset`` for a species
        outside the superset).
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
        moves = None if mc is ... or mc is None else mc_moves(mc)
        attached = moves if moves is not None else None if mc is None else self._mc_moves
        if attached:
            self._check_mc(attached, position, count)

        if self._runner is not None:  # the plan runs in a child (SPEC §8.7), which attaches the move set
            self._runner.run(self, steps=count, schedule=position if schedule is not None else None, mc=mc)
            if mc is not ...:
                self._mc_moves = moves
            return
        if mc is None:
            self._detach_mc()
        elif moves is not None:
            self._attach_mc(moves)
        self._schedule = position
        self._engine.state.protocol = position
        self._advance(count)

    # ------------------------------------------------------------------ MC (§7.6, §10.4; 3-T15)

    def _check_mc(self, blocks, position, count):
        """Check a move set against the workload over the ``count`` steps of this call (§7.6 "Simulation
        inputs"): every reason at once in one ``PlanRefused``. T09's driver adds its own checks at attach."""
        from fastmace.errors import PlanRefused, Reason
        from fastmace.planner.minimal import _symbols, species_numbers

        workload = self.workload
        reasons = []
        moves, named = [], set()
        for block in blocks:
            moves.append(block.move)
            for attribute in ("pairs", "species"):
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
                      else float(self._engine.integrator.temperature_K) if self._engine is not None
                      else float(self.integrator.temperature_K))
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
        if reasons:
            raise PlanRefused.from_reasons(reasons)

    def _attach_mc(self, moves):
        """Attach ``moves`` through a new ``MCDriver``; the streams, counters and cycle position of the move set
        attached before (or detached since) carry over, so a re-attach resumes them (§7.6)."""
        from fastmace.mc import driver as mc_driver

        previous = self._mc.state_dict() if self._mc is not None else self._mc_record
        if self._mc is not None:
            self._mc.detach()  # its captured attempts go with it (3-T09 follow-up)
        driver = mc_driver.MCDriver(moves, seed=self.seed)
        driver.attach(self._engine, max_displacement_A=self.workload.max_displacement_A)
        if previous is not None:
            driver.load_state_dict(_carried_mc_record(previous, driver.state_dict()), other_moves=True)
        self._mc, self._mc_moves, self._mc_untimed = driver, tuple(moves), driver
        self._mc_record = driver.state_dict()
        self._engine.state.mc = self._mc_record
        self._report.mc = self._mc_summary()

    def _detach_mc(self):
        """Detach the attached move set (``run(mc=None)``); its state stays for a later re-attach."""
        if self._mc is not None:
            self._mc_record = self._mc.state_dict()
            self._mc.detach()  # its captured attempts go with it (3-T09 follow-up)
        self._mc = None
        self._mc_moves = None
        if self._engine is not None:
            self._engine.state.mc = None

    def _resume_mc(self, record):
        """Re-attach the move set of a restart's ``mc`` record (§7.6: the attached move set is restart state)."""
        from fastmace.mc import driver as mc_driver

        moves = tuple(move_from_record(entry) for entry in record["moves"])
        driver = mc_driver.MCDriver(moves, seed=self.seed)
        driver.attach(self._engine, max_displacement_A=self.workload.max_displacement_A)
        driver.load_state_dict(record)
        self._mc, self._mc_moves, self._mc_untimed = driver, moves, driver
        self._mc_record = driver.state_dict()
        self._engine.state.mc = self._mc_record

    def _synchronised_time(self):
        """``time.perf_counter()`` once the engine's queued work is done (flush, then a CUDA synchronisation)."""
        import torch

        engine = self._engine
        engine.flush()
        if engine.device.type == "cuda":
            torch.cuda.synchronize(engine.device)
        return time.perf_counter()

    def _run_block(self, driver, step):
        """One MC block at ``step``, with the restart guard set while it runs (T08's ``_mc_block_active``). Its
        wall time, between synchronisations (the block quiesces and reads its counters anyway), is left out of
        the speed line."""
        self._mc_block_active = True
        started = self._synchronised_time()
        try:
            block = driver.run_block(self._engine, step)
        finally:
            self._mc_block_active = False
            self._mc_block_s += self._synchronised_time() - started
        if block is None:
            return
        self._mc_blocks += 1
        if self._mc_untimed is driver:  # its first block's time includes the attempts' captures (T09)
            self._mc_untimed = None
        elif block.get("device_ms") is not None:
            self._mc_device_ms = (self._mc_device_ms or 0.0) + float(block["device_ms"])
            self._mc_timed_attempts += sum(int(entry["attempted"]) for entry in block["moves"])
        if block.get("accepted"):
            self._composition_changed()  # at the block's canonical point: one read of the new composition

    def _mc_counters(self):
        """``counters.mc``: the cumulative counters per move type, or ``None`` if no move set was ever attached."""
        if self._mc is not None:
            counters = self._mc.counters
        elif self._mc_record is not None:
            counters = self._mc_record["counters"]
        else:
            return None
        return {move: dict(counts) for move, counts in counters.items()}

    def _mc_tail(self):
        """The MC thermo columns' values (``mc_columns``), as of the last completed block: no device read."""
        counters = self._mc_counters() or {}
        return tuple(int(counters.get(move, {}).get(counter, 0)) for move in self.workload.mc_moves
                     for counter in MC_THERMO_COUNTERS)

    def _mc_summary(self):
        """The report's MC record (``fastmace.report.Report.mc``), or ``None`` without MC."""
        counters = self._mc_counters()
        if counters is None and not self._mc_moves:
            return None
        moves = self._mc_moves
        if moves is None and self._mc_record is not None:
            moves = tuple(move_from_record(entry) for entry in self._mc_record["moves"])
        return {"moves": tuple(moves or ()), "ranges": [list(pair) for pair in self._mc_ranges],
                "blocks": self._mc_blocks, "device_ms": self._mc_device_ms, "timed_attempts": self._mc_timed_attempts,
                "counters": counters or {}}

    def _mc_simulation_record(self):
        """``simulation.json``'s MC entry: what a resume needs beyond the restart's ``mc`` record."""
        if self._mc_record is None and not self._mc_ranges:
            return None
        return {"attached": self._mc is not None,
                "record": self._mc.state_dict() if self._mc is not None else self._mc_record,
                "ranges": [list(pair) for pair in self._mc_ranges], "blocks": self._mc_blocks,
                "device_ms": self._mc_device_ms, "timed_attempts": self._mc_timed_attempts,
                "block_s": self._mc_block_s_before + self._mc_block_s}

    def _composition_changed(self):
        """Read the committed species and masses once, for every later frame (``io.trajectory``)."""
        import torch

        values = composition_values(self._engine.state)
        self._frame_composition = {"numbers": torch.from_numpy(values["species"].copy()),
                                   "masses": torch.from_numpy(values["masses"].copy())}  # host tensors, as a snapshot

    def _snapshot(self):
        """The engine's output snapshot, with the current species and masses after a composition change."""
        snapshot = self._engine.snapshot()
        if self._frame_composition is not None:
            snapshot.values = {**snapshot.values, **self._frame_composition}
        return snapshot

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
        driver = self._mc
        if driver is not None:  # the steps of this call, run with the move set attached
            first = int(engine.state.step)
            if self._mc_ranges and self._mc_ranges[-1][1] == first:
                self._mc_ranges[-1][1] = first + count
            else:
                self._mc_ranges.append([first, first + count])
        try:
            for _ in range(count):
                if position is not None:
                    self._fill(resolved.target(position.step + 1))
                engine.step()
                if position is not None:
                    position.step += 1
                step = engine.state.step
                self._queue.drain()
                if driver is not None and driver.due(step):  # after the step's MD, before its output and restart
                    self._run_block(driver, step)
                row = thermo_every is not None and step % thermo_every == 0
                frame = frame_every is not None and step % frame_every == 0
                if row or frame:
                    self._submit_output(step, row=row, frame=frame)
                if self._sinks:
                    for interval, sink in self._sinks:
                        if step % interval == 0:
                            self._queue.submit(self._snapshot(), sink)
                if restart_every is not None and step % restart_every == 0:
                    self.save_restart(Path(output.directory) / RESTART_NAME)
                if step == window_first:
                    self._window_clock = synchronised_clock() - self._mc_block_s  # MD only: blocks left out
                elif step == window_last and self._window_clock is not None and self._report.measured is None:
                    elapsed = synchronised_clock() - self._mc_block_s - self._window_clock
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
        if self._mc_record is not None or self._mc is not None:
            if self._mc is not None:
                self._mc_record = self._mc.state_dict()
            self._report.mc = self._mc_summary()
        recapture = getattr(engine, "recapture_warning", None)  # past the re-capture budget (3-T04)
        if recapture and recapture not in self._report.extra_warnings:
            self._report.add_warning(recapture)
        self._report.finish(max(self._loop_s - self._mc_block_s, 1e-12) / self._steps_run, steps=self._steps_run)
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
        header = THERMO_HEADER + (NPT_COLUMNS if self._npt else ()) + mc_columns(self.workload)
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
        tail = self._mc_tail() if row and self.workload.mc_moves else ()
        self._queue.submit(self._snapshot(), lambda step, values: self._write(step, values, sample, frame, tail))

    def _write(self, step, values, sample, frame, tail=()):
        from fastmace.errors import NonFiniteError

        if sample is not None:
            for name in ("energy", "kinetic"):
                value = float(values[name][0])
                if not math.isfinite(value):
                    raise NonFiniteError(f"non-finite {name} ({value}) at step {step}",
                                         why="the run stops at a non-finite energy; it is never fixed (SPEC §9.6)")
        self._writer.write(values, sample, frame, tail=tail)

    # ------------------------------------------------------------------ report and metadata

    def _new_report(self, start_step=None):
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
            gate=None if self._gate.status == "skipped" else self._gate, setup_s=self._setup_s,
            outputs=outputs, charge_spin=(model.charge, model.spin_multiplicity) if reads_charge else None,
            cell_heights_A=tuple(float(h) for h in heights), cutoff_A=self.force_field.traits.r_max,
            warmup_steps=reports.DEFAULT_WARMUP_STEPS, window_steps=reports.DEFAULT_WINDOW_STEPS,
            start_step=int(self._engine.state.step) if start_step is None else start_step)

    def _metadata_record(self):
        from fastmace import provenance

        if self._runner is not None:
            return self._runner.metadata(self)
        state = self._engine.state
        timing = {"status": self._status, "steps": self._steps_run, "loop_s": self._loop_s}
        plan, built = self._plan, getattr(self._engine, "plan", None)
        if built is not None and built.revision == plan.revision:
            plan = built  # the built plan carries the per-record self-checks (2-T04)
        return provenance.metadata_record(
            checkpoint=self.force_field.model, identity=self.force_field.identity, plan=plan, seed=self.seed,
            generators={"integrator": state.rng.generator}, cursors={"integrator": int(state.rng.cursor)},
            schedule=self._schedule, gate=self._gate, counters={"mc": self._mc_counters()},
            engine_info=self._engine.info(),
            restart_points=self._restart_points, setup=self._setup_s, timing=timing, report=self._report,
            device=str(self._engine.device), force_field=self._engine.force_field)

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
        integrator record, workload, ``Output``, seed, ``allow_naive_cueq``, the planner options,
        species, masses and PBC. A plan that runs in a child copies the child's last hand-off, a
        canonical restart point at the committed step (2-T07).

        Only at a committed step: refused (``RuntimeError``) inside an MC block and while the engine
        holds an uncommitted (overflowed) step, before anything is written (§10.6, 3-T08).
        """
        self._check_restart_point()
        path = Path(path)
        if path.suffix != ".npz":
            path = path / RESTART_NAME
        if path.name != RESTART_NAME:
            raise ValueError(f"save_restart writes {RESTART_NAME!r} (Simulation.resume reads <directory>/"
                             f"{RESTART_NAME}), got {path.name!r}")
        if self._runner is not None:
            self._runner.save_restart(self, path)
            return
        from fastmace.io import restart

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

    def _check_restart_point(self):
        """Restart writes happen only at committed steps: never inside an MC block, never from an overflowed
        step (SPEC §10.6, ``GraphEngine.uncommitted``, 3-T04). Without an engine in this process (a plan run
        in a child) there is nothing to check here."""
        if self._mc_block_active:
            raise RuntimeError("a restart write inside an MC block: restart files are written only at committed "
                               "steps, between MC blocks (SPEC §10.6)")
        if self._engine is not None and self._engine.uncommitted():
            raise RuntimeError("a restart write while the engine holds an uncommitted (overflowed) step: restart "
                               "files are written only at committed steps (SPEC §10.6)")

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
            "planner_options": self.options.to_record(),
            "structure": {"numbers": [int(z) for z in atoms.numbers],
                          "masses": [float(m) for m in atoms.get_masses()],
                          "pbc": [bool(p) for p in atoms.pbc]},
            "plan_revision": int(self._plan.revision),
            "restart_points": list(self._restart_points),
            "mc": self._mc_simulation_record(),
        }

    @classmethod
    def resume(cls, directory: Any, *, allow_plan_change: bool = False, check: Check = "skip") -> Simulation:
        """Continue from the restart in ``directory``. A different force-field identity raises
        ``RestartMismatch``, except that another MACE or FastMACE version alone warns and is recorded
        as a plan revision and in the report (owner decision 23); a different plan needs
        ``allow_plan_change=True`` and makes a new revision (§9.5, §10.6). ``check`` as in the
        constructor.

        The model is loaded again from ``simulation.json``'s file, checked against its SHA-256; the
        engine is built at the restart's positions, cell and composition (species and masses at the
        restart step, not the initial structure's; 3-T08) through the planner, then takes the
        restart's state, edge bucket and neighbour capacity (``io.restart.resume_engine``). Output
        chunks after the restart point are deleted and new ones opened; nothing is rewritten.
        """
        import json

        from fastmace.io import restart

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
        if record.get("simulation_record_version") not in (1, SIMULATION_RECORD_VERSION):
            raise ValueError(f"{record_path}: record version {record.get('simulation_record_version')!r} is not "
                             f"1 or {SIMULATION_RECORD_VERSION}")
        options = PlannerOptions.from_record(record.get("planner_options"))
        stored = restart.read_restart(file, engine=ENGINE_NAME)
        ff = _record_force_field(record)
        versions = _resume_identity(stored, ff.identity)
        if versions:
            import warnings

            warnings.warn(_versions_text(versions), UserWarning, stacklevel=2)
        output = record["output"]
        if output is not None:
            from fastmace.io.output import Output

            if output["directory"] == ".":
                output["directory"] = str(directory)
            output = Output(**output)
        sim = cls.__new__(cls)
        sim._resuming = True  # a plan that runs in a child takes this restart as its first hand-off (2-T07)
        from fastmace.planner import fallback

        previous = fallback.stored_fallback_plan(restart.plan_record(stored))
        if previous is not None:  # a fallback revision is rebuilt as it is, never searched again (2-T15)
            sim._resume_plan = previous
        sim.__init__(_restart_atoms(record["structure"], stored), ff, integrator=_integrator(record["integrator"]),
                     workload=_workload(record["workload"]), output=None, seed=record["seed"], check=check,
                     allow_naive_cueq=record["allow_naive_cueq"], mode=options.mode,
                     allow_reference=options.allow_reference, fallback=options.fallback, gate=options.gate,
                     runner=options.runner, engine=options.engine, neighbours=options.neighbours)
        del sim._resuming
        sim.__dict__.pop("_resume_plan", None)
        sim._resume(stored, record, output, allow_plan_change, versions, source=directory)
        return sim

    def _resume(self, stored, record, output, allow_plan_change, versions=None, source=None):
        from fastmace.io import restart
        from fastmace.protocols import ScheduleState

        engine = self._engine
        if engine is None:  # the plan runs in a child: it resumes this restart (2-T07)
            self._plan = self._resumed_plan(stored, int(stored["step"]), allow_plan_change, versions)
            self._runner.resume(self, source, record, output)
            if versions:
                self._report.add_warning(_versions_text(versions))
            return
        restart.resume_engine(stored, engine)
        step = int(engine.state.step)
        self._plan = self._resumed_plan(stored, step, allow_plan_change, versions)
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
        mc = record.get("mc")  # 3-T15: the MC history, and the state of a detached move set
        if mc is not None:
            self._mc_ranges = [list(pair) for pair in mc["ranges"]]
            self._mc_blocks, self._mc_device_ms = int(mc["blocks"]), mc["device_ms"]
            self._mc_timed_attempts = int(mc.get("timed_attempts") or 0)
            self._mc_block_s_before = float(mc.get("block_s") or 0.0)
            self._mc_record = mc["record"]
        if engine.state.mc is not None:  # the restart's attached move set stays attached (§7.6)
            self._resume_mc(engine.state.mc)
        if output is not None:
            self.output = output  # before the report, whose output line names it
        self._report = self._new_report()
        self._report.mc = self._mc_summary()
        if versions:
            self._report.add_warning(_versions_text(versions))
        if output is not None:
            self._open_output(resume_from=restart.output_indices(stored))
        self._write_metadata()

    def _resumed_plan(self, stored, step, allow_plan_change, versions):
        """The plan a resume continues on: the restart's, or a revision to this one (§9.5, §10.6)."""
        from fastmace.errors import RestartMismatch
        from fastmace.io import restart
        from fastmace.planner.plan import Plan

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
            plan = previous.revise(
                f"resumed with a different plan ({', '.join(changed)}; allow_plan_change=True)", step,
                **{name: getattr(fresh, name) for name in (*_PLAN_FIELDS, *_PLAN_RECORDS)})
        else:
            plan = previous
        if versions:  # recorded as a revision of the same plan (owner decision 23)
            plan = plan.revise(_versions_text(versions), step)
        return plan

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

    def set_positions(self, positions: Any) -> None:
        """Host write of positions at a canonical point: quiesce, write, rebuild, re-evaluate (owner
        decision D4). ``positions`` in Å, shape ``(N, 3)``; the momenta stay."""
        array = self._host_values(positions, "positions")
        if self._runner is not None:  # applied by the next child, after its resume
            self._runner.host_write(self, "positions", array)
            return
        self.update(positions=array)

    def set_momenta(self, momenta: Any) -> None:
        """Host write of momenta at a canonical point (owner decision D4). ``momenta`` in ASE units,
        shape ``(N, 3)``; the engine integrates velocities, ``momenta / masses``."""
        array = self._host_values(momenta, "momenta")
        if self._runner is not None:
            self._runner.host_write(self, "momenta", array)
            return
        self.update(momenta=array)

    # Host mutations of composition and coefficients (SPEC §8.5, §7.6): one event each under the transaction
    # contract of fastmace.events, at a committed step; 3-T02's force field checks the superset.

    def update(self, *, species: Any = None, masses: Any = None, charges: Any = None, total_charge: Any = None,
               friction_per_fs: Any = None, positions: Any = None, momenta: Any = None) -> None:
        """Any combination of host changes as one transaction with one force evaluation (SPEC §8.5, §7.7).

        - ``species``: atomic numbers or symbols, ``(N,)``; in the ``ForceField``'s superset
          (``SpeciesNotInSuperset``). Masses follow the new species' standard masses unless ``masses`` is
          given too;
        - ``masses``: amu, ``(N,)``; the momenta stay (ASE), so the velocities change;
        - ``charges``: per-atom charges, ``(N,)``, and ``total_charge``, for models that take them;
        - ``friction_per_fs``: the Langevin friction in 1/fs (the record's unit); it drops the captured
          graphs, which bake it in (a recorded re-capture), and with the default barostat τ moves ``P_tau``
          and the barostat mass with it;
        - ``positions`` (Å) and ``momenta`` (ASE units), ``(N, 3)``; momenta use the new masses.

        Everything is validated before anything is written. The changes apply in this order: species,
        masses, charges, friction, positions, momenta; then a canonical point (neighbour list rebuilt at the
        committed positions, forces and stress evaluated once, ``integrator.start``), as a host write of
        positions (owner decision D4). Nothing given: nothing happens.
        """
        import numpy as np
        from ase import units

        from fastmace.events import Event, Transaction, species_numbers

        natoms = len(self._atoms)
        fields = {}
        if species is not None:
            fields["species"] = species_numbers(species, natoms)
        if masses is not None:
            fields["masses"] = Transaction._array(masses, "masses", natoms)
            if (fields["masses"] <= 0).any():
                raise ValueError("masses must be positive (amu)")
        if charges is not None:
            fields["charges"] = Transaction._array(charges, "charges", natoms)
        if total_charge is not None:
            if isinstance(total_charge, bool) or not np.isfinite(float(total_charge)):
                raise ValueError(f"the total charge must be a finite number, got {total_charge!r}")
            fields["total_charge"] = float(total_charge)
        if friction_per_fs is not None:
            if isinstance(friction_per_fs, bool) or not isinstance(friction_per_fs, (int, float, np.integer,
                                                                                      np.floating)):
                raise TypeError(f"friction_per_fs must be a number (1/fs), got {friction_per_fs!r}")
            if not (float(friction_per_fs) > 0 and np.isfinite(float(friction_per_fs))):
                raise ValueError(f"friction_per_fs must be positive and finite, got {friction_per_fs!r}")
            fields["friction_per_fs"] = float(friction_per_fs)
        if positions is not None:
            fields["positions"] = self._host_values(positions, "positions")
        if momenta is not None:
            fields["momenta"] = self._host_values(momenta, "momenta")
        if not fields:
            return
        writes = {"positions", "momenta"}
        if self._runner is not None:  # the next child applies them in this order (HOST_WRITE_KINDS)
            for name in ("species", "masses", "charges", "total_charge", "friction_per_fs", "positions", "momenta"):
                if name in fields:
                    self._runner.host_write(self, "friction" if name == "friction_per_fs" else name, fields[name])
            self._record_friction(fields.get("friction_per_fs"))
            return

        engine = self._engine
        events = []
        if "species" in fields:
            events.append(Event("species", values=fields["species"]))
        if "masses" in fields:
            events.append(Event("masses", values=fields["masses"]))
        if "charges" in fields or "total_charge" in fields:
            events.append(Event("charges", values={"charges": fields.get("charges"),
                                                   "total_charge": fields.get("total_charge")}))
        if "friction_per_fs" in fields:
            events.append(Event("friction", values=fields["friction_per_fs"] / units.fs))
        self._queue.drain()  # frames queued at the committed step are written before it changes
        transaction = Transaction(engine, events)  # validates, before anything is written
        if "positions" in fields:
            transaction.host_writes.append(lambda state: state.positions.copy_(
                _tensor(fields["positions"], state.positions)))
        if "momenta" in fields:  # after the masses: the momenta of the new masses
            transaction.host_writes.append(lambda state: state.velocities.copy_(
                _tensor(fields["momenta"], state.velocities) / state.masses))
        transaction.run()
        if {"species", "masses"} & set(fields):
            self._composition_changed()
        self._record_friction(fields.get("friction_per_fs"))
        if writes & set(fields):
            if not self._host_writes:
                self._report.add_warning(HOST_WRITE_WARNING)
            self._host_writes += 1

    def _record_friction(self, friction_per_fs):
        """The public integrator record, its config and the plan's follow a friction change (restart records)."""
        if friction_per_fs is None:
            return
        self.integrator = dataclasses.replace(self.integrator, friction_per_fs=friction_per_fs)
        self._config = self.integrator.to_config(seed=self.seed)
        if getattr(self._plan, "integrator", None) is not None:
            self._plan = dataclasses.replace(self._plan, integrator=dataclasses.replace(
                self._plan.integrator, friction=self._config.friction))

    def set_species(self, species: Any) -> None:
        """Set every atom's species (atomic numbers or symbols, shape ``(N,)``), one event. Each must be in
        the ``ForceField``'s superset (``SpeciesNotInSuperset``); masses follow the new species' standard
        masses unless ``set_masses`` follows."""
        self.update(species=species)

    def set_masses(self, masses: Any) -> None:
        """Set every atom's mass in amu, shape ``(N,)``, one event; the momenta stay (ASE), so the velocities
        change: v' = v m_old / m_new."""
        self.update(masses=masses)

    def set_charges(self, charges: Any, *, total_charge: float | None = None) -> None:
        """Set the per-atom charges (models that take them), shape ``(N,)``, and optionally the total charge,
        one event. ``charges=None`` with ``total_charge`` changes the total only."""
        if charges is None and total_charge is None:
            raise ValueError("set_charges needs charges, total_charge or both")
        self.update(charges=charges, total_charge=total_charge)

    def set_friction(self, friction_per_fs: float) -> None:
        """Set the Langevin friction in 1/fs, one event; it drops the graphs that bake ``c1``/``c2`` in (a
        recorded re-capture)."""
        self.update(friction_per_fs=friction_per_fs)


def _carried_mc_record(previous, fresh):
    """The state a newly attached move set starts from: ``fresh``'s moves, with ``previous``'s streams, cycle
    position and cumulative counters (new move types from zero) and the step sizes of moves still present."""
    counters = {move: dict(counts) for move, counts in fresh["counters"].items()}
    counters.update({move: dict(counts) for move, counts in previous["counters"].items()})
    moves = {entry["move"] for entry in fresh["moves"]}
    return {**fresh, "cycle_position": previous["cycle_position"], "counters": counters,
            "streams": {stream: dict(position) for stream, position in previous["streams"].items()},
            "delta_A": {move: delta for move, delta in (previous.get("delta_A") or {}).items() if move in moves}}


def _tensor(array, like):
    import torch

    return torch.as_tensor(array, dtype=like.dtype, device=like.device)


# What a resume compares to decide whether the plan changed (§10.6: "another GPU or rewrite set"), and
# the rest of the record a revision carries over from the new plan.
_PLAN_FIELDS = ("tier", "rewrites", "engine", "precision", "neighbours", "polar", "integrator", "superset",
                "workload")
_PLAN_RECORDS = ("decisions", "requested", "auto_options_off", "software", "allowances")


def _planner(default, options):
    """``default`` (``minimal.default_planner``'s) with ``options``' engine and neighbour options, if given.

    A copy: the default planner's process record, software and test settings stay, and the default
    planner itself is not changed.
    """
    if options.engine is None and options.neighbours is None:
        return default
    import copy

    planner = copy.copy(default)
    if options.engine is not None:
        planner.engine = options.engine
    if options.neighbours is not None:
        planner.neighbours = options.neighbours
    return planner


def _write_json(path, record):
    """Canonical JSON (sorted keys), written durably and atomically (``io.durable.atomic_write``)."""
    import json

    from fastmace.io import durable

    durable.atomic_write(path, json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _load_model(source):
    """The pristine model of a ``simulation.json`` source: the recorded file, checked by its SHA-256.

    A label loads from its recorded path when that file is there (a run started with an explicit
    checkpoint resumes on that file), else wherever the label finds it (a run moved to another
    machine); the hash is the trust decision either way.
    """
    from fastmace.model import loading

    options = dict(sha256=source["sha256"], charge=source.get("charge"),
                   spin_multiplicity=source.get("spin_multiplicity"))
    if source.get("label"):
        if source.get("path") and Path(source["path"]).is_file():
            options["checkpoint"] = source["path"]
        return loading.load_pristine(source["label"], **options)
    return loading.load_pristine(source["path"], source.get("head"), **options)


def _resume_identity(stored, identity):
    """``RestartMismatch`` unless the restart's force field is ``identity`` up to its versions.

    Returns the version changes (``ForceFieldIdentity.version_changes``), which a resume takes with
    a warning (owner decision 23); every other difference refuses through ``check_identity``.
    """
    from fastmace.io import restart

    previous = restart.stored_identity(stored)
    versions = {} if previous is None else previous.version_changes(identity, _same_reference)
    restart.check_identity(stored, identity.with_versions_of(previous, tuple(versions)) if versions else identity)
    return versions


def _versions_text(versions):
    changes = "; ".join(f"{name} {old!r} in the file, {new!r} now" for name, (old, new) in versions.items())
    return (f"resumed under other versions than the restart's ({changes}): the same force field and corrections, "
            "recorded as a plan revision (SPEC §10.6, owner decision 23)")


def _known_reference_record(record_id):
    """The default reference record with this id that this FastMACE builds, or ``None``."""
    from fastmace.errors import FastMACEError
    from fastmace.reference import records

    parts = str(record_id).split("/")
    if len(parts) != 4 or parts[0] != records.ID_PREFIX:
        return None
    try:
        record = records.record_for(parts[1], parts[2])
    except FastMACEError:
        return None
    return record if record.id == record_id else None


def _same_reference(first, second):
    """Whether two reference record ids name the same content (they may differ in the MACE version)."""
    records = [_known_reference_record(record_id) for record_id in (first, second)]
    return None not in records and records[0].content_digest() == records[1].content_digest()


def _record_force_field(record):
    """The ``ForceField`` of a ``simulation.json`` record: its model loaded again (by SHA-256), precision,
    rewrites and species."""
    from fastmace.model.forcefield import ForceField

    rewrites = record["force_field"]["rewrites"]
    if rewrites != "auto":
        from fastmace.config import RewriteConfig

        rewrites = RewriteConfig(**rewrites)
    return ForceField(_load_model(record["model"]), precision=record["force_field"]["precision"], rewrites=rewrites,
                      species=record["force_field"]["species"])


def _restart_atoms(structure, stored):
    """The run's ``Atoms`` at the restart's positions and cell, with its composition at the restart step
    (species and masses from the restart file, ``io.restart.composition``; SPEC §10.6), not the initial
    structure's: an accepted swap or transmutation changed them since. PBC from the record."""
    from ase import Atoms

    from fastmace.io import restart

    composition = restart.composition(stored) or {}
    numbers = composition.get("species") or structure["numbers"]
    masses = composition.get("masses") or structure["masses"]
    cell = stored["cell_state"] if "cell_state" in stored else stored["cell"]
    atoms = Atoms(numbers=numbers, positions=stored["positions"], cell=cell, pbc=structure["pbc"])
    atoms.set_masses(masses)
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
