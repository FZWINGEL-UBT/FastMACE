"""The optional calibration (SPEC §9.1; plan task 5-T09, owner decision 55, D5 (a) of plans/phase-5.md).

``CalibrationConfig(calibrate=True)`` (``Simulation(calibration=)``, ``fm.md(calibrate=True)``, ``fastmace run
--calibrate``) measures, on this run's structure, the candidates the rules rank equal, and may break the tie.
Off by default. Decision 55 fixes what it may do:

- **the backend dimension only.** A tie is the rules' hybrid backend (``EngineConfig.backend="auto"``, OEq
  builds) against plain cuEq, where both pass the hard rules. The mode is no tie (graph beat compiled 1.3-4.7x
  in MD, 3-T06, and 1.1-3.9x in relaxations, 4-T05), the force precision is the ``ForceField``'s (physics, §9.2),
  and a backend the caller set is never calibrated;
- **MD plans only** (conductor's answer to 5-T09's question 2): ``calibrate=True`` with a relaxation or with MC
  moves is refused (``WORKLOAD_CODE``). An MC run's step time is not its MD steps', and a relaxation's steps
  are no steady state;
- **in children, one at a time** (§8.7: the two backends are two compile families; and one GPU's memory): each
  block is a fresh child of the subprocess runner (``SubprocessRunner.calibrate``) that builds the candidate,
  runs ``WARMUP_STEPS`` steps, then times ``CalibrationConfig.steps`` steps between synchronisations. The
  blocks alternate A B A B (``PAIRS`` pairs, the rules' plan first), so both candidates see the same drift.
  Two resident workers would need twice the memory where the run itself fits (conductor's answer 3);
- **the choice:** plain cuEq replaces the rules' choice only when it is faster than the rules' plan by more
  than ``CalibrationConfig.margin`` (5 %, above the 3 % cross-process noise) in **every** pair;
- **nothing persists:** the children work in the run's hand-off directory (``<output>/.fastmace-handoff/``, or
  the cache's ``handoff/`` without output), removed afterwards; nothing is read from an earlier run. The
  record (``Plan.calibration``, ``CalibrationRecord``) is this run's, in provenance and in the report, as
  measured (§7.7: no predicted speed-up). Its cost, every child's wall time and the probes, is ``setup_s``,
  also the report's ``setup: calibration``.

The plan the run then builds is planned in this process as any other (the winning child is not kept: the
parent builds it again, conductor's answer 7). A changed backend is in the ``EngineConfig.backend`` decision,
reason code ``CODE`` (no §9.3 rule: the backend rule is §9.4's), and in the run's recorded options, so a
resume plans the same backend. A resume and the subprocess runner's own children never calibrate.

The constants are this module's (conductor's answer 4): no per-GPU or per-structure value, nothing learnt.
"""

from __future__ import annotations

import copy
import dataclasses
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastmace.config import CalibrationConfig
    from fastmace.planner.plan import CalibrationRecord, Plan

TASK = "plan task 5-T09"
# Steps each child runs before its timed steps: the first steps' captures and allocations stay out of the time.
WARMUP_STEPS = 50
# Pairs of blocks, each the rules' plan then the alternative: A B A B.
PAIRS = 2
# The candidates, as CalibrationCandidate.change: the rules' own plan, and the plain backend.
RULES = ""
ALTERNATIVE = "backend cueq"
BACKEND_KEY = "EngineConfig.backend"
# The rules' backend reason that makes a tie: auto chose hybrid because OEq builds (minimal.MinimalPlanner._backend).
TIE_CODE = "backend-hybrid"
# The backend decision's code when the calibration changed it.
CODE = "backend-calibrated"
# calibrate=True for a workload the calibration does not cover (a relaxation, MC moves).
WORKLOAD_CODE = "calibration-workload"


@dataclass(frozen=True)
class Tie:
    """What ``tie`` found: the rules' plan (``None`` where it was not planned), the alternative that ties with it
    (``None``: no tie), and why not."""

    rules: Plan | None
    alternative: Plan | None
    why: str | None = None


@dataclass(frozen=True)
class Outcome:
    """A calibration's result: its record, the planner the run plans with (the backend set when the calibration
    changed it), the report's note when it did not measure, and a warning when a child failed."""

    record: CalibrationRecord | None
    planner: Any
    note: str | None = None
    warning: str | None = None

    @property
    def changed(self) -> bool:
        return self.record is not None and self.record.changed_plan


def workload_refusal(*, optimizer: Any, workload: Any) -> Any:
    """The ``Reason`` refusing ``calibrate=True`` for a relaxation or an MC workload, or ``None``."""
    from fastmace.errors import Reason

    if optimizer is not None:
        what = "a relaxation (its steps are no steady state to time)"
    elif workload is not None and workload.mc_moves:
        what = f"MC moves {', '.join(workload.mc_moves)} (the run's step time is not its MD steps')"
    else:
        return None
    return Reason(f"CalibrationConfig(calibrate=True) measures MD plans only, not {what}; calibrate an MD run of "
                  "the structure, or pass calibrate=False (SPEC §9.1, decision 55)", code=WORKLOAD_CODE)


def probe(planner: Any, backend: str | None = None) -> Any:
    """A copy of ``planner`` that plans as it would, without claiming anything in this process: its process
    record is a copy (a plan's compile family claim stays there), and ``backend`` replaces the engine's."""
    copied = copy.copy(planner)
    copied.process = copy.deepcopy(planner.process)
    if backend is not None:
        copied.engine = dataclasses.replace(planner.engine, backend=backend)
    return copied


def tie(planner: Any, plan_with: Callable[[Any], Plan], workload: Any) -> Tie:
    """The rules' plan and, where the backend ties, the plain cuEq plan (decision 55).

    ``plan_with(planner)`` plans the run (``workload``) with a planner; both plans are made on ``probe`` copies, so
    nothing is claimed, and a reproducible workload is not planned here at all (its plan prepares the process).
    No tie (``alternative`` ``None``, ``why`` says it) for ``reproducible=True``, when the caller set the backend,
    the rules did not choose hybrid for OEq (CPU, OEq unavailable, MACE imported first, the process's compile
    family), the plan is the reference tier or runs in a child, or plain cuEq is refused.
    """
    from fastmace.errors import PlanRefused

    if workload.reproducible:
        return Tie(None, None, "reproducible=True: rule 5 leaves plain cuEq only")
    rules = plan_with(probe(planner))
    if planner.engine.backend != "auto":
        return Tie(rules, None, f"the backend is set (EngineConfig.backend={planner.engine.backend!r})")
    if rules.tier == "reference":
        return Tie(rules, None, "the reference tier has one backend")
    if rules.runner != "in-process":
        return Tie(rules, None, "the plan runs in a child process (SPEC §8.7)")
    _key, backend, reason = rules.decision(BACKEND_KEY)
    if getattr(reason, "code", None) != TIE_CODE:
        return Tie(rules, None, f"the rules chose {backend} ({reason})")
    try:
        alternative = plan_with(probe(planner, "cueq"))
    except PlanRefused as error:
        return Tie(rules, None, f"plain cuEq is refused ({', '.join(str(code) for code in error.codes)})")
    if alternative.runner != "in-process":
        return Tie(rules, None, "plain cuEq would run in a child process (SPEC §8.7)")
    return Tie(rules, alternative)


def order(pairs: int = PAIRS) -> tuple[str, ...]:
    """The blocks in the order they run: the rules' plan, then the alternative, ``pairs`` times (A B A B)."""
    return (RULES, ALTERNATIVE) * pairs


def faster_in_every_pair(rules_ms, other_ms, margin: float) -> bool:
    """Whether ``other_ms`` is faster than ``rules_ms`` by more than ``margin`` (a share) in every pair."""
    pairs = list(zip(rules_ms, other_ms, strict=True))
    return bool(pairs) and all(other < rules * (1.0 - margin) for rules, other in pairs)


def calibrate(planner: Any, found: Tie, launch: Callable[[str], float], config: CalibrationConfig, *,
              clock: Callable[[], float] = time.perf_counter, setup_s: float = 0.0) -> Outcome:
    """Measure ``found``'s candidates with ``launch(change)`` (one block, its mean step time in ms) in ``order()``
    and choose (decision 55). ``setup_s``: what finding the tie cost, added to the record's.

    Without a tie nothing runs: no record, and the note says why. A block that raises ends the measurement:
    the rules' plan stays, the record keeps the blocks measured, and the warning says what failed.
    """
    from fastmace.planner.plan import CalibrationCandidate, CalibrationRecord

    if found.alternative is None:
        return Outcome(None, planner, note=f"not run: no tie, {found.why}")
    started = clock()
    times: dict[str, list[float]] = {RULES: [], ALTERNATIVE: []}
    warning = None
    for change in order():
        try:
            times[change].append(float(launch(change)))
        except Exception as error:  # an optional measurement: the rules' plan stays (the child's log is kept)
            from fastmace.planner.fallback import error_text

            warning = (f"calibration stopped: the {change or 'rules'} candidate's block failed "
                       f"({error_text(error)}); the rules' plan stays")
            break
    complete = warning is None
    changed = complete and faster_in_every_pair(times[RULES], times[ALTERNATIVE], config.margin)
    record = CalibrationRecord(
        candidates=(CalibrationCandidate(RULES, times[RULES]), CalibrationCandidate(ALTERNATIVE, times[ALTERNATIVE])),
        chosen=ALTERNATIVE if changed else RULES, changed_plan=changed, steps=config.steps, margin=config.margin,
        setup_s=float(setup_s + clock() - started))
    return Outcome(record, with_backend(planner, "cueq") if changed else planner, warning=warning)


def with_backend(planner: Any, backend: str) -> Any:
    """``planner`` with ``backend`` (a copy that keeps the process record: the run's planner claims as usual)."""
    copied = copy.copy(planner)
    copied.engine = dataclasses.replace(planner.engine, backend=backend)
    return copied


def apply(plan: Plan, outcome: Outcome) -> Plan:
    """``plan`` with the calibration's record; where it changed the backend, the backend decision's reason says
    so (``CODE``) and ``requested`` keeps the caller's options only (the backend was not the caller's)."""
    from fastmace.errors import Reason

    if outcome.record is None:
        return plan
    if not outcome.changed:
        return dataclasses.replace(plan, calibration=outcome.record)
    record = outcome.record
    rules, other = (_mean(candidate.step_ms) for candidate in record.candidates)
    reason = Reason(f"calibrated: plain cuEq {other:.3g} ms/step against hybrid {rules:.3g} ms/step, faster by more "
                    f"than {record.margin * 100:g} % in every pair, measured in this run (SPEC §9.1, decision 55); "
                    "the rules gave hybrid", code=CODE)
    decisions = tuple((key, choice, reason) if key == BACKEND_KEY else (key, choice, why)
                      for key, choice, why in plan.decisions)
    requested = tuple(entry for entry in plan.requested if entry[0] != BACKEND_KEY)
    return dataclasses.replace(plan, calibration=record, decisions=decisions, requested=requested)


def report_line(record: CalibrationRecord | None, note: str | None = None) -> str | None:
    """The report's ``backend:`` line of a calibration (§7.7: measured numbers only), or ``None`` without one."""
    if record is None:
        return None if note is None else f"calibration {note}"

    def name(change):
        return "rules' plan (hybrid)" if change == RULES else change

    parts = [f"{name(c.change)} " + ", ".join(f"{ms:.3g}" for ms in c.step_ms) + " ms/step"
             for c in record.candidates]
    verdict = "calibration chose plain cuEq" if record.changed_plan else "calibration kept the rules' plan"
    return (f"{verdict}; measured in this run, {record.steps} steps per block: " + "; ".join(parts)
            + f" (margin {record.margin * 100:g} % in every pair; cost {record.setup_s:.1f} s, in setup)")


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


class ChildLauncher:
    """``launch(change)`` for ``calibrate``: one block of ``simulation``'s run in a fresh child of the subprocess
    runner, with the backend of ``found``'s candidate; its mean step time, ms. ``close()`` removes the hand-off
    directory."""

    def __init__(self, simulation: Any, planner: Any, found: Tie, config: CalibrationConfig):
        from fastmace.engine.subprocess import SubprocessRunner

        self.simulation, self.planner, self.config = simulation, planner, config
        self.backends = {RULES: found.rules.engine.backend, ALTERNATIVE: found.alternative.engine.backend}
        # A fresh child per block, one at a time (conductor's answer 3); the run's own runner options otherwise.
        self.runner = SubprocessRunner(dataclasses.replace(simulation.options.runner, persistent=False))

    def __call__(self, change: str) -> float:
        from fastmace.config import AdaptationConfig, CalibrationConfig

        backend = self.backends[change]
        options = dataclasses.replace(
            self.simulation.options, engine=dataclasses.replace(self.planner.engine, backend=backend),
            adaptation=AdaptationConfig(adapt="off"), calibration=CalibrationConfig())
        seconds = self.runner.calibrate(self.simulation, options, warmup_steps=WARMUP_STEPS, steps=self.config.steps)
        return 1000.0 * seconds

    def close(self) -> None:
        self.runner.close()


__all__ = ("ALTERNATIVE", "BACKEND_KEY", "CODE", "ChildLauncher", "Outcome", "PAIRS", "RULES", "TASK", "TIE_CODE",
           "Tie", "WARMUP_STEPS", "WORKLOAD_CODE", "apply", "calibrate", "faster_in_every_pair", "order", "probe",
           "report_line", "tie", "with_backend", "workload_refusal")
