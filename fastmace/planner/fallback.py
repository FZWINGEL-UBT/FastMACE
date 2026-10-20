"""The fallback algorithm (SPEC §9.4, §9.3 rules 13–14, §9.5; decision 29).

Plan task 2-T01 fixes the contract; plan task 2-T06 implements ``candidate`` and ``activate``, and
plan task 2-T15 the search (``Fallback``, ``diagnose``, ``ddmin``, ``terminal_sequence``).

On a failure before the first step (build, compile, capture, self-check or gate, classified by the
stage that raised it), with ``FallbackConfig.fallback == "auto"``:

1. diagnose the dimension: re-run the failing check on the full rewrite set in eager mode. If eager
   passes, drop the graph mode; if not, ddmin over the rewrite set (respecting ``requires``) with
   the reduced gate (one rattled copy, the failing quantity only);
2. at most ``FallbackConfig.diagnosis_budget`` reduced-gate runs; full gate runs are counted apart,
   one per candidate to activate;
3. when the budget is spent, the terminal sequence: the same mode with every rewrite off; eager
   with every rewrite off; the reference tier if it fits and ``reference_last`` allows it;
4. a backend change after a compiled model goes eager in process or to the subprocess runner;
5. nothing passes: ``FallbackExhausted`` with every tried candidate.

Candidates are built from the pristine model (decision 24), never by undoing rewrites in place,
and a candidate becomes active only after ``activate`` validated it. Search dimensions in phase 2:
the mode (graph → eager → reference) and the rewrite set (registry records only). The physics never
changes. Every tried candidate is a ``TriedCandidate`` in ``Plan.fallbacks`` and the revision chain.

Candidates and activation (2-T06):

- ``candidate(plan, changes, force_field, atoms, *, planner=None)`` is the revision
  ``plan.revise("fallback: <change>: <reason>", 0, ...)`` with one dimension changed. The rewrite
  set is changed in the plan's resolved ``RewriteConfig`` (as rule 13's ``selfcheck_revision``
  does); the mode and the backend re-plan through ``planner`` (2-T02's eager mode, 2-T03's
  reference tier), keeping every rewrite the predecessor had off. A backend candidate with
  ``runner="subprocess"`` plans in a fresh process record, because its child plans for itself
  (§9.4 step 3, 2-T07). A candidate is only a plan: its model is the pristine one, converted again
  when it is built (decision 24).
- ``activate(candidate, planner, force_field, atoms, *, references, ...)`` builds the candidate
  through the planner in the mode it will run in (the rewrite self-checks run in the build), runs
  the gate on that engine with ``references`` (computed once by the caller, the physics being
  fixed), and makes it active only when both pass. The result is an ``Activation``: the active
  plan with its ``TriedCandidate`` appended to ``fallbacks``, or, for a failed candidate, the
  predecessor with the record appended, from which the next candidate starts. A failure is
  classified by the stage that raised it; ``ProcessConstraintError`` (2-T15 routes it to the
  runner), out-of-memory and non-finite errors propagate.

This module imports neither torch nor MACE at import time.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

TASK_CANDIDATES = "plan task 2-T06"
TASK_SEARCH = "plan task 2-T15"
# The dimensions a phase 2 fallback changes (SPEC §9.3): precision is not one (conductor answers).
DIMENSIONS = ("mode", "rewrites", "backend")
OUTCOMES = ("activated", "failed", "skipped")
# Where a failure was raised; the fallback classifies by stage, never by exception type alone.
STAGES = ("build", "compile", "capture", "self-check", "gate")


@dataclass(frozen=True)
class TriedCandidate:
    """One candidate the fallback tried (``Plan.fallbacks``, provenance).

    - ``dimension``: one of ``DIMENSIONS``; ``change``: what changed, readable ("eager",
      "rewrites fold_cutoff, density_first off");
    - ``outcome``: ``"activated"``, ``"failed"`` or ``"skipped"`` (a duplicate of an earlier one);
    - ``reason``: why it was tried, or why it failed; ``stage``: where it failed (``STAGES``);
    - ``revision``: the plan revision it became, ``None`` until activated;
    - ``reduced_gate_runs``, ``full_gate_runs``: what validating it cost.
    """

    dimension: str
    change: str
    outcome: str
    reason: str
    stage: str | None = None
    revision: int | None = None
    reduced_gate_runs: int = 0
    full_gate_runs: int = 0

    def __post_init__(self):
        if self.dimension not in DIMENSIONS:
            raise ValueError(f"TriedCandidate.dimension {self.dimension!r} is not one of {DIMENSIONS}")
        if self.outcome not in OUTCOMES:
            raise ValueError(f"TriedCandidate.outcome {self.outcome!r} is not one of {OUTCOMES}")
        if self.stage is not None and self.stage not in STAGES:
            raise ValueError(f"TriedCandidate.stage {self.stage!r} is not one of {STAGES}")


@dataclass
class Budget:
    """The fallback's counters (§9.4 step 2): reduced-gate runs against ``limit``, full gate runs apart."""

    limit: int
    reduced_gate_runs: int = 0
    full_gate_runs: int = 0

    @property
    def spent(self) -> bool:
        return self.reduced_gate_runs >= self.limit


# --------------------------------------------------------------------------- 2-T06

# The reason of a candidate revision starts with this (docs/design/contracts-2.md).
PREFIX = "fallback: "
REWRITE_OFF_CODE = "fallback-rewrite-off"
REWRITE_ON_CODE = "fallback-rewrite-on"
RUNNER_CODE = "runner-subprocess-fallback"
# What a candidate may never change (§9.4 step 1: candidates are never a change of physics).
_PHYSICS = ("integrator", "workload", "superset")
# The plan fields a candidate takes from its re-plan; the rest are the revision's or empty.
_NOT_REPLANNED = ("revision", "predecessor", "reason", "step", "fallbacks", "selfchecks")


@dataclass(frozen=True)
class Activation:
    """What ``activate`` did with one candidate.

    - ``activated``: whether the candidate passed and is now active;
    - ``plan``: the active plan (the built candidate, its ``TriedCandidate`` last in ``fallbacks``), or,
      when it failed, its predecessor with the record appended, from which the next candidate starts;
    - ``candidate``: the candidate as tried (``FallbackExhausted.revisions`` lists these);
    - ``engine``: the active engine, ``None`` when it failed;
    - ``tried``: the ``TriedCandidate``; ``gate``: the gate record, ``None`` if the gate did not run;
    - ``error``: the original exception of a failure, ``None`` when activated.
    """

    activated: bool
    plan: Any
    candidate: Any
    tried: TriedCandidate
    engine: Any = None
    gate: Any = None
    error: BaseException | None = None


def _off(value) -> bool:
    return value is False or value == "off"


def _switches(rewrites) -> tuple[str, ...]:
    """The ``RewriteConfig`` fields that switch registry records (``Toggle`` or ``bool``), not its options."""
    return tuple(f.name for f in dataclasses.fields(rewrites) if f.type in ("Toggle", "bool"))


def _rewrite_values(plan, wanted):
    """``wanted`` (a ``RewriteConfig`` or a mapping of its switches) as the switches that change, normalised."""
    from fastmace.config import RewriteConfig

    current = plan.rewrites
    if isinstance(wanted, RewriteConfig):
        wanted = {f.name: getattr(wanted, f.name) for f in dataclasses.fields(RewriteConfig)
                  if getattr(wanted, f.name) != getattr(current, f.name)}
    if not isinstance(wanted, Mapping):
        raise ValueError(f"changes['rewrites'] takes a RewriteConfig or a mapping of its fields, got {wanted!r}")
    names = _switches(current)
    changed = {}
    for name, value in wanted.items():
        if name not in names:
            raise ValueError(f"{name!r} is not a rewrite switch of RewriteConfig (the rewrite set is the registry's "
                             "records)")
        if not (isinstance(value, bool) or value in ("on", "off")):
            raise ValueError(f"rewrite {name}={value!r}: a candidate's rewrite set is resolved, \"on\" or \"off\"")
        enabled = value is True or value == "on"
        value = enabled if isinstance(getattr(current, name), bool) else ("on" if enabled else "off")
        if value != getattr(current, name):
            changed[name] = value
    if not changed:
        raise ValueError(f"changes['rewrites'] changes nothing: {sorted(wanted)} already have those values")
    return changed


def _rewrite_candidate(plan, wanted, why):
    from fastmace.errors import Reason

    changed = _rewrite_values(plan, wanted)
    off = [name for name, value in changed.items() if _off(value)]
    on = [name for name in changed if name not in off]
    change = "rewrites " + "; ".join(part for part in (f"{', '.join(off)} off" if off else "",
                                                        f"{', '.join(on)} on" if on else "") if part)
    reasons = {name: Reason(f"{PREFIX}{why} (SPEC §9.4)", code=REWRITE_OFF_CODE if name in off else REWRITE_ON_CODE)
               for name in changed}
    decisions = []
    for key, choice, reason in plan.decisions:
        field = key.split(".", 1)[1] if key.startswith("RewriteConfig.") else None
        if field in reasons:
            decisions.append((key, "off" if field in off else "on", reasons.pop(field)))
        else:
            decisions.append((key, choice, reason))
    decisions += [(f"RewriteConfig.{field}", "off" if field in off else "on", reason)
                  for field, reason in reasons.items()]
    fields = {"rewrites": dataclasses.replace(plan.rewrites, **changed), "decisions": tuple(decisions)}
    return change, fields


def _keep_off(predecessor, planned):
    """``planned`` with every rewrite the predecessor had off still off, with the predecessor's decision."""
    changed = {name: getattr(predecessor.rewrites, name) for name in _switches(predecessor.rewrites)
               if _off(getattr(predecessor.rewrites, name)) and not _off(getattr(planned.rewrites, name))}
    if not changed:
        return planned
    keys = {f"RewriteConfig.{name}" for name in changed}
    earlier = {entry[0]: entry for entry in predecessor.decisions if entry[0] in keys}
    decisions = tuple(earlier.get(entry[0], entry) for entry in planned.decisions)
    decisions += tuple(entry for key, entry in earlier.items() if key not in {d[0] for d in decisions})
    return dataclasses.replace(planned, rewrites=dataclasses.replace(planned.rewrites, **changed),
                               decisions=decisions)


def _replanned(plan, planner, force_field, atoms, *, mode, subprocess):
    from fastmace.errors import Reason

    if planner is None:
        raise ValueError("a mode or backend candidate re-plans: pass planner= (the planner of the failed plan)")
    planned = planner.plan(force_field, atoms, plan.integrator, plan.workload,
                           allow_naive_cueq="allow_naive_cueq" in plan.allowances,
                           allow_reference=mode == "reference", mode=mode)
    if subprocess and planned.runner != "subprocess":
        planned = dataclasses.replace(planned, runner="subprocess", decisions=(*planned.decisions, (
            "runner", "subprocess", Reason("a fallback candidate of another backend runs in a child process: one "
                                           "compile family per process (SPEC §8.7, §9.4 step 3)", code=RUNNER_CODE))))
    return _keep_off(plan, planned)


def candidate(plan: Any, changes: Mapping[str, Any], force_field: Any, atoms: Any, *, planner: Any = None) -> Any:
    """A plan revision of ``plan`` with one dimension changed, built from the pristine ``force_field``
    (decision 24). Not active until ``activate`` validated it.

    ``changes`` holds ``"reason"`` (why the candidate is tried) and exactly one of:

    - ``"rewrites"``: a ``RewriteConfig`` or a mapping of its fields to ``"on"``/``"off"`` (``bool``
      for a ``bool`` field), applied to the plan's resolved rewrite set;
    - ``"mode"``: one of ``minimal.MODES``, re-planned through ``planner``;
    - ``"backend"``: ``"cueq"`` or ``"hybrid"``, re-planned through a copy of ``planner`` with that
      backend; with ``"runner": "subprocess"`` the candidate runs in a child and claims no compile
      family here.

    The revision is ``plan.revise("fallback: <change>: <reason>", 0, ...)``. Raises ``ValueError`` for
    anything else, or when the candidate would change nothing or the physics; refusals and
    ``ProcessConstraintError`` of the re-plan propagate.
    """
    from fastmace.planner import minimal

    changes = dict(changes)
    why = changes.pop("reason", None)
    runner = changes.pop("runner", None)
    dimensions = [name for name in changes if name in DIMENSIONS]
    unknown = sorted(set(changes) - set(DIMENSIONS))
    if unknown:
        raise ValueError(f"candidate changes {unknown} are not fallback dimensions; they are one of {DIMENSIONS} "
                         "(precision is not one in phase 2, and the physics never changes)")
    if runner is not None and (dimensions != ["backend"] or runner != "subprocess"):
        raise ValueError("changes['runner'] is only 'subprocess', with a backend change (SPEC §9.4 step 3)")
    if len(dimensions) != 1:
        raise ValueError(f"a candidate changes one dimension, one of {DIMENSIONS}; got {dimensions or 'none'}")
    if not why:
        raise ValueError("a candidate needs changes['reason']: why it is tried")
    dimension = dimensions[0]
    value = changes[dimension]
    if dimension == "rewrites":
        change, fields = _rewrite_candidate(plan, value, why)
    else:
        if dimension == "mode":
            if value not in minimal.MODES:
                raise ValueError(f"mode must be one of {minimal.MODES}, got {value!r}")
            if value == plan.tier:
                raise ValueError(f"changes['mode'] changes nothing: the plan's mode is {plan.tier!r}")
            planned = _replanned(plan, planner, force_field, atoms, mode=value, subprocess=False)
            change = f"mode {value}"
        else:
            if value not in ("cueq", "hybrid"):
                raise ValueError(f"backend must be 'cueq' or 'hybrid', got {value!r}")
            if value == plan.engine.backend:
                raise ValueError(f"changes['backend'] changes nothing: the plan's backend is {value!r}")
            if planner is None:
                raise ValueError("a mode or backend candidate re-plans: pass planner= (the planner of the failed plan)")
            other = copy.copy(planner)
            other.engine = dataclasses.replace(planner.engine, backend=value)
            if runner == "subprocess":
                from fastmace.engine import constraints

                other.process = constraints.ProcessRecord()  # the child plans for itself (2-T07)
            planned = _replanned(plan, other, force_field, atoms, mode=plan.tier, subprocess=runner == "subprocess")
            change = f"backend {value}" + (" (subprocess)" if planned.runner == "subprocess" else "")
        for name in _PHYSICS:
            if getattr(planned, name) != getattr(plan, name):
                raise ValueError(f"the {dimension} candidate changes the physics ({name}); a fallback never does "
                                 "(SPEC §9.4 step 1)")
        if planned.precision.dtype != plan.precision.dtype:
            raise ValueError(f"the {dimension} candidate changes the physics (force precision "
                             f"{plan.precision.dtype} -> {planned.precision.dtype}); a fallback never does")
        fields = {f.name: getattr(planned, f.name) for f in dataclasses.fields(planned)
                  if f.name not in _NOT_REPLANNED}
    return plan.revise(f"{PREFIX}{change}: {why}", 0, selfchecks=(), **fields)


def _parse(candidate_plan):
    """``(dimension, change, reason)`` of a candidate revision; ``ValueError`` for any other plan."""
    reason = str(candidate_plan.reason)
    parts = reason.split(": ", 2)
    if candidate_plan.predecessor is None or not reason.startswith(PREFIX) or len(parts) != 3:
        raise ValueError(f"activate takes a fallback candidate (fallback.candidate), not a plan with reason "
                         f"{reason!r}")
    dimension = parts[1].split(" ", 1)[0]
    if dimension not in DIMENSIONS:
        raise ValueError(f"the candidate's change {parts[1]!r} names no dimension of {DIMENSIONS}")
    return dimension, parts[1], parts[2]


def _propagates(error) -> bool:
    """Errors that are not candidate failures (conductor answers, T15 Q7): process constraints, memory, non-finite."""
    from fastmace.errors import NonFiniteError, ProcessConstraintError

    if isinstance(error, (ProcessConstraintError, NonFiniteError, MemoryError)):
        return True
    return type(error).__name__ == "OutOfMemoryError"  # torch.cuda.OutOfMemoryError, without importing torch


def _build_stage(error) -> str:
    from fastmace.errors import SelfCheckError

    if isinstance(error, SelfCheckError):
        return "self-check"
    if type(error).__module__.startswith(("torch._dynamo", "torch._inductor")):
        return "compile"
    return "build"


def _text(error) -> str:
    """The error's class and first line; a gate error adds its record's reason."""
    first = str(error).splitlines()[0] if str(error) else ""
    text = f"{type(error).__name__}: {first}" if first else type(error).__name__
    record = getattr(error, "gate_record", None)
    if record is not None and getattr(record, "reason", None):
        text += f" ({record.reason})"
    return text


def activate(candidate: Any, planner: Any, force_field: Any, atoms: Any, *, references: Any, reduced: bool = False,
             quantities: Sequence[str] | None = None, budget: Budget | None = None, compile_cache: Any = None,
             temperature_K: float | None = None) -> Activation:
    """Run the self-checks and the gate on ``candidate`` in the mode it will run in, and only then
    make it active; a failing candidate is recorded and never activated (§9.4 step 4).

    - ``planner`` builds the candidate (``planner.build``) from ``force_field``'s pristine model;
    - ``references``: the gate's references (``selfcheck.references``), computed once by the caller;
      a skipped one cannot validate, so it raises ``ValueError``;
    - ``reduced``: these references are the reduced gate's (one rattled copy), so the run counts as a
      reduced-gate run, else as a full one; ``quantities`` go to ``selfcheck.production`` (the failing
      quantity of the reduced gate);
    - ``budget``: counted, when given.

    Only a gate record that passed (``passed``, ``passed-reduced``) validates; ``unavailable`` does not.
    A ``runner="subprocess"`` candidate cannot be built here: ``ProcessConstraintError``, which 2-T15
    routes to the runner, whose child validates it.
    """
    from fastmace.errors import GateFailed, ProcessConstraintError
    from fastmace.planner import selfcheck

    dimension, change, why = _parse(candidate)
    if references is None or references.skipped:
        raise ValueError("activate needs the gate's references (selfcheck.references with check='gate'): a "
                         "candidate is validated by the gate before activation (SPEC §9.4 step 4)")
    if candidate.runner == "subprocess":
        raise ProcessConstraintError(
            f"the fallback candidate ({change}) runs in a child process (runner='subprocess'); it is built and "
            "validated there, through the subprocess runner (SPEC §8.7, §9.4 step 3)")

    def failed(stage, error, gate=None, runs=0):
        tried = TriedCandidate(dimension=dimension, change=change, outcome="failed",
                               reason=f"{why}; failed at {stage}: {_text(error)}", stage=stage,
                               reduced_gate_runs=runs if reduced else 0, full_gate_runs=0 if reduced else runs)
        base = dataclasses.replace(candidate.predecessor, fallbacks=(*candidate.predecessor.fallbacks, tried))
        return Activation(activated=False, plan=base, candidate=candidate, tried=tried, gate=gate, error=error)

    try:
        engine = planner.build(candidate, force_field, atoms, compile_cache=compile_cache,
                               temperature_K=temperature_K)
    except Exception as error:
        if _propagates(error):
            raise
        return failed(_build_stage(error), error)

    if budget is not None:
        if reduced:
            budget.reduced_gate_runs += 1
        else:
            budget.full_gate_runs += 1
    extra = {} if quantities is None else {"quantities": tuple(quantities)}
    try:
        gate = selfcheck.production(engine, references, check=references.check, **extra)
    except GateFailed as error:
        del engine
        return failed("gate", error, gate=error.gate_record, runs=1)
    except Exception as error:
        if _propagates(error):
            raise
        del engine
        return failed("gate", error, runs=1)
    if not gate.passed:
        del engine
        return failed("gate", RuntimeError(f"gate {gate.status}: {gate.reason}"), gate=gate, runs=1)

    tried = TriedCandidate(dimension=dimension, change=change, outcome="activated", reason=why,
                           revision=candidate.revision, reduced_gate_runs=1 if reduced else 0,
                           full_gate_runs=0 if reduced else 1)
    built = getattr(engine, "plan", None) or candidate
    active = dataclasses.replace(built, fallbacks=(*candidate.fallbacks, tried))
    engine.plan = active
    return Activation(activated=True, plan=active, candidate=candidate, tried=tried, engine=engine, gate=gate)


# --------------------------------------------------------------------------- 2-T15


class Fallback:
    """The §9.4 search for one failure: ``Fallback(planner, force_field, atoms, integrator, references,
    config).run(failure)`` returns the activated plan revision or raises ``FallbackExhausted``."""

    def __init__(self, planner: Any, force_field: Any, atoms: Any, integrator: Any, references: Any,
                 config: Any = None):
        from fastmace.config import FallbackConfig

        self.planner = planner
        self.force_field = force_field
        self.atoms = atoms
        self.integrator = integrator
        self.references = references
        self.config = FallbackConfig() if config is None else config

    def run(self, failure: BaseException) -> Any:
        raise NotImplementedError(f"the fallback search is {TASK_SEARCH}")


def diagnose(plan: Any, failure: BaseException, *args: Any, **kwargs: Any) -> str:
    """The failing dimension (§9.4 step 2): ``"mode"`` if eager passes, else ``"rewrites"``."""
    raise NotImplementedError(f"the fallback diagnosis is {TASK_SEARCH}")


def ddmin(rewrites: Sequence[str], test: Callable[[frozenset[str]], bool], *,
          requires: Mapping[str, Sequence[str]], budget: Budget | None) -> frozenset[str]:
    """A 1-minimal failing subset of ``rewrites`` under the ``requires`` closure (delta debugging).

    ``test(subset)`` is True when the reduced gate fails with ``subset`` enabled; each call counts
    against ``budget``."""
    raise NotImplementedError(f"ddmin over the rewrite set is {TASK_SEARCH}")


def terminal_sequence(plan: Any, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
    """The fixed candidates after the budget is spent (§9.4 step 2): the same mode with every rewrite
    off, eager with every rewrite off, the reference tier if it fits."""
    raise NotImplementedError(f"the terminal sequence is {TASK_SEARCH}")
