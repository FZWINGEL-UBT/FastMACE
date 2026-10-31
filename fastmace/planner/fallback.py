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

Mid-run (plan task 5-T08, §9.4 step 5, decision 54): ``Simulation`` runs the same search for a capture error at a
re-capture or a compile error of the compiled tier during ``run()`` (``mid_run_stage``), on the committed
structure of its last canonical point, with ``Fallback(step=, note=)``: every candidate revision is made at the
point's step and carries ``mid_run_note``'s text. ``mid_run_fallbacks`` reads them back from a plan chain
(``counters.fallbacks``), ``mid_run_revision`` adds the decision ``fallback.mid_run``, and ``mid_run_limit`` is
the stop at a second failure in one canonical interval.

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
# The dimensions a fallback changes (SPEC §9.3): the MC energy precision first (3-T11, §9.4 step 2), then
# phase 2's. The force precision is not one (conductor answers): it is the physics the user chose.
DIMENSIONS = ("mc_energy", "mode", "rewrites", "backend")
# "passed": a diagnosis run that passed the reduced gate (never activated; 2-T15).
OUTCOMES = ("activated", "passed", "failed", "skipped")
# Where a failure was raised; the fallback classifies by stage, never by exception type alone.
STAGES = ("build", "compile", "capture", "self-check", "gate")


@dataclass(frozen=True)
class TriedCandidate:
    """One candidate the fallback tried (``Plan.fallbacks``, provenance).

    - ``dimension``: one of ``DIMENSIONS``; ``change``: what changed, readable ("eager",
      "rewrites fold_cutoff, density_first off");
    - ``outcome``: ``"activated"``, ``"passed"`` (a diagnosis run that passed the reduced gate, never
      activated), ``"failed"`` or ``"skipped"`` (a duplicate of an earlier one, or not planned);
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
    reasons = {name: Reason(f"{PREFIX}{why} (SPEC §9.4)", rule=14,
                            code=REWRITE_OFF_CODE if name in off else REWRITE_ON_CODE)
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
                           allow_reference=mode == "reference", mode=mode, optimizer=plan.optimizer)
    if subprocess and planned.runner != "subprocess":
        planned = dataclasses.replace(planned, runner="subprocess", decisions=(*planned.decisions, (
            "runner", "subprocess", Reason("a fallback candidate of another backend runs in a child process: one "
                                           "compile family per process (SPEC §8.7, §9.4 step 3)", rule=14,
                                           code=RUNNER_CODE))))
    return _keep_off(plan, planned)


def candidate(plan: Any, changes: Mapping[str, Any], force_field: Any, atoms: Any, *, planner: Any = None,
              step: int = 0, note: str | None = None) -> Any:
    """A plan revision of ``plan`` with one dimension changed, built from the pristine ``force_field``
    (decision 24). Not active until ``activate`` validated it.

    ``changes`` holds ``"reason"`` (why the candidate is tried) and exactly one of:

    - ``"mc_energy"``: ``"float64"``, the MC energy path in float64 while MD forces keep their
      precision (3-T11; nothing is re-planned);

    - ``"rewrites"``: a ``RewriteConfig`` or a mapping of its fields to ``"on"``/``"off"`` (``bool``
      for a ``bool`` field), applied to the plan's resolved rewrite set;
    - ``"mode"``: one of ``minimal.MODES``, re-planned through ``planner``;
    - ``"backend"``: ``"cueq"`` or ``"hybrid"``, re-planned through a copy of ``planner`` with that
      backend; with ``"runner": "subprocess"`` the candidate runs in a child and claims no compile
      family here.

    The revision is ``plan.revise("fallback: <change>: <reason>", step, ...)``; a mid-run fallback (5-T08) passes
    the committed point's ``step`` and its ``note`` (``mid_run_note``), which goes before the reason. Raises
    ``ValueError`` for anything else, or when the candidate would change nothing or the physics; refusals
    and ``ProcessConstraintError`` of the re-plan propagate.
    """
    from fastmace.planner import minimal

    changes = dict(changes)
    why = changes.pop("reason", None)
    runner = changes.pop("runner", None)
    dimensions = [name for name in changes if name in DIMENSIONS]
    unknown = sorted(set(changes) - set(DIMENSIONS))
    if unknown:
        raise ValueError(f"candidate changes {unknown} are not fallback dimensions; they are one of {DIMENSIONS} "
                         "(the force precision is not one, and the physics never changes)")
    if runner is not None and (dimensions != ["backend"] or runner != "subprocess"):
        raise ValueError("changes['runner'] is only 'subprocess', with a backend change (SPEC §9.4 step 3)")
    if len(dimensions) != 1:
        raise ValueError(f"a candidate changes one dimension, one of {DIMENSIONS}; got {dimensions or 'none'}")
    if not why:
        raise ValueError("a candidate needs changes['reason']: why it is tried")
    if note:
        why = f"{note}: {why}"
    dimension = dimensions[0]
    value = changes[dimension]
    if dimension == "mc_energy":
        # The MC energy path only (3-T11): MD forces keep their precision, so the force path is unchanged.
        if value != "float64":
            raise ValueError(f"changes['mc_energy'] is 'float64' (MC energies move up, never below the force "
                             f"precision), got {value!r}")
        if plan.precision.mc_energy == value:
            raise ValueError("changes['mc_energy'] changes nothing: the plan's MC energies are float64 already")
        return plan.revise(f"{PREFIX}mc_energy {value}: {why}", step, selfchecks=(),
                           precision=dataclasses.replace(plan.precision, mc_energy=value))
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
    revised = plan.revise(f"{PREFIX}{change}: {why}", step, selfchecks=(), **fields)
    return memory_revised(revised, force_field, atoms, planner) if dimension == "rewrites" else revised


def memory_revised(plan: Any, force_field: Any, atoms: Any, planner: Any = None) -> Any:
    """``plan`` with rule 12 run again for its rewrites (5-T03): a revision that turns a rewrite off without a
    re-plan (a rewrite candidate, rule 13) would otherwise keep a prediction that counted the memory the rewrite
    saved. Unchanged when the plan has no estimate or the force field no traits."""
    from fastmace.planner import minimal

    traits = getattr(force_field, "traits", None)
    if plan.predicted_memory_bytes is None or traits is None:
        return plan
    reviser = planner if isinstance(planner, minimal.MinimalPlanner) else minimal.MinimalPlanner(engine=plan.engine)
    return reviser.revise_memory(plan, traits, atoms, reviser.hardware())


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
    """Errors that are not candidate failures (conductor answers, T15 Q7): process constraints, memory, non-finite,
    and input errors (an input no candidate can run)."""
    from fastmace.errors import InputError, NonFiniteError, ProcessConstraintError

    if isinstance(error, (ProcessConstraintError, NonFiniteError, InputError, MemoryError)):
        return True
    return _out_of_memory(error)


def _with_estimate(error, plan):
    """An out-of-memory error with rule 12's estimate in its message (decision 51); any other error as it is."""
    from fastmace.planner.memory import annotate_out_of_memory

    return annotate_out_of_memory(error, plan)


def _out_of_memory(error) -> bool:
    """The caching allocator's ``torch.cuda.OutOfMemoryError`` (without importing torch), a host ``MemoryError``,
    or the driver's "CUDA error: out of memory" (``torch.AcceleratorError``, e.g. a launch or capture that could
    not reserve its stack; CAP-T07): never a fallback trigger, since the search's re-plans read the free memory
    (``cudaMemGetInfo``), which fails once such an error left the context sticky."""
    if isinstance(error, MemoryError) or type(error).__name__ == "OutOfMemoryError":
        return True
    return "cuda error: out of memory" in str(error).lower()


def _context_usable(device) -> bool:
    """Can the process still use ``device`` after an error (CAP-T07)? A CUDA error during a capture can leave the
    context sticky: every later call fails, ``cudaMemGetInfo`` included. ``True`` off CUDA."""
    if not str(device).startswith("cuda"):
        return True
    try:
        import torch

        torch.cuda.synchronize(device)
        torch.cuda.mem_get_info(torch.device(device))
    except Exception:  # noqa: BLE001 - any failure here means the context cannot run a rebuild
        return False
    return True


def memory_rescue(plan: Any, error: BaseException, force_field: Any, atoms: Any, *, planner: Any, stage: str,
                  usable: Callable[[Any], bool] = _context_usable) -> Any:
    """The plan to rebuild with after ``error`` at ``stage`` (``"build"`` or ``"capture"``), or ``None``
    (plans/capacity.md CAP-T07; the owner's rule: memory-saving records only when memory would not be enough).

    ``None`` unless ``error`` is an out-of-memory error and the plan has a record of ``memory.MEMORY_RESCUE`` off
    that the caller left "auto" and whose requirements are on, or a record of ``memory.RESCUE_OFF`` on that the
    caller left "auto". Then, if the device is still usable, a revision of ``plan`` with the first such record in
    ``MEMORY_RESCUE``'s order set (``candidate``: "fallback: rewrites recompute_conv_weights on: out of memory at
    build; ..."), which the run record keeps in ``Plan.revisions``. One record per rebuild (CAP-T08):
    ``recompute_node_segments`` only after ``recompute_conv_weights`` is on. If the context is not usable (a sticky
    CUDA error), ``error`` is raised again with a note naming the fix: the records set in a fresh process.
    """
    from fastmace.planner.memory import MEMORY_RESCUE, annotate_out_of_memory, rescue_value

    if not _out_of_memory(error):
        return None
    requested = getattr(force_field, "rewrites", "auto")
    names = []
    for name in MEMORY_RESCUE:
        if requested != "auto" and getattr(requested, name, "auto") != "auto":
            continue  # the caller's explicit value is kept
        if not _rescuable(plan, name):
            continue
        record = _record(name)
        if record is None:
            continue
        if rescue_value(name) == "off" or all(not _off(getattr(plan.rewrites, other, "off"))
                                              for other in record.requires):
            names.append(name)
            break  # one record per rebuild, in MEMORY_RESCUE's order
    if not names:
        return None
    if not usable(plan.engine.device):
        fix = ", ".join(f'{name}="{rescue_value(name)}"' for name in names)
        note = (f"FastMACE (CAP-T07): out of memory at {stage}, and the CUDA context is not usable afterwards, so "
                f"this process cannot rebuild with {', '.join(names)} on. Run again in a fresh process with "
                f"fm.ForceField(..., rewrites=RewriteConfig({fix})).")
        annotate_out_of_memory(error, plan)
        message = str(error.args[0]) if error.args else ""
        if note not in message:
            error.args = (f"{message}\n{note}" if message else note, *error.args[1:])
        raise error
    why = (f"out of memory at {stage}; " + ", ".join(f"{name} auto -> {rescue_value(name)}" for name in names)
           + " (the memory rescue, rule 12; CAP-T07)")
    return candidate(plan, {"rewrites": {name: rescue_value(name) for name in names}, "reason": why}, force_field,
                     atoms, planner=planner)


def _rescuable(plan: Any, name: str) -> bool:
    """Whether the memory rescue can still set ``name``: a record it turns on is off in ``plan``, a record it turns
    off (``memory.RESCUE_OFF``) is on. A field the plan lacks counts as already set."""
    from fastmace.planner.memory import rescue_value

    value = getattr(plan.rewrites, name, None)
    if value is None:
        return False
    return not _off(value) if rescue_value(name) == "off" else _off(value)


def raise_if_context_unusable(failure: Any, force_field: Any, *,
                              usable: Callable[[Any], bool] = _context_usable) -> None:
    """Raise ``failure.error`` with an explanation when the CUDA context is not usable after it (CAP-T07).

    A driver out-of-memory error during a launch can surface asynchronously as another error (on WSL2, NbCl5
    46,656 atoms: ``CUBLAS_STATUS_EXECUTION_FAILED``) and leave the context sticky; the fallback search's re-plans
    then fail in ``cudaMemGetInfo``. When rule 12 had warned that the plan may not fit, the note names the memory
    rescue's records (set "on", when they are off) or a smaller cell as the fix."""
    from fastmace.planner.memory import MEMORY_RESCUE, rescue_value

    plan = failure.plan
    if usable(plan.engine.device):
        return
    error = failure.error
    try:
        tight = plan.decision("memory")[1] == "warn"
    except (KeyError, AttributeError, TypeError, ValueError):
        tight = False
    off = [name for name in MEMORY_RESCUE if _rescuable(plan, name)]
    note = (f"FastMACE: the CUDA context is not usable after this {failure.stage} error, so the fallback cannot run "
            "in this process.")
    if tight:
        fix = "a smaller cell"
        if off:
            fix = ("fm.ForceField(..., rewrites=RewriteConfig("
                   + ", ".join(f'{name}="{rescue_value(name)}"' for name in off) + "))")
        note += (f" Rule 12 had warned that the plan may not fit the free GPU memory, so it is most likely out of "
                 f"memory: run again in a fresh process with {fix}.")
    message = str(error.args[0]) if error.args else ""
    if note not in message:
        error.args = (f"{message}\n{note}" if message else note, *error.args[1:])
    raise error


def _record(name):
    from fastmace.planner import minimal

    registry = minimal._rewrite_registry()
    return registry[name] if name in registry.names() else None


def stage_of(error, default: str = "build") -> str:
    """The stage an error before the first step was raised in: ``"capture"`` for an error the graph
    engine tagged while capturing (``engine.graph.CAPTURE_STAGE``, 2-T15b), ``"self-check"``,
    ``"compile"`` (Dynamo or Inductor), else ``default``."""
    from fastmace.errors import SelfCheckError

    if getattr(error, "fastmace_stage", None) == "capture":
        return "capture"
    if isinstance(error, SelfCheckError):
        return "self-check"
    if type(error).__module__.startswith(("torch._dynamo", "torch._inductor")):
        return "compile"
    return default


def _build_stage(error) -> str:
    return stage_of(error)


def _prepare(engine) -> None:
    """The engine's captures before its first step and before the gate (``GraphEngine.prepare``, 2-T15b)."""
    prepare = getattr(engine, "prepare", None)
    if prepare is not None:
        prepare()


def detach(error: BaseException | None) -> BaseException | None:
    """``error`` without the frames its traceback holds (plan task 2-T15c), for every error the fallback keeps.

    A traceback holds each frame it passed through, and those frames hold their locals: the failed
    engine (``selfcheck.production(engine, ...)``, a caller's ``engine``), and on CUDA its memory,
    which then stays allocated while the next candidate is built. The type, message and chain stay;
    the formatted traceback is kept as text in ``error.fastmace_traceback``; then the frames of the
    error and of its ``__cause__``/``__context__`` chain are cleared and dropped. Idempotent.
    """
    import traceback

    if error is None:
        return None
    if getattr(error, "fastmace_traceback", None) is None:
        text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        try:
            error.fastmace_traceback = text
        except AttributeError:  # pragma: no cover - an exception type without a __dict__
            pass
    seen, todo = set(), [error]
    while todo:
        current = todo.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if current.__traceback__ is not None:
            traceback.clear_frames(current.__traceback__)  # frames still executing are left as they are
            current.__traceback__ = None
        todo += [current.__cause__, current.__context__]
    return error


# An error text that names a cause stays on one line of about this many characters (3-T17).
ERROR_TEXT_LIMIT = 300


def _first_line(error) -> str:
    lines = str(error).splitlines()
    return lines[0].strip() if lines else ""


def _innermost(error: BaseException) -> BaseException:
    """The innermost error of ``error``'s chain: ``__cause__``, else ``__context__`` unless suppressed."""
    seen, current = {id(error)}, error
    while True:
        following = current.__cause__
        if following is None and not current.__suppress_context__:
            following = current.__context__
        if following is None or id(following) in seen:
            return current
        seen.add(id(following))
        current = following


def error_text(error: BaseException, head: str | None = None) -> str:
    """The error's class and first line (or ``head``), and the innermost error of its chain (3-T17).

    A wrapped compile error's first line names no cause: Inductor's ``BackendCompilerFailed`` reads
    ``backend='inductor' raised:``, and 2-T21d's reason stopped there, dropping ``RuntimeError:
    Attempting to use a closed pool``. The innermost error of the ``__cause__``/``__context__`` chain
    (or, with no chain, the line under a trailing ``raised:``) is added as ``Type: first line``. An
    error with no cause gives the class and first line alone, as before.
    """
    head = _first_line(error) if head is None else head
    text = f"{type(error).__name__}: {head}" if head else type(error).__name__
    inner = _innermost(error)
    if inner is not error:
        message = _first_line(inner)
        cause = f"{type(inner).__name__}: {message}" if message else type(inner).__name__
        if message and message in head:
            return text
    else:
        lines = [line.strip() for line in str(error).splitlines()[1:]]
        cause = next((line for line in lines if line), "") if head.endswith("raised:") else ""
        if not cause:
            return text
    text += (" " if head.endswith(":") else "; innermost ") + cause
    return text if len(text) <= ERROR_TEXT_LIMIT else text[:ERROR_TEXT_LIMIT - 1] + "…"


def _text(error) -> str:
    """The error's class and first line, with the innermost cause (``error_text``); a gate error adds its
    record's reason."""
    text = error_text(error)
    record = getattr(error, "gate_record", None)
    if record is not None and getattr(record, "reason", None) and record.reason not in text:
        text += f" ({record.reason})"
    return text


def activate(candidate: Any, planner: Any, force_field: Any, atoms: Any, *, references: Any, reduced: bool = False,
             quantities: Sequence[str] | None = None, budget: Budget | None = None, compile_cache: Any = None,
             temperature_K: float | None = None, moves: Any = None) -> Activation:
    """Run the self-checks and the gate on ``candidate`` in the mode it will run in, and only then
    make it active; a failing candidate is recorded and never activated (§9.4 step 4).

    - ``planner`` builds the candidate (``planner.build``) from ``force_field``'s pristine model;
    - ``references``: the gate's references (``selfcheck.references``), computed once by the caller;
      a skipped one cannot validate, so it raises ``ValueError``;
    - ``reduced``: these references are the reduced gate's (one rattled copy), so the run counts as a
      reduced-gate run, else as a full one; ``quantities`` go to ``selfcheck.production`` (the failing
      quantity of the reduced gate);
    - ``budget``: counted, when given;
    - ``moves``: an MC plan's move-energy references (``selfcheck.move_references``); the candidate then
      passes the move-energy gate too (``selfcheck.move_gate``), and its MC energy precision is the one
      the gate chose (``mc_energy_revision``, 3-T11).

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
        detach(error)  # the Activation keeps it; its frames would keep this candidate's engine (2-T15c)
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
            raise _with_estimate(error, candidate)
        return failed(_build_stage(error), error)
    try:
        _prepare(engine)
    except Exception as error:
        if _propagates(error):
            raise _with_estimate(error, candidate)
        del engine
        return failed(stage_of(error, "capture"), error)

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
        return failed(stage_of(error, "gate"), error, runs=1)
    if not gate.passed:
        del engine
        return failed("gate", RuntimeError(f"gate {gate.status}: {gate.reason}"), gate=gate, runs=1)
    if moves is not None:
        try:
            gate = selfcheck.move_gate(engine, moves, plan=candidate, force_field=force_field, record=gate)
        except GateFailed as error:
            del engine
            return failed("gate", error, gate=error.gate_record, runs=0)

    tried = TriedCandidate(dimension=dimension, change=change, outcome="activated", reason=why,
                           revision=candidate.revision, reduced_gate_runs=1 if reduced else 0,
                           full_gate_runs=0 if reduced else 1)
    built = getattr(engine, "plan", None) or candidate
    active = mc_energy_revision(dataclasses.replace(built, fallbacks=(*candidate.fallbacks, tried)), gate)
    engine.plan = active
    return Activation(activated=True, plan=active, candidate=candidate, tried=tried, engine=engine, gate=gate)


# --------------------------------------------------------------------------- 2-T15

FAILED_CODE = "fallback-exhausted"
# The reasons of the search's candidates start with these (the report and tests read them).
MODE_WHY = "eager passes the reduced gate: the fault is in capture or compilation"
DDMIN_WHY = "ddmin: minimal failing set"
TERMINAL_WHY = "terminal sequence"
RULE_13_WHY = "rule 13"


class DiagnosisBudgetSpent(Exception):
    """The reduced-gate budget of one fallback is spent (§9.4 step 2): the terminal sequence runs next."""


@dataclass(frozen=True)
class Failure:
    """What failed before the first step: the ``plan``, the ``stage`` that raised (``STAGES``), the
    original ``error`` and, for a gate failure, its ``gate`` record."""

    plan: Any
    stage: str
    error: BaseException
    gate: Any = None

    def __post_init__(self):
        if self.stage not in STAGES:
            raise ValueError(f"Failure.stage {self.stage!r} is not one of {STAGES}")

    def quantities(self) -> tuple[str, ...]:
        """The quantities the reduced gate judges: the gate's failing ones, else all (§9.4 step 2)."""
        from fastmace.planner import selfcheck

        record = self.gate
        if record is None or getattr(record, "nonfinite", ()):
            return selfcheck.QUANTITIES
        failing = tuple(q for q in selfcheck.QUANTITIES
                        if not (getattr(record, "quantities", None) or {}).get(q, {}).get("passed", True))
        return failing or selfcheck.QUANTITIES


def move_energy_only(failure: Failure) -> bool:
    """Did only the move-energy part of the gate fail (SPEC §9.4 step 2; 3-T11)? The energy, force and
    stress gate passed or was skipped, and ``move_energy.status`` is ``failed``."""
    record = failure.gate
    part = getattr(record, "move_energy", None) or {}
    return (failure.stage == "gate" and part.get("status") == "failed"
            and getattr(record, "status", None) in ("passed", "passed-reduced", "skipped"))


MC_ENERGY_WHY = "the move-energy gate"


def mc_energy_revision(plan: Any, record: Any) -> Any:
    """``plan`` with the MC energy precision the move-energy gate chose (``record.mc_energy_precision``), as a
    fallback revision in the ``mc_energy`` dimension with its ``TriedCandidate`` (3-T11; SPEC §9.4 step 2,
    decision 35). The plan itself when the gate kept the plan's precision or did not run.

    The engine is the same: MD forces keep their precision, and only the MC evaluator reads the field.
    """
    chosen = getattr(record, "mc_energy_precision", None)
    if chosen is None or plan.precision is None or chosen == plan.precision.mc_energy:
        return plan
    part = getattr(record, "move_energy", None) or {}
    why = part.get("reason") or MC_ENERGY_WHY
    revised = candidate(plan, {"mc_energy": chosen, "reason": why}, None, None)
    tried = TriedCandidate(dimension="mc_energy", change=f"mc_energy {chosen}", outcome="activated", reason=why,
                           stage=None, revision=revised.revision)
    return dataclasses.replace(revised, fallbacks=(*plan.fallbacks, tried))


@dataclass(frozen=True)
class Result:
    """The fallback's outcome: the active ``plan`` revision (every tried candidate in its
    ``fallbacks``), its ``engine`` and validating ``gate`` record, the ``budget`` and, when the
    candidate runs in a child (§9.4 step 3), ``runner=True`` with no engine here."""

    plan: Any
    engine: Any
    gate: Any
    budget: Budget
    tried: tuple[TriedCandidate, ...]
    runner: bool = False


def replay_off(stored: Any, planned: Any) -> Any:
    """``planned`` (a fresh plan in ``stored``'s mode) with every rewrite ``stored`` had off still off,
    with ``stored``'s decisions: how a resume rebuilds a fallback revision (conductor answers, T15 Q9)."""
    return _keep_off(stored, planned)


def stored_fallback_plan(record: Mapping[str, Any] | None) -> Any:
    """The plan of a restart's plan record when it, or a revision before it, is a fallback; else ``None``."""
    from fastmace.planner.plan import Plan

    if not record:
        return None
    plan = Plan.from_restart(record)
    reasons = [str(plan.reason), *(str(p.reason) for p in plan.chain())]
    return plan if any(reason.startswith(PREFIX) for reason in reasons) else None


def enabled_rewrites(plan: Any) -> tuple[str, ...]:
    """The ``RewriteConfig`` switches that are on in ``plan``, in the registry's application order."""
    if plan is None or plan.rewrites is None:
        return ()
    on = [name for name in _switches(plan.rewrites) if not _off(getattr(plan.rewrites, name))
          and getattr(plan.rewrites, name) in ("on", True)]
    order = {name: index for index, name in enumerate(_field_order())}
    return tuple(sorted(on, key=lambda name: (order.get(name, len(order)), name)))


def _field_order(registry=None):
    from fastmace.rewrites import registry as reg

    registry = registry or reg.default_registry()
    return tuple(dict.fromkeys(registry.config_field(name) for name in registry.order()))


def field_requires(units: Sequence[str], registry: Any = None) -> dict[str, tuple[str, ...]]:
    """``requires`` between the switches in ``units``: a switch requires another when one of its
    records requires one of the other's (the registry's records, §6.1)."""
    from fastmace.rewrites import registry as reg

    registry = registry or reg.default_registry()
    units = set(units)
    out: dict[str, list[str]] = {}
    for name in registry.order():
        field = registry.config_field(name)
        if field not in units:
            continue
        for required in registry[name].requires:
            other = registry.config_field(required) if required in registry.names() else required
            if other != field and other in units and other not in out.setdefault(field, []):
                out[field].append(other)
    return {field: tuple(required) for field, required in out.items() if required}


def _closure(subset, requires, universe):
    found, todo = set(subset), list(subset)
    while todo:
        for required in requires.get(todo.pop(), ()):
            if required in universe and required not in found:
                found.add(required)
                todo.append(required)
    return frozenset(found)


def dependants(culprits: Sequence[str] | frozenset[str], requires: Mapping[str, Sequence[str]],
               universe: Sequence[str]) -> tuple[str, ...]:
    """The members of ``universe`` that require one of ``culprits``, directly or through others, in order."""
    found = set(culprits)
    changed = True
    while changed:
        changed = False
        for unit in universe:
            if unit not in found and found & set(requires.get(unit, ())):
                found.add(unit)
                changed = True
    return tuple(unit for unit in universe if unit in found and unit not in culprits)


def _split(items, n):
    size, rest = divmod(len(items), n)
    parts, start = [], 0
    for index in range(n):
        end = start + size + (1 if index < rest else 0)
        parts.append(items[start:end])
        start = end
    return [part for part in parts if part]


def ddmin(rewrites: Sequence[str], test: Callable[[frozenset[str]], bool], *,
          requires: Mapping[str, Sequence[str]], budget: Budget | None) -> frozenset[str]:
    """A 1-minimal failing subset of ``rewrites`` under the ``requires`` closure (delta debugging).

    ``test(subset)`` is True when the reduced gate fails with ``subset`` enabled; ``subset`` always
    holds what its members require (the closure), so a culprit that requires another is tested with
    it and blamed alone. The caller knows that the whole set fails and the empty set passes. Each
    call counts against ``budget`` (``None``: the caller counts); a closed subset is never tested
    twice. Raises ``DiagnosisBudgetSpent`` when the budget is spent.
    """
    universe = tuple(rewrites)
    known: dict[frozenset[str], bool] = {frozenset(universe): True, frozenset(): False}

    def fails(part):
        closed = _closure(part, requires, universe)
        if closed not in known:
            if budget is not None:
                if budget.spent:
                    raise DiagnosisBudgetSpent(f"the diagnosis budget of {budget.limit} reduced-gate runs is spent")
                budget.reduced_gate_runs += 1
            known[closed] = bool(test(closed))
        return known[closed]

    current, n = list(universe), 2
    while len(current) >= 2:
        parts = _split(current, n)
        reduced = next((part for part in parts if fails(part)), None)
        if reduced is not None:
            current, n = reduced, 2
            continue
        complement = None
        if n > 2:
            for part in parts:
                rest = [unit for unit in current if unit not in part]
                if fails(rest):
                    complement = rest
                    break
        if complement is not None:
            current, n = complement, max(n - 1, 2)
        elif n < len(current):
            n = min(2 * n, len(current))
        else:
            break
    return frozenset(current)


def diagnose(plan: Any, failure: BaseException | Failure | None, *, test: Callable[[str, frozenset[str]], bool]
             ) -> str | None:
    """The failing dimension (§9.4 step 2): ``"mode"`` if eager passes, else ``"rewrites"``.

    ``test(mode, enabled)`` is True when the reduced gate fails in ``mode`` with the switches
    ``enabled`` on. Eager on the full rewrite set passes: ``"mode"`` (a graph or compiled plan; for an
    eager plan the reduced gate cannot see the fault: ``None``). Eager fails and the empty set passes:
    ``"rewrites"``. The empty set fails too: ``None``, the fault is not in the rewrites.
    """
    units = frozenset(enabled_rewrites(plan))
    if not test("eager", units):
        return "mode" if plan.tier not in ("eager", "reference") else None
    if not units or test("eager", frozenset()):
        return None
    return "rewrites"


def terminal_sequence(plan: Any, *, reference: bool = True) -> tuple[tuple[str, str], ...]:
    """The fixed candidates after the budget is spent (§9.4 step 2), as ``(mode, label)``: the same mode
    with every rewrite off, eager with every rewrite off, the reference tier if it fits (``reference``:
    ``FallbackConfig.reference_last``). Duplicates are left out."""
    steps = [(plan.tier, f"{plan.tier} with every rewrite off"), ("eager", "eager with every rewrite off")]
    if reference:
        steps.append(("reference", "the reference tier"))
    out, seen = [], set()
    for mode, label in steps:
        if mode not in seen:
            seen.add(mode)
            out.append((mode, label))
    return tuple(out)


def _release(*objects):
    """Drop engines built for a test before the next one is built (one engine at a time)."""
    import gc

    del objects
    gc.collect()
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a dependency
        return
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


class Fallback:
    """The §9.4 search for one failure: ``Fallback(planner, force_field, atoms, integrator, references,
    config).run(failure)`` returns a ``Result`` with the activated plan revision, or raises
    ``FallbackExhausted``.

    - ``references``: the gate's (``selfcheck.references``); the reduced gate takes their first copy
      and the failing quantity. Under ``check="skip"`` they are skipped, and the search computes one
      rattled copy's references once, at the first run that needs them (``seed``, ``gate``);
    - ``failure``: a ``Failure``;
    - ``step`` and ``note``: a mid-run fallback's (5-T08): the committed point's step, at which every candidate
      revision is made, and ``mid_run_note``'s text in each candidate's reason. ``atoms`` is then the committed
      structure and ``references`` the gate's on it.

    The loop keeps (mode, rewrites off) and, per round, reacts to the last failure: rule 13 for a
    self-check failure; otherwise ``diagnose``: the mode (compiled for a graph plan, then eager), or
    ddmin over the eager rewrite set, whose minimal failing set and its dependants go off; then the
    candidate is validated (``activate``). Every round changes the mode or switches rewrites off, so
    the loop ends. When the diagnosis cannot locate the fault or its budget is spent, the terminal
    sequence runs.
    """

    def __init__(self, planner: Any, force_field: Any, atoms: Any, integrator: Any, references: Any,
                 config: Any = None, *, seed: int = 0, gate: Any = None, moves: Any = None, step: int = 0,
                 note: str | None = None):
        from fastmace.config import FallbackConfig

        self.planner = planner
        self.force_field = force_field
        self.atoms = atoms
        self.integrator = integrator
        self.references = references
        self.config = FallbackConfig() if config is None else config
        self.seed = seed
        self.gate = gate
        self.moves = moves  # an MC plan's move-energy references: every run and candidate passes them too (3-T11)
        self.step, self.note = int(step), note
        self.budget = Budget(limit=self.config.diagnosis_budget)
        self.tried: list[TriedCandidate] = []
        self.revisions: list[Any] = []
        self._cache: dict[tuple, bool] = {}
        self._bases: dict[str, Any] = {}
        self._one_copy = None

    def _at(self):
        """The step and the mid-run note of every candidate revision (``candidate``)."""
        return {"step": self.step, "note": self.note}

    # ---- references

    def _full_references(self):
        refs = self.references
        if refs is None or refs.skipped or refs.unavailable is not None:
            return None
        return refs

    def _single(self):
        """One rattled copy's references (all quantities), computed once (``check="skip"``)."""
        from fastmace.planner import selfcheck

        if self._one_copy is None:
            self._one_copy = selfcheck.references(self.failed, self.force_field, self.atoms, self.integrator,
                                                  check="gate", seed=self.seed, gate=self.gate, count=1)
        return self._one_copy

    def _reduced_references(self):
        full = self._full_references()
        if full is None:
            return dataclasses.replace(self._single(), quantities=tuple(self.quantities))
        return dataclasses.replace(full, copies=list(full.copies[:1]), values=list(full.values[:1]),
                                   reference_copies=list(full.reference_copies[:1]),
                                   quantities=tuple(self.quantities))

    def _validation_references(self):
        """The full gate's references, or under ``check="skip"`` the one-copy reduced gate (conductor answers)."""
        return self._full_references() or self._single()

    # ---- plans

    def _base(self, mode):
        """The failed plan re-planned in ``mode`` (cached); ``None`` when that mode is refused."""
        from fastmace.errors import PlanRefused, ProcessConstraintError

        if mode == self.failed.tier:
            return self.failed
        if mode not in self._bases:
            try:
                self._bases[mode] = candidate(self.failed, {"mode": mode, "reason": "diagnosis"}, self.force_field,
                                              self.atoms, planner=self.planner, **self._at())
            except (PlanRefused, ProcessConstraintError, ValueError):
                self._bases[mode] = None
        return self._bases[mode]

    def _units(self, mode):
        base = self._base(mode)
        return enabled_rewrites(base) if base is not None else ()

    def _variant(self, mode, enabled):
        base = self._base(mode)
        values = {}
        for name in enabled_rewrites(base):
            if name not in enabled:
                values[name] = False if isinstance(getattr(base.rewrites, name), bool) else "off"
        return dataclasses.replace(base, rewrites=dataclasses.replace(base.rewrites, **values)) if values else base

    # ---- runs

    def _evaluate(self, plan, references, quantities=None):
        """Build ``plan`` and gate it once: ``(passed, stage, error)``; the engine is dropped."""
        from fastmace.errors import GateFailed
        from fastmace.planner import selfcheck

        try:
            engine = self.planner.build(plan, self.force_field, self.atoms)
            _prepare(engine)
        except Exception as error:
            if _propagates(error):
                raise _with_estimate(error, plan)
            return False, _build_stage(error), detach(error)
        extra = {} if quantities is None else {"quantities": tuple(quantities)}
        try:
            record = selfcheck.production(engine, references, check=references.check, **extra)
        except GateFailed as error:
            _release(engine)
            return False, "gate", detach(error)
        except Exception as error:
            if _propagates(error):
                raise
            _release(engine)
            return False, stage_of(error, "gate"), detach(error)
        if record.passed and self.moves is not None:
            try:
                selfcheck.move_gate(engine, self.moves, plan=plan, force_field=self.force_field, record=record)
            except GateFailed as error:
                _release(engine)
                return False, "gate", detach(error)
        _release(engine)
        if not record.passed:
            return False, "gate", RuntimeError(f"gate {record.status}: {record.reason}")
        return True, None, None

    def _test(self, mode, enabled):
        """One reduced-gate run (counted): True when it fails in ``mode`` with ``enabled`` on (cached)."""
        base = self._base(mode)
        if base is None:
            return True
        units = enabled_rewrites(base)
        enabled = frozenset(enabled) & frozenset(units)
        key = (mode, enabled, tuple(self.quantities))
        if key in self._cache:
            return self._cache[key]
        if self.budget.spent:
            raise DiagnosisBudgetSpent(f"the diagnosis budget of {self.budget.limit} reduced-gate runs is spent")
        self.budget.reduced_gate_runs += 1
        plan = self._variant(mode, enabled)
        passed, stage, error = self._evaluate(plan, self._reduced_references())
        off = [name for name in units if name not in enabled]
        whole = not off
        change = f"mode {mode}" + ("" if whole else ", every rewrite off" if not enabled
                                   else f", rewrites {', '.join(off)} off")
        dimension = "mode" if whole and mode != self.failed.tier else "rewrites"
        what = "reduced gate (one rattled copy, " + ", ".join(self.quantities) + ")"
        reason = f"diagnosis: {what} " + ("passed" if passed else f"failed at {stage}: {_text(error)}")
        self.tried.append(TriedCandidate(dimension=dimension, change=change, outcome="passed" if passed else "failed",
                                         reason=reason, stage=None if passed else stage, reduced_gate_runs=1))
        self._cache[key] = not passed
        return not passed

    def _skipped(self, dimension, change, reason):
        self.tried.append(TriedCandidate(dimension=dimension, change=change, outcome="skipped", reason=reason))

    def _candidate(self, mode, off, why, rule13=None):
        """The candidate chain from the failed plan: rewrites ``off`` (rule 13's decisions for a self-check
        failure), then the mode. ``None`` (recorded) when it changes nothing or is refused."""
        from fastmace.errors import PlanRefused, ProcessConstraintError
        from fastmace.planner import reactions

        plan = self.failed
        change = f"mode {mode}" if mode != plan.tier else "rewrites"
        try:
            here = [name for name in enabled_rewrites(plan) if name in off]
            if here:
                plan = candidate(plan, {"rewrites": {name: "off" for name in here}, "reason": why},
                                 self.force_field, self.atoms, **self._at())
                if rule13 is not None:
                    plan = dataclasses.replace(plan, decisions=reactions.selfcheck_revision(
                        self.failed, rule13).decisions)
                    plan = memory_revised(plan, self.force_field, self.atoms, self.planner)
            if mode != plan.tier:
                plan = candidate(plan, {"mode": mode, "reason": why}, self.force_field, self.atoms,
                                 planner=self.planner, **self._at())
                left = [name for name in enabled_rewrites(plan) if name in off]
                if left:
                    plan = candidate(plan, {"rewrites": {name: "off" for name in left}, "reason": why},
                                     self.force_field, self.atoms, **self._at())
        except (PlanRefused, ProcessConstraintError, ValueError) as error:
            self._skipped("mode" if mode != self.failed.tier else "rewrites", change,
                          f"{why}; not planned: {_text(error)}")
            return None
        if plan is self.failed:
            self._skipped("rewrites", change, f"{why}; the candidate would be the failed plan")
            return None
        return plan

    def _validate(self, mode, off, why, rule13=None):
        """Build, self-check and gate the candidate (``activate``); the ``Activation``, a ``Result`` for a
        child, or ``None`` when there was nothing to try."""
        from fastmace.errors import ProcessConstraintError
        from fastmace.planner import selfcheck

        plan = self._candidate(mode, off, why, rule13)
        if plan is None:
            return None
        if mode == "reference":  # the last candidate only if it fits (§9.4 step 5; 2-T03)
            fits, reason = self.planner.reference_fits(plan, self.force_field, self.atoms, self._full_references())
            if not fits:
                self._skipped("mode", "mode reference", f"{why}; does not fit: {reason}")
                return None
        self.revisions.append(plan)
        try:
            activation = activate(plan, self.planner, self.force_field, self.atoms,
                                  references=self._validation_references(), budget=self.budget, moves=self.moves)
        except ProcessConstraintError:
            if plan.runner != "subprocess":
                raise
            tried = TriedCandidate(dimension=_parse(plan)[0], change=_parse(plan)[1], outcome="activated",
                                   reason=f"{why}; validated in its child process (SPEC §9.4 step 3)",
                                   revision=plan.revision)
            self.tried.append(tried)
            return Result(plan=dataclasses.replace(plan, fallbacks=tuple(self.tried)), engine=None, gate=None,
                          budget=self.budget, tried=tuple(self.tried), runner=True)
        except Exception as error:
            if mode == "reference" and selfcheck.is_out_of_memory(error):
                self._skipped("mode", "mode reference", f"{why}; does not fit: {_text(error)}")
                return None
            raise
        self.tried.append(activation.tried)
        return activation

    def _result(self, activation):
        plan = dataclasses.replace(activation.plan, fallbacks=tuple(self.tried))
        activation.engine.plan = plan
        return Result(plan=plan, engine=activation.engine, gate=activation.gate, budget=self.budget,
                      tried=tuple(self.tried))

    # ---- the search

    def run(self, failure: Failure) -> Result:
        if self.config.fallback == "refuse":
            raise failure.error
        if not isinstance(failure, Failure):
            raise TypeError(f"Fallback.run takes a fallback.Failure, got {type(failure).__name__}")
        # The failed engine must be gone before the first candidate is built (2-T15c); "refuse"
        # above raises the error with its traceback intact.
        detach(failure.error)
        self.failed = failure.plan
        self.first = failure
        self.quantities = failure.quantities()
        first = self._mc_energy_first(failure)
        if first is not None:
            return first
        mode, off, last = self.failed.tier, set(), failure
        rounds = len(_switches(self.failed.rewrites)) + 4 if self.failed.rewrites is not None else 4
        try:
            for _ in range(rounds):
                outcome = self._round(mode, off, last)
                if outcome is None:
                    break
                mode, off, result = outcome
                if isinstance(result, Result):
                    return result
                if result is None:
                    break
                if result.activated:
                    return self._result(result)
                last = Failure(plan=result.candidate, stage=result.tried.stage or "gate", error=result.error,
                               gate=result.gate)
                if last.stage == "gate" and last.gate is not None:
                    self.quantities = last.quantities()
        except DiagnosisBudgetSpent:
            pass
        return self._terminal()

    def _mc_energy_first(self, failure):
        """The MC energy precision dimension, before any other (SPEC §9.4 step 2; 3-T11): when only the
        move-energy part failed, "auto" was asked and the MC energies are not float64 yet, the candidate
        with float64 MC energies is validated first. ``None`` when that does not apply or it failed."""
        plan = self.failed
        if (self.moves is None or not move_energy_only(failure) or plan.workload is None
                or plan.workload.mc_energy_precision != "auto" or plan.precision.mc_energy == "float64"):
            return None
        why = f"{MC_ENERGY_WHY}: only the move-energy part failed ({_text(failure.error)})"
        revised = candidate(plan, {"mc_energy": "float64", "reason": why}, self.force_field, self.atoms,
                            **self._at())
        self.revisions.append(revised)
        activation = activate(revised, self.planner, self.force_field, self.atoms,
                              references=self._validation_references(), budget=self.budget, moves=self.moves)
        self.tried.append(activation.tried)
        return self._result(activation) if activation.activated else None

    def _round(self, mode, off, last):
        """One reaction to ``last``: ``(mode, off, activation)``, or ``None`` when the diagnosis is done."""
        reaction = getattr(last.error, "reaction", None) if last.stage == "self-check" else None
        if reaction is not None and not set(reaction.fields) <= off:
            off = off | set(reaction.fields)
            why = (f"{RULE_13_WHY}: rewrite {reaction.rewrite} failed its self-check ({reaction.reason}); "
                   f"{', '.join(reaction.disabled)} off")
            return mode, off, self._validate(mode, off, why, rule13=last.error)
        current = self._variant(mode, frozenset(self._units(mode)) - off) if self._base(mode) else None
        if current is None:
            return None
        dimension = diagnose(current, last, test=self._test)
        if dimension == "mode":
            # §9.4 step 2: compiled without graphs first, then eager (3-T06). A compiled candidate that is
            # refused or fails makes the next round, which tries eager.
            if mode == "graph":
                activation = self._validate("compiled", off, f"{MODE_WHY}; compiled first ({_text(last.error)})")
                if activation is not None:
                    return "compiled", off, activation
            return "eager", off, self._validate("eager", off, f"{MODE_WHY} ({_text(last.error)})")
        if dimension != "rewrites":
            return None
        # The set the diagnosis just saw failing in eager: ddmin's whole set (its precondition).
        seen = set(enabled_rewrites(current))
        units = tuple(name for name in self._units("eager") if name not in off and name in seen)
        requires = field_requires(units, self._registry())
        culprits = ddmin(units, lambda subset: self._test("eager", subset), requires=requires, budget=None)
        ordered = tuple(name for name in units if name in culprits)
        everything = tuple(dict.fromkeys((*self._units("eager"), *self._units(mode))))
        along = dependants(culprits, field_requires(everything, self._registry()), everything)
        along = tuple(name for name in along if name not in off)
        off = off | set(ordered) | set(along)
        why = f"{DDMIN_WHY} {', '.join(ordered)}" + (f" (and {', '.join(along)}, which require it)" if along else "")
        return mode, off, self._validate(mode, off, f"{why} ({_text(self.first.error)})")

    def _registry(self):
        from fastmace.rewrites import registry as reg

        return getattr(self.planner, "registry", None) or reg.default_registry()

    def _terminal(self):
        """The terminal sequence (§9.4 step 2), each step validated; ``FallbackExhausted`` when none passes."""
        spent = (f"the diagnosis budget of {self.budget.limit} reduced-gate runs is spent" if self.budget.spent
                 else "the reduced gate cannot locate the fault")
        everything = set(_switches(self.failed.rewrites)) if self.failed.rewrites is not None else set()
        for mode, label in terminal_sequence(self.failed, reference=self.config.reference_last):
            key = (mode, frozenset(), tuple(self.quantities))
            if self._cache.get(key):
                self._skipped("mode" if mode != self.failed.tier else "rewrites", f"mode {mode}, every rewrite off",
                              f"{TERMINAL_WHY}: {label}: it failed the reduced gate in the diagnosis")
                continue
            result = self._validate(mode, everything, f"{TERMINAL_WHY}: {label} ({spent}; "
                                                      f"{_text(self.first.error)})")
            if isinstance(result, Result):
                return result
            if result is not None and result.activated:
                return self._result(result)
        raise self._exhausted()

    def _exhausted(self):
        from fastmace.errors import FallbackExhausted, Reason

        error = FallbackExhausted(
            [Reason(f"no fallback candidate passed its validation (SPEC §9.4 step 5): {len(self.tried)} tried, "
                    f"{self.budget.reduced_gate_runs} reduced-gate and {self.budget.full_gate_runs} full gate runs; "
                    f"the first failure: {_text(self.first.error)}", rule=14, code=FAILED_CODE)],
            why="every candidate (diagnosis, ddmin and the terminal sequence) failed its self-checks or gate, so the "
                "planner refuses rather than run a plan it could not validate",
            revisions=(self.failed, *self.revisions))
        error.tried = tuple(self.tried)
        error.budget = self.budget
        error.failure = self.first
        return error


# --------------------------------------------------------------------------- 5-T08: mid-run fallbacks

# SPEC §9.4 step 5 and decision 54 (D4 (a)): during run(), a capture error at a re-capture and a compile error
# of the compiled tier fall back from the last committed canonical point. The candidate revisions carry the
# point's step and MID_RUN_NOTE's text; the plan's decision "fallback.mid_run" carries MID_RUN_CODE.
MID_RUN_CODE = "fallback-mid-run"
# A second trigger before the next canonical point: the run stops (decision 54, conductor's answer).
MID_RUN_LIMIT_CODE = "fallback-mid-run-limit"
MID_RUN_NOTE = "mid-run on the "
# Fallbacks per canonical interval (decision 54: one, each recorded).
MID_RUN_PER_INTERVAL = 1
# The stages of an error raised by a step that are mid-run triggers (decision 54).
MID_RUN_STAGES = ("capture", "compile")


def mid_run_stage(error: BaseException) -> str | None:
    """The trigger stage of an error a step raised during ``run()`` (decision 54): ``"capture"`` for an error the
    graph engine tagged while capturing (a re-capture), ``"compile"`` for a Dynamo or Inductor error (the
    compiled tier at a new edge count); ``None`` for anything else. Out-of-memory, non-finite values, refusals
    and process constraints are never triggers."""
    from fastmace.errors import FastMACEError

    if _propagates(error) or isinstance(error, FastMACEError):
        return None
    stage = stage_of(error, default="")
    return stage if stage in MID_RUN_STAGES else None


def mid_run_note(tier: str, point: int, failed: int, stage: str, error: BaseException, *, setup: bool) -> str:
    """The text every candidate of a mid-run fallback carries in its reason (``candidate(note=)``): the failed
    tier, the committed point it starts from, and the failure."""
    where = "the setup state" if setup else "the last restart point"
    return (f"{MID_RUN_NOTE}{tier} tier from step {int(point):,} ({where}; {stage} error at step {int(failed):,}: "
            f"{_text(error)})")


def _mid_run(entry) -> str | None:
    """The head of a fallback revision's mid-run note, ``"mid-run on the <tier> tier from step <n>"`` (a plan or a
    chain summary), or ``None``. One fallback per canonical interval: the head names it."""
    reason = str(getattr(entry, "reason", ""))
    if not reason.startswith(PREFIX):
        return None
    parts = reason.split(": ", 2)
    if len(parts) < 3 or not parts[2].startswith(MID_RUN_NOTE):
        return None
    return parts[2].split(" (", 1)[0]


def is_mid_run(plan: Any) -> bool:
    """Whether ``plan`` (a revision or a chain summary) is a candidate revision of a mid-run fallback."""
    return _mid_run(plan) is not None


def mid_run_fallbacks(plan: Any) -> list[dict[str, Any]]:
    """``counters.fallbacks``: one ``{step, from, to, reason, revision}`` per mid-run fallback in ``plan``'s chain,
    oldest first. A fallback's candidate chain (rewrites off, then the mode) counts once, as its last revision;
    ``from`` is the tier the note names, ``to`` the tier of that revision's mode change, else ``from``."""
    entries = [] if plan is None else [*plan.chain(), plan]
    out = []
    for index, entry in enumerate(entries):
        note = _mid_run(entry)
        if note is None:
            continue
        following = entries[index + 1] if index + 1 < len(entries) else None
        if following is not None and _mid_run(following) == note:
            continue  # the next revision of the same fallback's candidate chain
        from_tier = note[len(MID_RUN_NOTE):].split(" ", 1)[0]
        to_tier = from_tier
        for revision in entries[:index + 1]:
            if _mid_run(revision) == note:
                change = str(revision.reason)[len(PREFIX):].split(": ", 1)[0]
                if change.startswith("mode "):
                    to_tier = change[len("mode "):].split(" ", 1)[0]
        out.append({"step": int(entry.step or 0), "from": from_tier, "to": to_tier, "reason": str(entry.reason),
                    "revision": int(entry.revision)})
    return out


def mid_run_limit(error: BaseException, point: int, failed: int, previous: Mapping[str, Any]) -> BaseException:
    """The error that stops a run at a second trigger before the next canonical point (decision 54):
    ``FallbackExhausted`` with rule 14's ``MID_RUN_LIMIT_CODE``, ``previous`` the interval's fallback record,
    raised from ``error``."""
    from fastmace.errors import FallbackExhausted, Reason

    stopped = FallbackExhausted(
        [Reason(f"a second mid-run failure before the next canonical point ({stage_of(error, 'step')} error at step "
                f"{int(failed):,}: {_text(error)}), after the fallback to plan revision {previous['revision']} from "
                f"step {int(point):,}: at most {MID_RUN_PER_INTERVAL} fallback per canonical interval (SPEC §9.4 "
                "step 5, decision 54)", rule=14, code=MID_RUN_LIMIT_CODE)],
        why="the fallback's validated revision failed again before the run reached a new committed point, so the "
            "planner stops rather than search again from the same state",
        next_step="Resume from the last restart point (Simulation.resume); the record names both failures.",
        revisions=())
    stopped.failure = error
    stopped.previous = dict(previous)
    return stopped


def mid_run_revision(plan: Any) -> Any:
    """The activated revision of a mid-run fallback with the decision ``fallback.mid_run`` (rule 14,
    ``MID_RUN_CODE``): ``"<from> -> <to>"`` and the revision's reason. An earlier interval's decision is replaced."""
    from fastmace.errors import Reason

    (record,) = mid_run_fallbacks(plan)[-1:]
    reason = Reason(f"{record['reason'][len(PREFIX):]} (SPEC §9.4 step 5, decision 54)", rule=14, code=MID_RUN_CODE)
    decisions = tuple(entry for entry in plan.decisions if entry[0] != "fallback.mid_run")
    return dataclasses.replace(plan, decisions=(*decisions, ("fallback.mid_run", f"{record['from']} -> {record['to']}",
                                                             reason)))


def mid_run_runner(candidate_plan: Any, error: BaseException) -> BaseException:
    """The error for a mid-run fallback whose validated candidate runs in a child (§9.4 step 3): not handed over
    to the subprocess runner during a run yet (a 5-T08 follow-up), so the run stops with the record."""
    from fastmace.errors import FallbackExhausted, Reason

    stopped = FallbackExhausted(
        [Reason(f"the mid-run fallback's candidate (plan revision {candidate_plan.revision}, {candidate_plan.reason}) "
                "runs in a child process, which a run cannot hand over to mid-run yet (SPEC §8.7, §9.4 step 3)",
                rule=14, code=MID_RUN_LIMIT_CODE)],
        why=f"the failure ({_text(error)}) needs a backend that this process cannot compile next to its own",
        next_step="Resume from the last restart point (Simulation.resume), which plans in a fresh process.",
        revisions=(candidate_plan,))
    stopped.failure = error
    return stopped
