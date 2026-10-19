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

This module imports neither torch nor MACE at import time.
"""

from __future__ import annotations

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


def candidate(plan: Any, changes: Mapping[str, Any], force_field: Any, atoms: Any) -> Any:
    """A plan revision of ``plan`` with one dimension changed (``changes``: mode, backend or the
    rewrite set), built from the pristine ``force_field`` (decision 24). Not active until
    ``activate`` validated it."""
    raise NotImplementedError(f"fallback candidates are {TASK_CANDIDATES}")


def activate(candidate: Any, *args: Any, **kwargs: Any) -> Any:
    """Run the self-checks and the gate on ``candidate`` in the mode it will run in, and only then
    make it active; a failing candidate is recorded and never activated (§9.4 step 4)."""
    raise NotImplementedError(f"validated activation is {TASK_CANDIDATES}")


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
