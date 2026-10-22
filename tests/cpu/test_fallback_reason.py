"""A fallback's reason names the innermost error of a chained compile error (plan task 3-T17).

2-T21d's case: Inductor's compile workers died (a script named ``bisect.py`` shadowed the standard
library), and the plan fell back to eager with a reason that stopped at
``BackendCompilerFailed: backend='inductor' raised:``, dropping the cause
``RuntimeError: Attempting to use a closed pool``.

- ``fallback.error_text`` keeps the outer type and adds the innermost error of the
  ``__cause__``/``__context__`` chain (or, with no chain, the error line under a trailing
  ``raised:``), on one line of at most about 300 characters;
- an error without a chain gives the same text as before;
- ``minimal._error_text`` (trial evaluation) uses the same helper;
- ``Fallback.run`` puts it into ``plan.reason`` and the report's plan line.
"""

from __future__ import annotations

import pytest
import test_fallback_search as tfs
import test_minimal_planner as tmp

from fastmace import report
from fastmace.errors import FastMACEError, GateFailed
from fastmace.planner import fallback, minimal, selfcheck

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

pristine = tfs.pristine
quiet_licences = tmp.quiet_licences

CLOSED_POOL = "Attempting to use a closed pool"


class BackendCompilerFailed(Exception):
    """Stands in for ``torch._dynamo.exc.BackendCompilerFailed``: its message, raised ``from`` the cause."""

    def __init__(self, backend, inner):
        super().__init__(f"backend={backend!r} raised:\n{type(inner).__name__}: {inner}\n\n"
                         "Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially "
                         "if you're reporting a bug to PyTorch).")
        self.inner_exception = inner


def inductor_failure():
    """The 2-T21d chain: the pool's RuntimeError, wrapped by BackendCompilerFailed."""
    try:
        try:
            raise RuntimeError(CLOSED_POOL)
        except RuntimeError as inner:
            raise BackendCompilerFailed("inductor", inner) from inner
    except BackendCompilerFailed as error:
        return error


def old_text(error):
    """``_text`` before 3-T17: the class and the first line."""
    first = str(error).splitlines()[0] if str(error) else ""
    return f"{type(error).__name__}: {first}" if first else type(error).__name__


# --------------------------------------------------------------------------- the helper


def test_the_2_t21d_chain_names_the_closed_pool():
    text = fallback.error_text(inductor_failure())
    assert text == f"BackendCompilerFailed: backend='inductor' raised: RuntimeError: {CLOSED_POOL}"
    assert "\n" not in text


def test_the_innermost_of_a_deeper_chain():
    try:
        try:
            try:
                raise OSError("SubprocPool unclean exit")
            except OSError as deepest:
                raise RuntimeError(CLOSED_POOL) from deepest
        except RuntimeError as middle:
            raise ValueError("compile failed") from middle
    except ValueError as error:
        outer = error
    assert fallback.error_text(outer) == "ValueError: compile failed; innermost OSError: SubprocPool unclean exit"


def test_an_implicit_context_counts_and_a_suppressed_one_does_not():
    try:
        try:
            raise RuntimeError(CLOSED_POOL)
        except RuntimeError:
            raise KeyError("lookup during handling")  # noqa: B904 - the implicit context is the point
    except KeyError as error:
        implicit = error
    assert fallback.error_text(implicit).endswith(f"; innermost RuntimeError: {CLOSED_POOL}")
    try:
        try:
            raise RuntimeError(CLOSED_POOL)
        except RuntimeError:
            raise KeyError("lookup during handling") from None
    except KeyError as error:
        suppressed = error
    assert fallback.error_text(suppressed) == old_text(suppressed)


def test_a_cycle_ends():
    first, second = RuntimeError("first"), RuntimeError("second")
    first.__cause__, second.__cause__ = second, first
    assert fallback.error_text(first) == "RuntimeError: first; innermost RuntimeError: second"


def test_without_a_chain_the_text_is_unchanged():
    for error in (RuntimeError("CUDA graph capture of stage B failed"), RuntimeError(), ValueError("a\nb"),
                  FastMACEError("what happened", why="because")):
        assert fallback.error_text(error) == old_text(error)


def test_without_a_chain_the_line_under_raised_is_the_cause():
    """A wrapped error whose chain was lost (pickled from a child, say) still names the cause it printed."""
    error = BackendCompilerFailed("inductor", RuntimeError(CLOSED_POOL))
    assert error.__cause__ is None and error.__context__ is None
    expected = f"BackendCompilerFailed: backend='inductor' raised: RuntimeError: {CLOSED_POOL}"
    assert fallback.error_text(error) == expected


def test_a_cause_already_in_the_message_is_not_repeated():
    try:
        try:
            raise RuntimeError(CLOSED_POOL)
        except RuntimeError as inner:
            raise ValueError(f"the build failed: {inner}") from inner
    except ValueError as error:
        assert fallback.error_text(error) == old_text(error)


def test_the_text_is_one_line_of_about_300_characters():
    try:
        try:
            raise RuntimeError("x" * 2000 + "\nsecond line")
        except RuntimeError as inner:
            raise BackendCompilerFailed("inductor", inner) from inner
    except BackendCompilerFailed as error:
        text = fallback.error_text(error)
    assert len(text) <= 300 and "\n" not in text
    assert text.startswith("BackendCompilerFailed: backend='inductor' raised: RuntimeError: xxx")


def test_the_reasons_text_keeps_the_gate_record():
    record = selfcheck.GateRecord(status="failed", check="gate", reason="force error above the limit")
    try:
        try:
            raise RuntimeError(CLOSED_POOL)
        except RuntimeError as inner:
            raise GateFailed("the gate on this structure failed", gate_record=record) from inner
    except GateFailed as error:
        text = fallback._text(error)
    assert text == (f"GateFailed: the gate on this structure failed; innermost RuntimeError: {CLOSED_POOL} "
                    "(force error above the limit)")


def test_the_trial_evaluations_text_uses_the_helper():
    assert minimal._error_text(inductor_failure()) == fallback.error_text(inductor_failure())
    error = FastMACEError("what happened", why="because")
    assert minimal._error_text(error) == "FastMACEError: what happened (because)"


# --------------------------------------------------------------------------- the plan and the report


def test_the_eager_fallback_reason_and_report_line_name_the_closed_pool(pristine, monkeypatch):
    """2-T21d end to end: a compiled plan fails to compile; eager passes; the reason carries the cause."""
    planner, ff, atoms, _ = tfs.setup_case(pristine)
    plan = planner.plan(ff, atoms, tfs.INTEGRATOR, tmp.WORKLOAD, tfs.minimal_hardware(), mode="compiled")
    tfs.Faults(monkeypatch, bad_modes={"compiled"})
    failure = fallback.Failure(plan=plan, stage="compile", error=inductor_failure())
    _, result = tfs.run_fallback(planner, ff, atoms, failure)
    active = result.plan
    expected = f"BackendCompilerFailed: backend='inductor' raised: RuntimeError: {CLOSED_POOL}"
    assert active.tier == "eager" and active.reason.startswith("fallback: mode eager: ")
    assert expected in active.reason and fallback.MODE_WHY in active.reason
    lines = report.revision_lines(active)
    assert len(lines) == 1 and lines[0].startswith("plan revision 1 (fallback): mode eager, reason ")
    assert expected in lines[0]
