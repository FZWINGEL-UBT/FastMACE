"""Shared checks of the rule conformance suite (plan task 5-T02; SPEC §9.3, ``tests/cpu/test_rules_*.py``).

``assert_conforms`` runs ``planner.rules.check_reason`` on every reason of a plan or a refusal, so the suite
checks at run time what the static scan of ``test_rules_meta.py`` cannot read (a ``Reason`` built elsewhere
and passed through ``_Book.refuse``).
"""

from __future__ import annotations

from typing import Any

import pytest

from fastmace.errors import PlanRefused
from fastmace.planner import rules


def reasons(subject: Any) -> list:
    """The reasons of a ``Plan`` (its decisions) or of a ``PlanRefused``."""
    if isinstance(subject, PlanRefused):
        return list(subject.reasons)
    return [reason for _, _, reason in subject.decisions]


def assert_conforms(subject: Any) -> None:
    """Every reason of ``subject`` uses a code of its rule, and no rule's code comes without its number."""
    wrong = [(str(reason), problem) for reason in reasons(subject) for problem in rules.check_reason(reason)]
    assert not wrong, wrong


def codes(error: PlanRefused) -> list[str]:
    return [reason.code for reason in error.reasons]


def refused(call, *args, **kwargs) -> PlanRefused:
    """The ``PlanRefused`` of ``call``, checked against the rule table."""
    with pytest.raises(PlanRefused) as caught:
        call(*args, **kwargs)
    assert_conforms(caught.value)
    return caught.value


def by_rule(error: PlanRefused) -> dict[int | None, list[str]]:
    """``{rule: [codes]}`` of a refusal."""
    out: dict[int | None, list[str]] = {}
    for reason in error.reasons:
        out.setdefault(reason.rule, []).append(reason.code)
    return out
