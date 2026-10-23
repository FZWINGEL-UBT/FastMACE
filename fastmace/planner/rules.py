"""The decision rules of SPEC §9.3 as data (plan task 5-T01, conductor's choice 1; docs/design/planner-5.md).

Each ``Rule`` has its number (the strict order of §9.3), its kind (rules 1-7 hard, 8-12 preferences, 13-15
reactions), a short title, the stable codes its reasons and decisions use, and the ``allow_*`` flag that lifts
it (hard rules 1 and 6 only). Every ``fastmace.errors.Reason`` the planner emits for a rule carries
``rule=n`` and a code from ``codes(n)``. The planner does not check that at run time: the conformance suite
does (``check_reason``; ``tests/cpu/test_rules_meta.py`` scans the code and the tests).

An override of a preference (rules 8-12) is a decision whose reason is ``override(n, would_give)``: it reads
``choice (override; rule n would give X)`` in the plan, and ``Plan.requested`` keeps the value the caller set
(conductor's choice 4). Hard rules have no override: a request against one is refused, never downgraded.

This module holds no state and imports nothing from torch or MACE.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

Kind = Literal["hard", "preference", "reaction"]
KINDS = ("hard", "preference", "reaction")
# The code of every override decision (choice 4); valid for preferences only.
OVERRIDE_CODE = "override"
# The text of an override reason; ``overridden`` reads it back.
_OVERRIDE = re.compile(r"^override; rule (?P<rule>\d+) would give (?P<choice>.*?)(?:: (?P<detail>.*))?$", re.DOTALL)
_CODE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class Rule:
    """One row of SPEC §9.3: ``number`` (its order), ``kind``, ``title``, its ``codes`` and its ``allow`` flag."""

    number: int
    kind: Kind
    title: str
    codes: tuple[str, ...]
    allow: str | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"rule {self.number}: kind {self.kind!r} is not one of {KINDS}")
        if self.allow is not None and self.kind != "hard":
            raise ValueError(f"rule {self.number}: only a hard rule has an allow_* flag")
        bad = [code for code in self.codes if not _CODE.match(code) or code == OVERRIDE_CODE]
        if bad:
            raise ValueError(f"rule {self.number}: codes {bad} are not kebab-case or are reserved")


# §9.3's table. A code belongs to one rule only. Codes a later phase 5 task adds are marked with it; codes
# still emitted without their rule number are listed in tests/cpu/test_rules_meta.py until 5-T02 numbers them.
RULES: tuple[Rule, ...] = (
    Rule(1, "hard", "workload outside §4.4, or a variable cell without a barostat", (
        "non-3d-pbc", "constraints", "mc-with-constraints", "variable-cell-without-barostat",
        "fix-atoms-variable-cell", "fix-atoms-with-fixcm", "scheme-unknown", "scheme-ensemble", "scheme-cell-mode",
        "scheme-dof", "scheme-cell-filter", "polar-variable-cell-optimizer", "polar-unqualified-source",
        "allow-reference", "no-allow-reference"), allow="allow_reference"),
    Rule(2, "hard", "a species outside the ForceField's superset, in the structure or a declared move set",
         ("species-not-in-superset",)),
    Rule(3, "hard", "PolarMACE with a plain compiled backend", ("polar-plain-compiled-cueq",)),
    Rule(4, "hard", "inexact padding: no graph mode, compiled with exact edge counts", (
        "inexact-padding", "mode-compiled-rule-4", "compiled-exact-edges", "eager-exact-edges")),
    Rule(5, "hard", "reproducible=True: a qualified backend and model family only", (
        "reproducible-tier", "reproducible-family-not-qualified", "reproducible-backend", "reproducible-conv-fusion",
        "reproducible-cueq-version", "reproducible-kernel-not-deterministic", "reproducible-qualified",
        "reproducible-environment", "backend-cueq-reproducible")),
    Rule(6, "hard", "cuEquivariance's fused kernels do not load", ("naive-cueq", "fused-cueq", "allow-naive-cueq"),
         allow="allow_naive_cueq"),
    Rule(7, "hard", "atom-count changes: not in 1.0", ("atom-count-changes",)),
    Rule(8, "preference", "species changes keep graph mode; superset build", ("superset", "species-changes")),
    Rule(9, "preference", "host callbacks keep the graph tier (decision 38)", (
        "host-callbacks-graph-sync", "host-callbacks-compiled", "no-host-callbacks")),
    Rule(10, "preference", "stress path and k-headroom from the integrator or optimiser", ("rule-10-optimizer",)),
    Rule(11, "preference", "skin, headroom and re-capture budget; D3 and displacement skins", (
        "rule-11", "rule-11-fixed", "rule-11-md-only", "rule-11-optimizer", "rule-11-d3")),
    # 5-T03 adds the MACE model's estimate ("rule-12-memory") and the action's codes once D1 is decided.
    Rule(12, "preference", "GPU memory: drop memory-heavy rewrites or warn", (
        "rule-12-d3", "rule-12-d3-memory", "rule-12-d3-warn", "rule-12-memory")),
    Rule(13, "reaction", "a rewrite self-check fails: disable it and its dependants", (
        "selfcheck-failed", "selfcheck-dependant")),
    # 5-T08 adds the mid-run fallback's codes once D4 is decided.
    Rule(14, "reaction", "the gate fails: the fallback of §9.4", (
        "fallback-exhausted", "fallback-rewrite-off", "fallback-rewrite-on", "runner-subprocess-fallback")),
    # 5-T04 emits it (planner/warnings.py, the decisions "warn.model_structure...").
    Rule(15, "reaction", "known model-structure warning: warn, never refuse", ("model-structure-warning",)),
)
NUMBERS = tuple(rule.number for rule in RULES)
_BY_NUMBER = {rule.number: rule for rule in RULES}
_BY_CODE = {code: rule.number for rule in RULES for code in rule.codes}


def rule(number: int) -> Rule:
    """Rule ``number`` of §9.3; ``KeyError`` outside 1-15."""
    if isinstance(number, bool) or number not in _BY_NUMBER:
        raise KeyError(f"§9.3 has no rule {number!r}; its rules are {NUMBERS[0]}-{NUMBERS[-1]}")
    return _BY_NUMBER[number]


def codes(number: int) -> tuple[str, ...]:
    """The codes rule ``number`` uses (``OVERRIDE_CODE`` is valid for every preference besides these)."""
    return rule(number).codes


def kind(number: int) -> Kind:
    """``"hard"``, ``"preference"`` or ``"reaction"``."""
    return rule(number).kind


def allow_flag(number: int) -> str | None:
    """The ``allow_*`` flag that lifts hard rule ``number``, or ``None``."""
    return rule(number).allow


def rule_of(code: str) -> int | None:
    """The rule a code belongs to; ``None`` for a code of no rule (an option request, a process constraint)."""
    return _BY_CODE.get(code)


def numbers(kind_: Kind) -> tuple[int, ...]:
    """The rules of one kind, in order."""
    if kind_ not in KINDS:
        raise ValueError(f"kind {kind_!r} is not one of {KINDS}")
    return tuple(entry.number for entry in RULES if entry.kind == kind_)


def check_reason(reason: Any) -> tuple[str, ...]:
    """What is wrong with ``reason`` against the table: ``()`` when nothing is.

    A ``Reason``'s ``rule`` is ``None`` or one of 1-15; with a rule its code is one of that rule's codes (or
    ``OVERRIDE_CODE`` for a preference); a rule's code never appears without its number or with another one.
    A plain string has nothing to check. Tests call this; the planner does not (plan task 5-T01).
    """
    number, code = getattr(reason, "rule", None), getattr(reason, "code", None)
    if code is None and number is None:
        return ()
    problems = []
    if number is not None:
        if isinstance(number, bool) or number not in _BY_NUMBER:
            return (f"rule {number!r} is not a §9.3 rule (1-15)",)
        if code == OVERRIDE_CODE:
            if kind(number) != "preference":
                problems.append(f"rule {number} is {kind(number)}: only preferences (8-12) are overridden")
            elif not overridden(reason):
                problems.append(f"an override reason must read 'override; rule {number} would give X': {reason!s}")
            elif overridden(reason)[0] != number:
                problems.append(f"the override text names rule {overridden(reason)[0]}, its rule is {number}")
        elif code not in codes(number):
            problems.append(f"code {code!r} is not one of rule {number}'s codes {codes(number)}")
    owner = rule_of(code) if code is not None else None
    if owner is not None and owner != number:
        problems.append(f"code {code!r} is rule {owner}'s, but the reason has rule={number!r}")
    if number is None and code == OVERRIDE_CODE:
        problems.append("an override reason needs the rule it overrides")
    return tuple(problems)


def override(number: int, would_give: Any, detail: str | None = None):
    """The reason of a decision the caller replaced (choice 4): ``override; rule n would give X[: detail]``.

    Only preferences (rules 8-12) take overrides; a hard rule raises ``ValueError`` (§9.3: refused, never
    downgraded). The decision's choice is the caller's value, and ``Plan.requested`` keeps it.
    """
    from fastmace.errors import Reason

    if kind(number) != "preference":
        raise ValueError(f"rule {number} is {kind(number)}: only preferences (rules 8-12) are overridden (SPEC §9.3)")
    text = f"override; rule {number} would give {would_give}" + (f": {detail}" if detail else "")
    return Reason(text, rule=number, code=OVERRIDE_CODE)


def overridden(reason: Any) -> tuple[int, str] | None:
    """``(rule, what the rule would give)`` of an ``override`` reason, else ``None``."""
    if getattr(reason, "code", None) != OVERRIDE_CODE:
        return None
    match = _OVERRIDE.match(str(reason))
    return None if match is None else (int(match["rule"]), match["choice"])


__all__ = ["KINDS", "NUMBERS", "OVERRIDE_CODE", "RULES", "Rule", "allow_flag", "check_reason", "codes", "kind",
           "numbers", "override", "overridden", "rule", "rule_of"]
