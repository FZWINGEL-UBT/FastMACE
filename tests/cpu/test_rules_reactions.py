"""Conformance of the reactions 13-15 of SPEC §9.3 (plan task 5-T02, phase 5; conductor's choice 1).

- Rule 13: a failed self-check disables that rewrite and every rewrite requiring it, with rule 13's codes, as
  a new plan revision (``reactions.selfcheck_revision``).
- Rule 13 before rule 14: a candidate whose self-check fails is never gated (the self-checks run in the
  build, before the gate; §9.4 step 4).
- Rule 14: the fallback's candidates carry rule 14's codes, and a failing gate never activates a candidate.
- Rule 15 is plan task 5-T04's (``planner/warnings.py``); it adds its test with its decisions.

Plan-level stubs only (no model, no GPU): faults are injected here, never through a hook in the library.
"""

import dataclasses
from types import SimpleNamespace

import pytest
import test_fallback_candidates as tfc

from _support.rule_cases import assert_conforms
from _support.structures import nacl
from fastmace.errors import GateFailed, SelfCheckError
from fastmace.planner import fallback, reactions, rules, selfcheck


def gpu_plan():
    return tfc.gpu_plan()


def reaction(plan, failed, dependants=()):
    """A registry ``Reaction`` for the rewrite field ``failed`` and the fields requiring it."""
    fields = (failed, *dependants)
    return SimpleNamespace(rewrite=failed.upper(), disabled=tuple(name.upper() for name in fields), fields=fields,
                           reason="2e-3 > 1e-5")


def references():
    return selfcheck.References(check="gate", copies=[object()], ran="in-process")


def gate_spy(monkeypatch, error=None):
    calls = []

    def production(engine, refs, *, check, **kwargs):
        calls.append(engine)
        if error is not None:
            raise error
        return selfcheck.GateRecord(status="passed", check=check, reason="stub gate passed")

    monkeypatch.setattr(selfcheck, "production", production)
    return calls


class StubPlanner:
    """``planner.build`` for ``activate``: raises ``error`` or returns a bare engine."""

    def __init__(self, error=None):
        self.error = error
        self.built = []

    def build(self, plan, force_field, atoms, **kwargs):
        self.built.append(plan)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(plan=None)


# --------------------------------------------------------------------------- rule 13


@pytest.mark.rule(13)
def test_rule_13_disables_the_rewrite_and_its_dependants():
    _, plan = gpu_plan()
    failed, dependant = tfc.enabled_fields(plan)[:2]
    revised = reactions.selfcheck_revision(plan, reaction(plan, failed, (dependant,)))
    assert revised.revision == plan.revision + 1 and revised.predecessor is plan and revised.step == 0
    assert revised.reason.startswith("rule 13:")
    for field, code in ((failed, reactions.FAILED_CODE), (dependant, reactions.DEPENDANT_CODE)):
        assert getattr(revised.rewrites, field) in ("off", False)
        reason = revised.decision(f"RewriteConfig.{field}")[2]
        assert (reason.rule, reason.code) == (13, code) and rules.kind(13) == "reaction"
    for field in tfc.enabled_fields(plan)[2:]:
        assert getattr(revised.rewrites, field) == getattr(plan.rewrites, field)
    assert_conforms(revised)


@pytest.mark.rule(13)
@pytest.mark.rule(14)
@pytest.mark.precedence(13, 14)
def test_rule_13_runs_before_any_gate(monkeypatch):
    """A self-check failure stops the candidate in its build: the gate (rule 14's check) never runs on it."""
    _, plan = gpu_plan()
    cand = fallback.candidate(plan, {"rewrites": {tfc.enabled_fields(plan)[0]: "off"}, "reason": "ddmin"},
                              tfc.StubForceField(), nacl())
    calls = gate_spy(monkeypatch)
    budget = fallback.Budget(limit=64)
    planner = StubPlanner(SelfCheckError("self-check of FOLD_RADIAL_CUTOFF failed: 3e-3 > 1e-5"))
    activation = fallback.activate(cand, planner, tfc.StubForceField(), nacl(), references=references(),
                                   budget=budget)
    assert planner.built == [cand] and calls == []
    assert not activation.activated and activation.tried.stage == "self-check"
    assert (budget.reduced_gate_runs, budget.full_gate_runs) == (0, 0)


# --------------------------------------------------------------------------- rule 14


@pytest.mark.rule(14)
def test_rule_14_candidates_carry_its_codes():
    planner, plan = gpu_plan()
    field = tfc.enabled_fields(plan)[0]
    cand = fallback.candidate(plan, {"rewrites": {field: "off"}, "reason": "ddmin culprit"}, tfc.StubForceField(),
                              nacl())
    reason = cand.decision(f"RewriteConfig.{field}")[2]
    assert (reason.rule, reason.code) == (14, fallback.REWRITE_OFF_CODE)
    assert_conforms(cand)
    back = fallback.candidate(cand, {"rewrites": {field: "on"}, "reason": "ddmin: innocent"}, tfc.StubForceField(),
                              nacl())
    assert (back.decision(f"RewriteConfig.{field}")[2].rule, back.decision(f"RewriteConfig.{field}")[2].code) == (
        14, fallback.REWRITE_ON_CODE)
    child = fallback.candidate(plan, {"backend": "cueq", "runner": "subprocess", "reason": "hybrid capture fails"},
                               tfc.StubForceField(), nacl(), planner=planner)
    runner = child.decision("runner")[2]
    assert runner.code in rules.codes(14) + ("runner-subprocess-family",)
    assert_conforms(child)


@pytest.mark.rule(14)
def test_rule_14_never_runs_with_a_failing_gate(monkeypatch):
    _, plan = gpu_plan()
    cand = fallback.candidate(plan, {"rewrites": {tfc.enabled_fields(plan)[0]: "off"}, "reason": "ddmin"},
                              tfc.StubForceField(), nacl())
    record = selfcheck.GateRecord(status="failed", check="gate", reason="forces 3x the yardstick")
    calls = gate_spy(monkeypatch, error=GateFailed("the gate failed", gate_record=record))
    activation = fallback.activate(cand, StubPlanner(), tfc.StubForceField(), nacl(), references=references())
    assert len(calls) == 1 and not activation.activated and activation.engine is None
    assert activation.tried.stage == "gate" and activation.plan.revision == plan.revision
    assert activation.plan.fallbacks == (activation.tried,)


@pytest.mark.rule(14)
def test_rule_14_activates_only_after_the_gate_passed(monkeypatch):
    _, plan = gpu_plan()
    cand = fallback.candidate(plan, {"rewrites": {tfc.enabled_fields(plan)[0]: "off"}, "reason": "ddmin"},
                              tfc.StubForceField(), nacl())
    calls = gate_spy(monkeypatch)
    activation = fallback.activate(cand, StubPlanner(), tfc.StubForceField(), nacl(), references=references())
    assert len(calls) == 1 and activation.activated and activation.gate.passed
    assert activation.plan.revision == cand.revision
    assert dataclasses.replace(activation.plan, fallbacks=()).rewrites == cand.rewrites


def test_reactions_take_no_override():
    for number in rules.numbers("reaction"):
        with pytest.raises(ValueError, match="only preferences"):
            rules.override(number, "x")
