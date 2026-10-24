"""The rule table of SPEC §9.3 against the code and the tests (plan task 5-T01, conductor's choice 1).

- The table (``fastmace.planner.rules``) has §9.3's fifteen rules in order, with their kinds and ``allow_*`` flags,
  and every code belongs to one rule.
- ``rules.check_reason`` and the override record (choice 4) behave as docs/design/planner-5.md says.
- Every reason the library emits with a rule number uses a code of that rule (a static scan of ``fastmace/``).
- A rule's code never appears without its number (``NUMBERLESS`` lists the exceptions; 5-T02 emptied it).
- Every rule has at least one test marked ``pytest.mark.rule(n)`` (``UNTESTED`` lists the exceptions; 5-T04
  emptied it with rule 15's tests).
- Every precedence pair of 5-T02's matrix (``PAIRS``) has a test marked ``pytest.mark.precedence(a, b)``, and
  every marked pair is listed. Later tasks add their pairs here with their tests (5-T03: rule 5 against the
  MACE model's rule 12; 5-T07: rule 5 against runtime adaptation).
"""

import ast
from pathlib import Path

import pytest

from fastmace.errors import Reason
from fastmace.planner import rules

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "fastmace"
TESTS = REPO / "tests"
# Helpers that wrap Reason with a fixed rule: (module, function) -> (rule, index of the code argument).
HELPERS = {("integrators/registry.py", "_reason"): (1, 1), ("planner/reproducible.py", "block"): (5, 2)}
# Rules whose codes the library still emits without the number (5-T02 numbered them all).
NUMBERLESS = set()
# Rules without a test marked rule(n) yet (5-T04 marked rule 15: tests/cpu/test_rules_15.py).
UNTESTED = set()
# The precedence matrix of plan task 5-T02 (SPEC §9.3's strict order): (a, b) means rule a wins over b, where b is
# a later rule, "override" (a user's explicit value) or an allow_* flag. tests/cpu/test_rules_precedence.py.
PAIRS = {
    (4, 8), (4, 9), (4, "override"),
    (3, "override"),
    (5, 9), (5, 11), (5, 12), (5, "override"),
    (1, "allow_reference"),
    (11, 12),
    (13, 14),
}


# --------------------------------------------------------------------------- the static scan


def _constants():
    """{module path: {NAME: "code"}} for the module-level string constants named ``*_CODE``."""
    out = {}
    for path in SOURCE.rglob("*.py"):
        names = out.setdefault(path.relative_to(SOURCE).as_posix(), {})
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id.endswith("_CODE") and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)):
                names[node.targets[0].id] = node.value.value
    return out


def _codes(node, module, constants):
    """The literal codes ``node`` can be: a string, both branches of a conditional, a ``*_CODE`` constant."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.IfExp):
        return _codes(node.body, module, constants) + _codes(node.orelse, module, constants)
    if isinstance(node, ast.Name):
        value = constants.get(module, {}).get(node.id)
        return [] if value is None else [value]
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):  # module.NAME_CODE, by module name
        values = {names[node.attr] for path, names in constants.items()
                  if Path(path).stem == node.value.id and node.attr in names}
        return sorted(values) if len(values) == 1 else []
    return []


def _rule(node):
    """A literal rule number, ``None`` for no rule, or ``"?"`` where the scan cannot tell."""
    if node is None or (isinstance(node, ast.Constant) and node.value is None):
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return "?"


def emissions():
    """``(where, rule, code)`` of every reason in ``fastmace/`` whose code the scan can read."""
    constants = _constants()
    out = []
    for path in sorted(SOURCE.rglob("*.py")):
        relative = path.relative_to(SOURCE).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            if name == "Reason":
                rule, code = _rule(keywords.get("rule")), keywords.get("code")
            elif name == "refuse":
                rule = _rule(node.args[2] if len(node.args) > 2 else keywords.get("rule"))
                code = node.args[1] if len(node.args) > 1 else keywords.get("code")
            elif (relative, name) in HELPERS:
                rule, index = HELPERS[relative, name]
                code = node.args[index] if len(node.args) > index else None
            else:
                continue
            if code is None or rule == "?":
                continue
            for value in _codes(code, relative, constants):
                out.append((f"{relative}:{node.lineno}", rule, value))
    return out


def marked_rules():
    """The rule numbers of every ``pytest.mark.rule(n)`` in ``tests/``."""
    found = set()
    for path in TESTS.rglob("test_*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "rule"
                    and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "mark"
                    and node.args and isinstance(node.args[0], ast.Constant)):
                found.add(node.args[0].value)
    return found


def marked_pairs():
    """The ``(a, b)`` of every ``pytest.mark.precedence(a, b)`` in ``tests/``."""
    found = set()
    for path in TESTS.rglob("test_*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "precedence"
                    and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "mark"):
                assert len(node.args) == 2 and all(isinstance(arg, ast.Constant) for arg in node.args), (
                    f"{path.name}:{node.lineno}: precedence(a, b) takes two literals")
                found.add((node.args[0].value, node.args[1].value))
    return found


# --------------------------------------------------------------------------- the table


def test_the_table_is_section_9_3():
    assert rules.NUMBERS == tuple(range(1, 16))
    assert rules.numbers("hard") == tuple(range(1, 8))
    assert rules.numbers("preference") == tuple(range(8, 13))
    assert rules.numbers("reaction") == (13, 14, 15)
    assert {n: rules.allow_flag(n) for n in rules.NUMBERS if rules.allow_flag(n)} == {
        1: "allow_reference", 6: "allow_naive_cueq"}
    every = [code for n in rules.NUMBERS for code in rules.codes(n)]
    assert len(every) == len(set(every)), "a code belongs to one rule only"
    assert all(rules.codes(n) for n in rules.NUMBERS)
    assert rules.rule_of("species-not-in-superset") == 2 and rules.rule_of("hybrid-unavailable") is None
    with pytest.raises(KeyError, match="no rule 16"):
        rules.rule(16)
    with pytest.raises(KeyError):
        rules.rule(True)


def test_check_reason():
    assert rules.check_reason("a plain string") == ()
    assert rules.check_reason(Reason("x", code="hybrid-unavailable")) == ()  # an option request: no rule
    assert rules.check_reason(Reason("x", rule=4, code="inexact-padding")) == ()
    assert "not a §9.3 rule" in rules.check_reason(Reason("x", rule=16, code="inexact-padding"))[0]
    problems = rules.check_reason(Reason("x", rule=4, code="naive-cueq"))
    assert any("not one of rule 4's codes" in p for p in problems) and any("rule 6's" in p for p in problems)
    assert "rule 2's" in rules.check_reason(Reason("x", code="species-not-in-superset"))[0]  # number missing


def test_override_reason_reads_back():
    reason = rules.override(11, "1.0 A", detail="NeighbourConfig.skin=0.5")
    assert str(reason) == "override; rule 11 would give 1.0 A: NeighbourConfig.skin=0.5"
    assert reason.rule == 11 and reason.code == rules.OVERRIDE_CODE
    assert rules.overridden(reason) == (11, "1.0 A") and rules.check_reason(reason) == ()
    assert rules.overridden(Reason("x", rule=11, code="rule-11")) is None
    for hard in (1, 4, 7):
        with pytest.raises(ValueError, match="only preferences"):
            rules.override(hard, "x")
    with pytest.raises(ValueError, match="only preferences"):
        rules.override(14, "x")
    forged = Reason("override; rule 9 would give graph", rule=11, code=rules.OVERRIDE_CODE)
    assert "names rule 9" in rules.check_reason(forged)[0]


def test_plan_lists_its_overrides():
    from fastmace.planner.plan import Plan

    plan = Plan(tier="graph", decisions=(
        ("NeighbourConfig.skin", "0.5", rules.override(11, "1.0", detail="requested")),
        ("mode", "graph", Reason("graph tier", code="mode-graph"))),
        requested=(("NeighbourConfig.skin", "0.5"),))
    assert plan.overrides() == (("NeighbourConfig.skin", "0.5", 11, "1.0"),)
    assert "NeighbourConfig.skin = 0.5  (override; rule 11 would give 1.0: requested)" in plan.describe(full=True)
    assert "skin 0.5 (override; rule 11 would give 1.0)" in str(plan)  # the sectioned text (5-T05)


# --------------------------------------------------------------------------- the code against the table


def test_the_scan_sees_the_planner():
    found = emissions()
    assert len(found) > 50
    assert any(code == "inexact-padding" and rule == 4 for _, rule, code in found)
    assert any(code == "scheme-unknown" and rule == 1 for _, rule, code in found)  # through HELPERS
    assert any(code == "selfcheck-failed" and rule == 13 for _, rule, code in found)  # a *_CODE constant


def test_every_numbered_reason_uses_a_code_of_its_rule():
    wrong = [(where, rule, code) for where, rule, code in emissions() if rule is not None
             and (rule not in rules.NUMBERS or (code not in rules.codes(rule) and code != rules.OVERRIDE_CODE))]
    assert not wrong, f"codes outside their rule's table entry (add them to planner/rules.py): {wrong}"


@pytest.mark.parametrize("number", rules.NUMBERS)
def test_a_rule_code_carries_its_number(number, request):
    if number in NUMBERLESS:
        request.applymarker(pytest.mark.xfail(strict=True, reason="5-T02 numbers these reasons"))
    bare = [(where, code) for where, rule, code in emissions() if code in rules.codes(number) and rule != number]
    assert not bare, f"rule {number}'s codes emitted without rule={number}: {bare}"


@pytest.mark.parametrize("number", rules.NUMBERS)
def test_every_rule_has_a_marked_test(number, request):
    if number in UNTESTED:
        request.applymarker(pytest.mark.xfail(strict=True, reason="5-T02's conformance suite marks it"))
    assert number in marked_rules(), f"no test is marked pytest.mark.rule({number})"


def test_the_precedence_pairs_follow_the_order():
    """In every pair the winner comes first in §9.3's order; ``b`` is a later rule, an override or a flag."""
    flags = {rules.allow_flag(n) for n in rules.NUMBERS} - {None}
    for a, b in PAIRS:
        assert a in rules.NUMBERS, (a, b)
        if isinstance(b, int):
            assert b in rules.NUMBERS and a < b, (a, b)
        elif b == "override":
            assert rules.kind(a) == "hard", (a, b)  # a user override replaces a preference, never a hard rule
        else:
            assert b in flags and rules.kind(a) == "hard", (a, b)  # a flag never lifts more than it names


@pytest.mark.parametrize("pair", sorted(PAIRS, key=str), ids=str)
def test_every_precedence_pair_has_a_marked_test(pair):
    assert pair in marked_pairs(), f"no test is marked pytest.mark.precedence{pair}"


def test_every_marked_pair_is_listed():
    assert marked_pairs() <= PAIRS, f"marked but not in PAIRS: {sorted(marked_pairs() - PAIRS, key=str)}"
