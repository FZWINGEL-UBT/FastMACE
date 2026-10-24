"""Rule 15 of SPEC §9.3 in the planner (plan task 5-T04): known model-structure warnings (§4.3).

- Each row of ``planner.warnings.KNOWN_ISSUES`` gives the planner a ``warn.model_structure.<id>`` decision with
  rule 15's code on a matching model and structure, and none on a near miss; it never refuses.
- The model is identified as the report does: by label, or by checkpoint SHA-256 and head.
- The run report reads the plan's rule 15 decisions; without them (a refusal, a plan from before 5-T04) it
  matches the same rows itself, so its lines are unchanged (``test_report.py``).
"""

import types

import pytest
import test_minimal_planner as tm
import test_report as tr
from ase import Atoms

from _support.structures import nacl, nanbo3
from fastmace.config import IntegratorConfig, OptimizerConfig
from fastmace.errors import PlanRefused, Reason
from fastmace.model import labels
from fastmace.planner import rules
from fastmace.planner import warnings as known_issues

pytestmark = pytest.mark.rule(15)
NPT = IntegratorConfig(ensemble="npt", pressure_GPa=0.5)
RELAX_CELL = OptimizerConfig(scheme="lbfgs", cell_filter="frechet")


def perovskite(b, repeat=2):
    """Ideal cubic perovskite Na``b``O3 (a = 3.95 Å)."""
    atoms = nanbo3(repeat=1)
    atoms.set_chemical_symbols(["Na", b, "O", "O", "O"])
    return atoms.repeat((repeat,) * 3)


def molecular(repeat=2):
    """A loose C/H/N/O cell (no chemistry; the planner only reads the species and the cell)."""
    atoms = Atoms("CHNO", cell=[6.0] * 3, pbc=True,
                  scaled_positions=[(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)])
    return atoms.repeat((repeat,) * 3)


def oxide(repeat=2):
    """Rock-salt NaO-like cell: O without C or H (an inorganic oxide for the OMOL-0 row)."""
    atoms = nacl(repeat=1)
    atoms.set_chemical_symbols(["Na" if s == "Na" else "O" for s in atoms.get_chemical_symbols()])
    return atoms.repeat((repeat,) * 3)


def loaded(label, *, by_hash=False):
    """The ``loaded`` model's identity, as ``fm.load`` gives it: a label, or a path's SHA-256 and head."""
    entry = labels.LABELS[label]
    if by_hash:
        return types.SimpleNamespace(label=None, head=entry.head, sha256=entry.checkpoint.sha256)
    return types.SimpleNamespace(label=label, head=entry.head, sha256=entry.checkpoint.sha256)


def plan(label, atoms, integrator=None, *, by_hash=False, **flags):
    return tm.plan_label(label, atoms, integrator, loaded=loaded(label, by_hash=by_hash), **flags)


def rule_15(plan_):
    return {key: (choice, reason) for key, choice, reason in known_issues.decisions(plan_)}


# Each row: (issue id, label, structure, integrator, planner flags).
MATCHES = [
    ("polar1-s-nanbo3", "polar1-s", lambda: perovskite("Nb"), None, {}),
    ("polar1-natao3", "polar1-m", lambda: perovskite("Ta"), None, {}),
    ("omol0-nacl", "omol0-xl", nacl, None, {}),
    ("omol0-oxides", "omol0-xl", oxide, None, {}),
    ("off23-volume", "off23-medium", molecular, NPT, {}),
    ("off23-volume", "off23-small", molecular, None, {"optimizer": RELAX_CELL}),
    ("off23-large-memory", "off23-large", lambda: molecular(repeat=8), None, {}),
]
NEAR_MISSES = [
    ("polar1-m", lambda: perovskite("Nb"), None, {}),       # the Fukui runaway is POLAR-1-S only
    ("polar1-s", nacl, None, {}),                           # no Nb or O
    ("omol0-xl", molecular, None, {}),                      # O, but organic
    ("off23-medium", molecular, None, {}),                  # fixed cell
    ("off23-medium", molecular, None, {"optimizer": OptimizerConfig(scheme="lbfgs")}),  # fixed-cell relaxation
    ("off23-large", molecular, None, {}),                   # 32 atoms, below about 2,000
    ("off24-medium", molecular, NPT, {}),                   # OFF24 has no volume issue
    ("mh1-r2scan", lambda: perovskite("Ta"), None, {}),     # NaTaO3 is POLAR-1's issue
]


def test_the_table_and_its_ids():
    assert len(known_issues.ISSUE_IDS) == len(known_issues.KNOWN_ISSUES) == len(set(known_issues.ISSUE_IDS))
    for names, _needed, condition, _text in known_issues.KNOWN_ISSUES:
        assert set(names) <= set(labels.LABELS)
        assert condition in ("", "npt", "large", "inorganic")
    assert rules.codes(15) == (known_issues.MODEL_STRUCTURE_CODE,)


@pytest.mark.parametrize("issue_id, label, atoms, integrator, flags", MATCHES,
                         ids=[f"{row[0]}-{row[1]}" for row in MATCHES])
def test_a_matching_issue_is_a_plan_decision(issue_id, label, atoms, integrator, flags):
    decided = rule_15(plan(label, atoms(), integrator, **flags))
    assert list(decided) == [known_issues.DECISION_PREFIX + issue_id]
    choice, reason = decided[known_issues.DECISION_PREFIX + issue_id]
    text = known_issues.KNOWN_ISSUES[known_issues.ISSUE_IDS.index(issue_id)][3]
    assert (choice, str(reason)) == ("warn", text)
    assert isinstance(reason, Reason) and (reason.rule, reason.code) == (15, "model-structure-warning")
    assert rules.check_reason(reason) == ()


@pytest.mark.parametrize("label, atoms, integrator, flags", NEAR_MISSES,
                         ids=[f"{row[0]}-{i}" for i, row in enumerate(NEAR_MISSES)])
def test_a_near_miss_has_no_rule_15_decision(label, atoms, integrator, flags):
    assert rule_15(plan(label, atoms(), integrator, **flags)) == {}


def test_the_model_is_found_by_its_checkpoint_hash():
    decided = rule_15(plan("polar1-s", perovskite("Nb"), by_hash=True))
    assert list(decided) == ["warn.model_structure.polar1-s-nanbo3"]


def test_a_model_outside_the_table_has_no_known_issue():
    other = types.SimpleNamespace(label=None, head=None, sha256="ab" * 32)
    assert rule_15(tm.plan_label("polar1-s", perovskite("Nb"), loaded=other)) == {}
    assert rule_15(tm.plan_label("polar1-s", perovskite("Nb"))) == {}  # no loaded model: traits only


def test_two_issues_and_never_a_refusal():
    """OMOL-0 on an inorganic Na/Cl/O cell matches two rows; a warning never refuses (no PlanRefused)."""
    atoms = nacl(repeat=1)
    atoms.set_chemical_symbols(["Na", "Cl", "O", "Na", "Cl", "O", "Na", "Cl"])
    decided = rule_15(plan("omol0-xl", atoms.repeat(2)))
    assert list(decided) == ["warn.model_structure.omol0-nacl", "warn.model_structure.omol0-oxides"]


def test_the_report_reads_the_plans_decisions():
    """The report shows the plan's rule 15 decisions, even where its own inputs would match none."""
    issue = known_issues.Issue("polar1-s-nanbo3", known_issues.KNOWN_ISSUES[0][3])
    decision = (issue.key, "warn", Reason(issue.text, rule=15, code=known_issues.MODEL_STRUCTURE_CODE))
    text = tr.make_report(tr.make_plan(extra=(decision,))).setup_text()
    assert f"warn:    {issue.text}" in text.splitlines()
    assert "warn:    none" in tr.make_report().setup_text().splitlines()


def test_the_report_matches_rows_itself_without_the_decisions():
    """A plan without rule 15 decisions (from before 5-T04, or a refusal): the report matches as before; a
    variable-cell relaxation now counts for the OFF23 volume row, as the planner's does."""
    import dataclasses

    species = ("C", "H", "N", "O")
    relax = dataclasses.replace(tr.make_plan(), optimizer=RELAX_CELL)
    assert "below the experimental volume" in tr.make_report(relax, model="off23-medium",
                                                             species=species).setup_text()
    refused = PlanRefused.from_reasons([Reason("not 3D periodic", rule=1, code="non-3d-pbc")])
    text = tr.make_report(plan=None, refusal=refused, integrator=IntegratorConfig(), model="polar1-s",
                          species=("Na", "Nb", "O")).setup_text()
    assert "Fukui" in text
