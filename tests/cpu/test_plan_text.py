"""``str(plan)`` by sections (plan task 5-T05; SPEC §9.5; docs/design/planner-5.md, "``str(plan)`` layout").

- Golden texts for seven plans: hybrid NVT, NPT with D3, MC with a superset, the compiled tier by rule 4,
  the reference tier, a fallback revision, and a reproducible plan. ``FASTMACE_UPDATE_GOLDEN=1`` rewrites
  them (docs/development.md); 5-T03, 5-T07 and 5-T09 update their sections that way.
- Every golden text: the sections in the layout's order, no line over 120 characters, at most 30 lines.
- ``describe(full=True)`` keeps the every-decision list; provenance stores both texts.
- The placeholders (memory, calibration) and their filled forms; overrides; rule 15 warnings.
"""

import dataclasses
import os
from pathlib import Path

import pytest
import test_minimal_planner as tmp
import test_planner as tp
import test_reproducible as trep

from _support.structures import nacl
from fastmace import provenance
from fastmace.config import IntegratorConfig
from fastmace.errors import Reason
from fastmace.planner import fallback, plan_text, rules
from fastmace.planner import minimal as minimal_module
from fastmace.planner.fallback import TriedCandidate
from fastmace.planner.plan import CalibrationCandidate, CalibrationRecord, Plan
from fastmace.planner.workload import Workload

GOLDEN = Path(__file__).resolve().parent / "golden" / "plan_text"
UPDATE = os.environ.get("FASTMACE_UPDATE_GOLDEN") == "1"
# The line prefixes of the layout's sections, in order (section 1's first line is "Plan revision").
SECTIONS = (("header", ("backend ",)), ("hard rules", ("hard rules:",)), ("rewrites", ("rewrites ",)),
            ("sizes", ("sizes:",)), ("kernels", ("kernels:",)), ("memory", ("memory ",)),
            ("workload", ("workload:", "dispersion:", "rules 8-11:")), ("warnings", ("warning ",)),
            ("history", ("fallbacks", "calibration:", "self-checks:", "revisions:")))
ALWAYS = ("header", "rewrites", "sizes", "kernels", "memory", "workload", "history")


def _hybrid_nvt():
    return tmp.plan_label("mh1-omat")


def _npt_d3():
    from fastmace.longrange.d3 import D3

    plan = tmp.plan_label("mh1-r2scan", integrator=IntegratorConfig(ensemble="npt"), d3_cutoff_A=tmp.D3_CUTOFF,
                          variable_cell=True, workload=Workload(max_temperature_K=1000.0))
    return minimal_module._with_dispersion(plan, D3(functional="r2scan"))


def _mc_superset():
    return tmp.plan_label("mh1-r2scan", species=["Na", "Cl", "K"],
                          workload=Workload(max_temperature_K=300.0, **tmp.MC_SWAP))


def _compiled_rule_4():
    record = dataclasses.replace(tp.traits("mh1-r2scan"), padding_is_exact=False)
    return tmp.minimal().plan_traits(record, nacl(), IntegratorConfig(), tmp.WORKLOAD, tp.GPU, precision="float32")


def _reference():
    return tmp.plan_label("mh1-omat", mode="reference")


def _fallback_revision():
    plan = tmp.plan_label("mh1-omat")
    revision = fallback.candidate(plan, {"rewrites": {"streaming_conv": "off"}, "reason": "gate failed: forces"},
                                  None, None)
    tried = (TriedCandidate("rewrites", "rewrites fold_cutoff off", "failed", "gate failed: forces", stage="gate",
                            reduced_gate_runs=1),
             TriedCandidate("rewrites", "rewrites streaming_conv off", "activated", "gate failed: forces",
                            revision=revision.revision, reduced_gate_runs=1, full_gate_runs=1))
    return dataclasses.replace(revision, fallbacks=tried)


def _reproducible():
    return trep.plan()


PLANS = {"hybrid-nvt": _hybrid_nvt, "npt-d3": _npt_d3, "mc-superset": _mc_superset,
         "compiled-rule-4": _compiled_rule_4, "reference": _reference, "fallback-revision": _fallback_revision,
         "reproducible": _reproducible}


@pytest.fixture
def make(monkeypatch):
    monkeypatch.setattr(minimal_module, "_cuda_initialised", lambda: False)  # as test_reproducible's plans
    return lambda name: PLANS[name]()


def _sections(text):
    """The section of each line after the header's first line, in order."""
    found = []
    for line in text.splitlines()[1:]:
        if line.startswith("      "):  # a wrapped continuation line
            continue
        stripped = line.strip()
        names = [name for name, prefixes in SECTIONS if stripped.startswith(prefixes)]
        assert len(names) == 1, f"line of no section: {line!r}"
        found.append(names[0])
    return found


@pytest.mark.parametrize("name", sorted(PLANS))
def test_golden_text(make, name):
    text = make(name).describe()
    path = GOLDEN / f"{name}.txt"
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8", newline="\n")
    assert path.exists(), f"no golden text {path.name}; write it with FASTMACE_UPDATE_GOLDEN=1"
    assert text + "\n" == path.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", sorted(PLANS))
def test_sections_in_order_and_lines_fit(make, name):
    plan = make(name)
    text = plan.describe()
    lines = text.splitlines()
    assert str(plan) == text
    assert lines[0].startswith(f"Plan revision {plan.revision}, tier {plan.tier} (")
    assert len(lines) <= plan_text.MAX_LINES
    assert all(len(line) <= plan_text.WIDTH for line in lines), [line for line in lines if len(line) > 120]
    found = _sections(text)
    order = [name for name, _ in SECTIONS]
    assert [order.index(section) for section in found] == sorted(order.index(section) for section in found)
    assert {"header", *found} >= set(ALWAYS)


@pytest.mark.parametrize("name", sorted(PLANS))
def test_full_text_lists_every_decision(make, name):
    plan = make(name)
    full = plan.describe(full=True)
    assert full.splitlines()[0].startswith(f"Plan revision {plan.revision}, tier {plan.tier}: {plan.reason}")
    for key, choice, reason in plan.decisions:
        assert f"  {key} = {choice}  ({reason})" in full
    assert full != plan.describe()


def test_bare_plan():
    plan = Plan(tier="graph")
    assert plan.describe(full=True) == "Plan revision 0, tier graph: initial plan"
    text = str(plan)
    assert text.splitlines()[0] == "Plan revision 0, tier graph (initial plan)"
    assert "memory (rule 12): not estimated" in text and "calibration: none" in text


def test_header_hard_rules_and_kernels(make):
    rule_4 = make("compiled-rule-4").describe()
    assert rule_4.startswith("Plan revision 0, tier compiled (rule 4: ")
    assert "  hard rules: rule 4: mode compiled" in rule_4
    nvt = make("hybrid-nvt").describe()
    assert "hard rules" not in nvt
    assert "  backend hybrid | force float32 | MC energy none (no MC) | runner in-process" in nvt
    assert "  kernels: OEq yes, Triton yes, NVRTC yes, fused cuEq yes" in nvt
    assert "  hard rules: rule 5: " in make("reproducible").describe()


def test_rewrites_on_and_off_by_reason(make):
    plan = make("hybrid-nvt")
    text = plan.describe()
    on = [key.split(".", 1)[1] for key, choice, _ in plan.decisions
          if key.startswith("RewriteConfig.") and choice == "on"]
    assert f"  rewrites on ({len(on)}): {on[0]}, " in text
    assert "only an explicit 'on' enables it" in text
    assert "streaming_conv" in text.split("rewrites off")[0]
    after = make("fallback-revision").describe()
    off = after.split("rewrites off: ", 1)[1]
    assert "streaming_conv" in off


def test_placeholders_and_their_filled_forms(make):
    plan = make("hybrid-nvt")
    text = plan.describe()
    assert "  memory (rule 12): not estimated; GPU NVIDIA GeForce RTX 5060 Ti" in text
    assert "  calibration: none" in text and "  fallbacks: none" in text
    assert "  revisions: 0 minimal plan (phase 1b)" in text
    record = CalibrationRecord(candidates=(CalibrationCandidate("", (4.1, 4.2)),
                                           CalibrationCandidate("backend cueq", (4.6, 4.5))),
                               chosen="", changed_plan=False, steps=50, margin=0.05, setup_s=31.0)
    filled = dataclasses.replace(plan, predicted_memory_bytes=3 * 2**30, calibration=record).describe()
    assert "  memory (rule 12): 3.00 GiB predicted; GPU NVIDIA GeForce RTX 5060 Ti" in filled
    assert ("  calibration: kept the rules' plan; rules' plan 4.15 ms, backend cueq 4.55 ms (50 steps, "
            "margin 5 %, setup 31 s)") in filled


def test_fallback_revision_history(make):
    plan = make("fallback-revision")
    text = plan.describe()
    assert text.startswith("Plan revision 1, tier graph (")
    assert "  fallbacks tried (2): rewrites fold_cutoff off: failed at gate; rewrites streaming_conv off: " \
           "activated as revision 1" in text
    assert "  revisions: 0 minimal plan (phase 1b); 1 at step 0: fallback: rewrites streaming_conv off" in text


def test_workload_and_overrides(make):
    plan = make("mc-superset")
    text = plan.describe()
    assert "  workload: Langevin" in text and "MC swap from 300 K" in text and "species changes" in text
    assert "superset Na Cl K" in text
    key, choice, _ = plan.decision("NeighbourConfig.skin")
    decisions = tuple((k, c, rules.override(11, "0.3 A") if k == key else r) for k, c, r in plan.decisions)
    overridden = dataclasses.replace(plan, decisions=decisions).describe()
    assert f"skin {choice} (override; rule 11 would give 0.3 A)" in overridden
    assert "  dispersion: D3(functional=r2scan" in make("npt-d3").describe()


def test_warnings_section_only_with_rule_15():
    plan = tmp.plan_label("mh1-omat")
    assert "warning " not in plan.describe()
    reason = Reason("PolarMACE on a charged slab: the known spurious dipole drift; watch the dipole",
                    rule=15, code="model-structure-warning")
    warned = dataclasses.replace(plan, decisions=(*plan.decisions, ("warn.model_structure.dipole", "warn",
                                                                     reason))).describe()
    lines = [line for line in warned.splitlines() if line.startswith("  warning ")]
    assert lines == ["  warning (rule 15): PolarMACE on a charged slab: the known spurious dipole drift; watch the "
                     "dipole"]
    assert warned.index("  workload:") < warned.index(lines[0]) < warned.index("  fallbacks")


def test_long_lines_are_cut_and_overflow_counted():
    plan = tmp.plan_label("mh1-omat")
    long = Reason("x" * 400, rule=15, code="model-structure-warning")
    many = tuple((f"warn.model_structure.w{i}", "warn", long) for i in range(5))
    text = dataclasses.replace(plan, decisions=(*plan.decisions, *many)).describe()
    warnings = [line for line in text.splitlines() if line.startswith("  warning")]
    assert len(warnings) == 3 and warnings[-1].endswith("(+3 more; describe(full=True) lists them)")
    assert all(len(line) <= plan_text.WIDTH for line in text.splitlines())


def test_provenance_stores_both_texts(make):
    plan = make("fallback-revision")
    record = provenance._plan(plan)
    assert record["description"] == plan.describe() and record["description_full"] == plan.describe(full=True)
    revisions = provenance._plan_revisions(plan)
    assert [entry["description"] for entry in revisions] == [p.describe() for p in (*plan.chain(), plan)]
    assert [entry["description_full"] for entry in revisions] == [p.describe(full=True)
                                                                   for p in (*plan.chain(), plan)]
