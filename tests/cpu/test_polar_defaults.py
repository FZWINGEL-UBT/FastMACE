"""Decision 58 (OPT4-T02): ``fx_e3nn_linear`` and ``polar_graph_sums`` on by default for POLAR-1-S and -M.

- under ``auto`` the planner turns both on for PolarMACE with a qualified checkpoint (``POLAR_GRAPH_LABELS``),
  found by label or by SHA-256 and head as rule 15 finds it, and off for POLAR-1-L, MH-1, a generic model and a
  model of unknown identity, with the reason;
- an explicit ``on`` or ``off`` overrides the rule; float64 plans keep MACE's code, and reproducible POLAR-1 is
  refused anyway (rule 5).
"""

import types

import pytest
import test_minimal_planner as tm
import test_planner as tp

from fastmace.config import EngineConfig, RewriteConfig
from fastmace.errors import PlanRefused
from fastmace.model import labels
from fastmace.planner import fixed
from fastmace.planner.workload import Workload

RECORDS = fixed.POLAR_GRAPH_RECORDS
RULE = "on_for_hybrid_float32_unless_reproducible_for_qualified_polar1_checkpoints"


def loaded(label, *, by_hash=False):
    """The ``loaded`` model's identity, as ``fm.load`` gives it (test_rules_15.loaded)."""
    entry = labels.LABELS[label]
    return types.SimpleNamespace(label=None if by_hash else label, head=entry.head, sha256=entry.checkpoint.sha256)


def states(plan):
    return {name: getattr(plan.rewrites, name) for name in RECORDS}


def test_the_rule_and_the_qualified_labels():
    assert fixed.POLAR_GRAPH_LABELS == ("polar1-s", "polar1-m")
    assert set(fixed.POLAR_GRAPH_LABELS) <= set(labels.LABELS)
    for name in RECORDS:
        assert fixed.AUTO_RULES[f"RewriteConfig.{name}"] == RULE


@pytest.mark.parametrize("label", ["polar1-s", "polar1-m"])
@pytest.mark.parametrize("by_hash", [False, True])
def test_minimal_planner_turns_both_on_for_s_and_m(label, by_hash):
    plan = tm.plan_label(label, loaded=loaded(label, by_hash=by_hash))
    assert states(plan) == dict.fromkeys(RECORDS, "on")
    off = dict(plan.auto_options_off)
    assert not set(RECORDS) & set(off)


@pytest.mark.parametrize("label", ["polar1-l", "mh1-r2scan", "omat0-medium"])
def test_minimal_planner_leaves_other_models_off(label):
    plan = tm.plan_label(label, loaded=loaded(label))
    assert states(plan) == dict.fromkeys(RECORDS, "off")
    off = dict(plan.auto_options_off)
    for name in RECORDS:
        assert off[name].startswith("not a qualified POLAR-1 checkpoint, decision 58"), off[name]
        if label.startswith("polar"):
            assert f"label {label}" in off[name]


def test_unknown_identity_stays_off():
    plan = tm.plan_label("polar1-s")  # no loaded model: no checkpoint identity
    assert states(plan) == dict.fromkeys(RECORDS, "off")
    assert "label None" in dict(plan.auto_options_off)["fx_e3nn_linear"]


def test_fixed_planner_reads_the_label():
    for label, want in (("polar1-s", "on"), ("polar1-m", "on"), ("polar1-l", "off")):
        plan = tp.planner().plan(tp.traits(label), tp.nacl(), tp.IntegratorConfig(), tp.GPU, model_label=label)
        assert states(plan) == dict.fromkeys(RECORDS, want), label


def test_explicit_requests_override_the_rule():
    on = tm.plan_label("polar1-l", loaded=loaded("polar1-l"),
                       rewrites=RewriteConfig(fx_e3nn_linear="on", polar_graph_sums="on"))
    assert states(on) == dict.fromkeys(RECORDS, "on")
    off = tm.plan_label("polar1-s", loaded=loaded("polar1-s"), rewrites=RewriteConfig(fx_e3nn_linear="off"))
    assert states(off) == {"fx_e3nn_linear": "off", "polar_graph_sums": "on"}


def test_float64_and_reproducible_plans_keep_maces_code():
    # Plain cuEq float64: PolarMACE runs it uncompiled only (compiled it returns zero forces); both records off.
    plan = tm.plan_label("polar1-s", loaded=loaded("polar1-s"), precision="float64",
                         planner=tm.minimal(engine=EngineConfig(backend="cueq", compile_mode="none")))
    assert plan.precision.dtype == "float64"
    assert states(plan) == dict.fromkeys(RECORDS, "off")
    # reproducible=True qualifies only the MH-1 family on plain cuEq (rule 5): POLAR-1 is refused before any rewrite.
    with pytest.raises(PlanRefused, match="only the MH-1 family is qualified"):
        tm.plan_label("polar1-s", workload=Workload(max_temperature_K=1000.0, reproducible=True),
                      loaded=loaded("polar1-s"), precision="float64",
                      planner=tm.minimal(engine=EngineConfig(compile_mode="none")))
