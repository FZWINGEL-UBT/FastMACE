"""The planner on an unknown MACE (plan task 2-T04b; SPEC §6.3, §18 decision 2).

Three decisions the planner makes from the installed sources, each with its Reason and code,
tested with 2-T04's fake hashes (``test_versioned_registry.py``):

- capture: a multi-head model whose capture-safe mask does not hash runs on the eager tier from
  revision 0 (``versions.capture_support``);
- conv fusion: on an unhashed cuEq converter, plain cuEq follows MACE's own conversion rule
  instead of raising (``versions.mace_conv_fusion``);
- compilation: without MACE's compile API the plan does not compile (``versions.compile_support``).

On the qualified MACE none of them changes a plan (the default path).
"""

from __future__ import annotations

import dataclasses

import pytest
import test_minimal_planner as tmp
import test_planner as tp

from _support.structures import nacl
from fastmace import _patches
from fastmace.config import EngineConfig, IntegratorConfig
from fastmace.errors import PlanRefused
from fastmace.planner import fixed
from fastmace.rewrites import registry as reg
from fastmace.rewrites import versions

pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.skipif(reg.installed_mace_version() not in versions.QUALIFIED_MACE_VERSIONS,
                       reason="fake hashes against the qualified MACE 0.3.16"),
]

WRONG = "0" * 64


def _unpin(monkeypatch, name):
    monkeypatch.setitem(_patches.SOURCE_PINS, name, {target: WRONG for target in _patches.SOURCE_PINS[name]})


def _traits(heads=("matpes_r2scan", "omat_pbe")):
    return dataclasses.replace(tp.traits("mh1-r2scan"), heads=heads)


def _plan(traits=None, planner=None, **flags):
    planner = planner or tmp.minimal()
    return planner.plan_traits(traits or _traits(), nacl(), IntegratorConfig(), tmp.WORKLOAD, tp.GPU,
                               precision="float32", **flags)


def _evaluation(planner=None):
    planner = planner or tmp.minimal()
    return planner.plan_evaluation_traits(_traits(), nacl(), tp.GPU, precision="float32")


def _code(plan, key):
    return getattr(plan.decision(key)[2], "code", None)


# --------------------------------------------------------------------------- the default path


def test_the_qualified_mace_changes_no_plan():
    assert versions.compile_support().supported
    assert versions.capture_support(_traits()).supported
    assert versions.mace_conv_fusion("cuda:0") is None  # the converter hashes: FastMACE's own rule
    plan = _plan()
    assert plan.tier == fixed.TIER and plan.decision("capture")[1] == "on"
    assert plan.engine.compile_mode != "none"
    assert _code(plan, "mode") == "mode-graph"
    assert _code(plan, "EngineConfig.compile_mode") is None
    cueq = _plan(planner=tmp.minimal(engine=EngineConfig(backend="cueq")))
    assert cueq.decision("EngineConfig.cueq_conv_fusion")[1] == "on"


# --------------------------------------------------------------------------- (i) capture


def test_a_multi_head_model_without_the_hashed_mask_runs_eager(monkeypatch):
    _unpin(monkeypatch, "capture_safe_mask_head")
    plan = _plan()
    assert plan.tier == fixed.EAGER_TIER and plan.revision == 0
    assert plan.decision("capture")[1] == "off"
    assert plan.engine.compile_mode == "none"
    key, choice, reason = plan.decision("mode")
    assert choice == "eager" and reason.code == versions.CAPTURE_CODE and "mask" in reason
    assert not any(item.startswith("EngineConfig.") for item, _ in plan.requested)


def test_a_single_head_model_keeps_the_graph_tier(monkeypatch):
    _unpin(monkeypatch, "capture_safe_mask_head")
    plan = _plan(_traits(heads=("default",)))
    assert plan.tier == fixed.TIER and _code(plan, "mode") == "mode-graph"


def test_an_eager_request_stays_eager(monkeypatch):
    _unpin(monkeypatch, "capture_safe_mask_head")
    plan = _plan(mode="eager")
    assert plan.tier == fixed.EAGER_TIER and _code(plan, "mode") == "mode-eager-tier"


# --------------------------------------------------------------------------- (ii) conv fusion


def test_mace_conv_fusion_rule():
    assert versions.mace_conv_fusion("cuda:0") is None


def test_an_unhashed_converter_follows_mace_rule(monkeypatch):
    _unpin(monkeypatch, "cueq_conv_fusion")
    assert versions.mace_conv_fusion("cuda:0") is False  # MACE fuses only for the device string "cuda"
    assert versions.mace_conv_fusion("cuda") is True
    plan = _plan(planner=tmp.minimal(engine=EngineConfig(backend="cueq")))
    key, choice, reason = plan.decision("EngineConfig.cueq_conv_fusion")
    assert choice == "off" and reason.code == versions.CONV_FUSION_CODE
    assert plan.engine.cueq_conv_fusion == "off"
    with _patches.cueq_conv_fusion(False, plan.engine.device) as record:
        assert record is None  # the conversion needs no patch: nothing raises


def test_an_explicit_fusion_mace_cannot_give_is_refused(monkeypatch):
    _unpin(monkeypatch, "cueq_conv_fusion")
    with pytest.raises(PlanRefused) as caught:
        _plan(planner=tmp.minimal(engine=EngineConfig(backend="cueq", cueq_conv_fusion="on")))
    assert versions.CONV_FUSION_CODE in tmp.codes(caught.value)


def test_the_hybrid_is_unaffected_by_the_converter(monkeypatch):
    _unpin(monkeypatch, "cueq_conv_fusion")
    plan = _plan()
    assert plan.engine.backend == "hybrid" and _code(plan, "EngineConfig.cueq_conv_fusion") is None


# --------------------------------------------------------------------------- (iii) compilation


def test_compile_support_checks_mace_compile_api(monkeypatch):
    from mace.tools import compile as mace_compile

    monkeypatch.delattr(mace_compile, "configure_autograd_for_compile")
    support = versions.compile_support()
    assert not support.supported and "configure_autograd_for_compile" in support.reason


def test_without_the_compile_api_the_plan_does_not_compile(monkeypatch):
    from mace.tools import compile as mace_compile

    monkeypatch.delattr(mace_compile, "configure_autograd_for_compile")
    plan = _plan()
    assert plan.tier == fixed.TIER and plan.engine.compile_mode == "none"
    key, choice, reason = plan.decision("EngineConfig.compile_mode")
    assert choice == "none" and reason.code == versions.COMPILE_CODE
    assert "EngineConfig.compile_mode" not in dict(plan.requested)
    evaluation = _evaluation()
    assert evaluation.engine.compile_mode == "none" and _code(evaluation, "EngineConfig.compile_mode") == \
        versions.COMPILE_CODE


def test_all_three_together(monkeypatch):
    from mace.tools import compile as mace_compile

    monkeypatch.delattr(mace_compile, "configure_autograd_for_compile")
    _unpin(monkeypatch, "capture_safe_mask_head")
    _unpin(monkeypatch, "cueq_conv_fusion")
    plan = _plan(planner=tmp.minimal(engine=EngineConfig(backend="cueq")))
    assert plan.tier == fixed.EAGER_TIER and plan.engine.compile_mode == "none"
    assert plan.engine.cueq_conv_fusion == "off"
    assert _code(plan, "mode") == versions.CAPTURE_CODE
