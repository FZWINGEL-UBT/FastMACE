"""The precedence of SPEC §9.3's rules (plan task 5-T02, phase 5): pairs where an earlier rule wins.

Rules are evaluated in strict order. A later rule cannot weaken a hard rule (1-7), a user override cannot
lift one (except through its ``allow_*`` flag), and a later preference does not undo an earlier one. Each test
carries ``pytest.mark.precedence(a, b)``: rule ``a`` wins over ``b``, where ``b`` is a rule number,
``"override"`` (a user's explicit value) or an ``allow_*`` flag. ``test_rules_meta.PAIRS`` lists the pairs.

Pairs that need later tasks are added there: rule 5 against the MACE model's rule 12 (5-T03, owner decision
D1) and against runtime adaptation (5-T07, D2 and D3). Rule 15 has no pair: it only warns and overrides
nothing (5-T04; tests/cpu/test_rules_15.py).
"""

import dataclasses

import pytest
import test_minimal_planner as tmp
from ase.constraints import FixAtoms

from _support.rule_cases import assert_conforms, codes, refused
from _support.structures import nacl
from fastmace.config import DEFAULT_SKIN_A, EngineConfig, IntegratorConfig, NeighbourConfig
from fastmace.planner import minimal as minimal_module
from fastmace.planner import reproducible
from fastmace.planner.workload import Workload

WORKLOAD = tmp.WORKLOAD
MC_SWAP = dict(species_changes=True, mc_moves=("swap",), min_mc_temperature_K=300.0)
PINNED = {reproducible.CUEQ_OPS: reproducible.PINNED_CUEQ_OPS}
D3_CUTOFF = tmp.D3_CUTOFF
RULE_5_KEYS = ("EngineConfig.backend", "EngineConfig.cueq_conv_fusion", "reproducible", "reproducible.environment")


@pytest.fixture(autouse=True)
def _cuda_not_initialised(monkeypatch):
    monkeypatch.setattr(minimal_module, "_cuda_initialised", lambda: False)


def inexact(label="mh1-r2scan"):
    return dataclasses.replace(tmp.tp.traits(label), padding_is_exact=False)


def plan_inexact(workload=WORKLOAD, planner=None, **kwargs):
    planner = planner or tmp.minimal()
    return planner.plan_traits(inexact(), nacl(), IntegratorConfig(), workload, tmp.GPU, precision="float32", **kwargs)


def reproducible_plan(workload=None, planner=None, **kwargs):
    workload = workload or Workload(max_temperature_K=300.0, reproducible=True)
    return tmp.plan_label("mh1-r2scan", workload=workload, precision="float64",
                          planner=planner or tmp.minimal(versions=PINNED), **kwargs)


def rule_5_decisions(plan):
    return {key: (plan.decision(key)[1], plan.decision(key)[2].code if hasattr(plan.decision(key)[2], "code")
                  else None) for key in RULE_5_KEYS}


# --------------------------------------------------------------------------- rule 4 against 8, 9 and overrides


@pytest.mark.rule(4)
@pytest.mark.rule(8)
@pytest.mark.precedence(4, 8)
def test_rule_4_wins_over_rule_8():
    """Species changes keep graph mode (rule 8) only where graph mode is allowed: inexact padding plans the
    compiled tier, and the superset build stays."""
    got = plan_inexact(Workload(max_temperature_K=300.0, **MC_SWAP), species=["Na", "Cl", "K"])
    assert got.tier == "compiled"
    mode = got.decision("mode")[2]
    assert (mode.rule, mode.code) == (4, "mode-compiled-rule-4")
    assert got.decision("superset")[2].rule == 8 and got.superset == (11, 17, 19)
    assert_conforms(got)
    error = refused(plan_inexact, Workload(max_temperature_K=300.0, **MC_SWAP), mode="graph")
    assert codes(error) == ["inexact-padding"] and error.reasons[0].rule == 4


@pytest.mark.rule(4)
@pytest.mark.rule(9)
@pytest.mark.precedence(4, 9)
def test_rule_4_wins_over_rule_9():
    """Host callbacks keep the graph tier (rule 9) unless rule 4 forbids it."""
    got = plan_inexact(Workload(max_temperature_K=300.0, host_callbacks_every=5))
    assert got.tier == "compiled" and got.decision("mode")[2].rule == 4
    reason = got.decision("workload.host_callbacks_every")[2]
    assert (reason.rule, reason.code) == (9, "host-callbacks-compiled")
    assert_conforms(got)


@pytest.mark.rule(4)
@pytest.mark.precedence(4, "override")
def test_an_explicit_graph_mode_never_lifts_rule_4():
    error = refused(plan_inexact, Workload(max_temperature_K=300.0, host_callbacks_every=5), mode="graph")
    assert codes(error) == ["inexact-padding"]
    assert 'mode="compiled"' in error.reasons[0]  # refused with the way out named, never downgraded


# --------------------------------------------------------------------------- rule 3 against an explicit backend


@pytest.mark.rule(3)
@pytest.mark.precedence(3, "override")
def test_an_explicit_plain_compiled_backend_never_lifts_rule_3():
    """PolarMACE with ``backend="cueq"`` compiled is refused, never switched to hybrid or eager behind the
    caller's back; ``allow_naive_cueq`` and ``allow_reference`` do not lift it either."""
    planner = tmp.minimal(engine=EngineConfig(backend="cueq"))
    for flags in ({}, dict(allow_naive_cueq=True), dict(allow_reference=True)):
        error = refused(tmp.plan_label, "polar1-s", planner=planner, **flags)
        assert codes(error) == ["polar-plain-compiled-cueq"] and error.reasons[0].rule == 3


# --------------------------------------------------------------------------- rule 5 against 9, 11, 12 and overrides


@pytest.mark.rule(5)
@pytest.mark.rule(9)
@pytest.mark.precedence(5, 9)
def test_rule_5_wins_over_rule_9():
    """Host callbacks keep a reproducible plan on a qualified tier; the override ``mode="compiled"``, which rule 9
    allows, is refused under rule 5 (the compiled tier is not qualified)."""
    callbacks = Workload(max_temperature_K=300.0, reproducible=True, host_callbacks_every=5)
    got = reproducible_plan(callbacks)
    assert got.tier in reproducible.QUALIFIED_TIERS
    assert got.decision("workload.host_callbacks_every")[2].rule == 9
    assert rule_5_decisions(got) == rule_5_decisions(reproducible_plan())
    assert_conforms(got)
    error = refused(reproducible_plan, callbacks, mode="compiled")
    assert codes(error) == ["reproducible-tier"] and error.reasons[0].rule == 5


@pytest.mark.rule(5)
@pytest.mark.rule(11)
@pytest.mark.precedence(5, 11)
def test_rule_5_wins_over_rule_11():
    """Rule 11 sizes a reproducible MC plan and takes a skin override, and neither changes rule 5's
    dimensions (backend, fusion, qualification)."""
    base = rule_5_decisions(reproducible_plan())
    mc = reproducible_plan(Workload(max_temperature_K=1500.0, reproducible=True, **MC_SWAP))
    assert mc.neighbours.skin > DEFAULT_SKIN_A and mc.decision("NeighbourConfig.skin")[2].rule == 11
    assert rule_5_decisions(mc) == base
    assert_conforms(mc)
    skin = reproducible_plan(planner=tmp.minimal(versions=PINNED, neighbours=NeighbourConfig(skin=0.6)))
    assert skin.neighbours.skin == 0.6 and skin.overrides()[0][:3] == ("NeighbourConfig.skin", "0.6 A", 11)
    assert rule_5_decisions(skin) == base


@pytest.mark.rule(5)
@pytest.mark.rule(12)
@pytest.mark.precedence(5, 12)
def test_rule_5_wins_over_rule_12():
    """Rule 12 on the D3 list (3b-T02) warns on a reproducible plan and changes none of rule 5's dimensions.
    The MACE model's estimate (5-T03, D1) adds its own pair."""
    base = rule_5_decisions(reproducible_plan())
    fits = reproducible_plan(d3_cutoff_A=D3_CUTOFF)
    assert fits.decision("d3 list memory")[2].rule == 12 and rule_5_decisions(fits) == base
    small = dataclasses.replace(tmp.GPU, memory_bytes=2**24)
    planner = tmp.minimal(versions=PINNED)
    with pytest.warns(ResourceWarning, match="rule 12"):
        warned = planner.plan_traits(tmp.tp.traits("mh1-r2scan"), nacl(), IntegratorConfig(),
                                     Workload(max_temperature_K=300.0, reproducible=True), small,
                                     precision="float64", d3_cutoff_A=D3_CUTOFF)
    assert warned.decision("d3 list memory")[1] == "warn" and rule_5_decisions(warned) == base
    assert_conforms(warned)


@pytest.mark.rule(5)
@pytest.mark.precedence(5, "override")
@pytest.mark.parametrize(("engine", "code"), [
    (EngineConfig(backend="hybrid"), "reproducible-backend"),
    (EngineConfig(backend="cueq", cueq_conv_fusion="on"), "reproducible-conv-fusion"),
])
def test_an_explicit_option_never_lifts_rule_5(engine, code):
    """Refused, never downgraded to the qualified plain unfused backend."""
    error = refused(reproducible_plan, planner=tmp.minimal(versions=PINNED, engine=engine))
    assert codes(error) == [code] and error.reasons[0].rule == 5


# --------------------------------------------------------------------------- rule 1 against allow_reference


@pytest.mark.rule(1)
@pytest.mark.precedence(1, "allow_reference")
def test_allow_reference_never_lifts_mc_with_constraints():
    """Decision 10: MC moves with constraints are refused, also in the reference tier."""
    atoms = nacl()
    atoms.set_constraint(FixAtoms(indices=[0]))
    workload = Workload(max_temperature_K=300.0, **MC_SWAP)
    for mode in (None, "reference"):
        error = refused(tmp.plan_label, "mh1-r2scan", atoms=atoms.copy(), integrator=IntegratorConfig(fixcm=False),
                        workload=workload, allow_reference=True, mode=mode)
        assert codes(error) == ["mc-with-constraints"] and error.reasons[0].rule == 1
    slab = nacl()
    slab.pbc = (True, True, False)
    error = refused(tmp.plan_label, "mh1-r2scan", atoms=slab, allow_reference=True, mode="reference")
    assert codes(error) == ["non-3d-pbc"]


# --------------------------------------------------------------------------- rule 11 against rule 12


@pytest.mark.rule(11)
@pytest.mark.rule(12)
@pytest.mark.precedence(11, 12)
def test_rule_12_never_shrinks_rule_11s_sizes():
    """Short of memory, rule 12 warns on the D3 list; the D3 skin, the MACE skin and the headroom rule 11 sized
    stay (under every option of D1; 5-T03 adds the MACE model's half)."""
    roomy = tmp.minimal().plan_traits(tmp.tp.traits("mh1-r2scan"), nacl(), IntegratorConfig(),
                                      Workload(max_temperature_K=1500.0, **MC_SWAP), tmp.GPU, precision="float32",
                                      d3_cutoff_A=D3_CUTOFF)
    small = dataclasses.replace(tmp.GPU, memory_bytes=2**24)
    with pytest.warns(ResourceWarning, match="rule 12"):
        tight = tmp.minimal().plan_traits(tmp.tp.traits("mh1-r2scan"), nacl(), IntegratorConfig(),
                                          Workload(max_temperature_K=1500.0, **MC_SWAP), small, precision="float32",
                                          d3_cutoff_A=D3_CUTOFF)
    assert tight.decision("d3 list memory")[1] == "warn"
    for part in ("d3_skin", "skin", "capacity_headroom"):
        assert getattr(tight.neighbours, part) == getattr(roomy.neighbours, part), part
    assert tight.engine.recapture_budget == roomy.engine.recapture_budget
    for key in ("NeighbourConfig.d3_skin", "NeighbourConfig.skin", "NeighbourConfig.headroom"):
        assert tight.decision(key) == roomy.decision(key), key
    assert_conforms(tight)
