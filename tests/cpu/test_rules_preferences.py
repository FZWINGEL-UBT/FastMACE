"""Conformance of the preferences 8-12 of SPEC §9.3 (plan task 5-T02, phase 5; conductor's choices 1 and 4).

- Each preference's default decision, with its rule number and a code of its rule.
- A user override replaces the decision and is recorded uniformly (§9.5, choice 4): the reason reads
  ``override; rule n would give X``, its code is ``override``, ``Plan.overrides()`` lists it and
  ``Plan.requested`` keeps the value. A requested value equal to the rule's choice is no override.
- The explicit ``mode="compiled"`` keeps its code ``mode-compiled-tier`` (the report reads it; conductor's
  answer to 5-T02 question 2) and says that it was requested.
- Rule 12 on today's D3 list (3b-T02). The MACE model's estimate is plan task 5-T03's, after owner decision D1.

The tiny model's traits with injected ``SoftwareTraits`` and ``Hardware``: no GPU.
"""

import dataclasses

import pytest
import test_minimal_planner as tmp

from _support.rule_cases import assert_conforms, codes, refused
from _support.structures import nacl
from fastmace.config import DEFAULT_SKIN_A, EngineConfig, IntegratorConfig, NeighbourConfig
from fastmace.planner import rules
from fastmace.planner.workload import Workload

WORKLOAD = tmp.WORKLOAD
MC_SWAP = dict(species_changes=True, mc_moves=("swap",), min_mc_temperature_K=300.0)
MC = Workload(max_temperature_K=1500.0, **MC_SWAP)
D3_CUTOFF = tmp.D3_CUTOFF


def plan(label="mh1-r2scan", **kwargs):
    return tmp.plan_label(label, **kwargs)


def numbered(plan_, key, rule, code=None):
    """The reason of decision ``key``: rule ``rule`` (and ``code``), a preference."""
    reason = plan_.decision(key)[2]
    assert rules.kind(rule) == "preference"
    assert reason.rule == rule, (key, reason.rule, str(reason))
    if code is not None:
        assert reason.code == code, (key, reason.code)
    return reason


def overridden(plan_, key, rule, choice, would_give):
    """Decision ``key`` is the caller's ``choice``, recorded as an override of ``rule``."""
    item, got, reason = plan_.decision(key)
    assert got == choice and reason.code == rules.OVERRIDE_CODE and reason.rule == rule
    assert rules.overridden(reason) == (rule, would_give)
    assert str(reason).startswith(f"override; rule {rule} would give {would_give}")
    assert (key, choice, rule, would_give) in plan_.overrides()
    return reason


# --------------------------------------------------------------------------- rule 8


@pytest.mark.rule(8)
def test_rule_8_species_changes_keep_the_graph_tier_and_build_for_the_superset():
    got = plan(workload=MC, species=["Na", "Cl", "K"])
    assert got.tier == "graph" and got.superset == (11, 17, 19)
    numbered(got, "superset", 8, "superset")
    reason = numbered(got, "workload.species_changes", 8, "species-changes")
    assert got.decision("workload.species_changes")[1] == "on"
    assert "graph mode" in reason and "superset" in reason and "not in 1b" not in reason
    assert_conforms(got)


@pytest.mark.rule(8)
def test_rule_8_default_without_species_changes():
    got = plan()
    numbered(got, "superset", 8, "superset")
    reason = numbered(got, "workload.species_changes", 8, "species-changes")
    assert got.decision("workload.species_changes")[1] == "off" and "not in 1b" not in reason
    assert got.tier == "graph" and got.overrides() == ()
    assert_conforms(got)


# --------------------------------------------------------------------------- rule 9


@pytest.mark.rule(9)
def test_rule_9_host_callbacks_keep_the_graph_tier():
    got = plan(workload=Workload(max_temperature_K=300.0, host_callbacks_every=5))
    assert got.tier == "graph"
    numbered(got, "workload.host_callbacks_every", 9, "host-callbacks-graph-sync")
    numbered(plan(), "workload.host_callbacks_every", 9, "no-host-callbacks")
    assert_conforms(got)


@pytest.mark.rule(9)
def test_rule_9_override_mode_compiled_is_kept_and_says_so():
    got = plan(workload=Workload(max_temperature_K=300.0, host_callbacks_every=5), mode="compiled")
    assert got.tier == "compiled"
    reason = got.decision("mode")[2]
    assert reason.code == "mode-compiled-tier" and "requested" in reason and "rules 4 and 9" in reason
    numbered(got, "workload.host_callbacks_every", 9, "host-callbacks-compiled")
    assert_conforms(got)


# --------------------------------------------------------------------------- rule 10


@pytest.mark.rule(10)
@pytest.mark.parametrize(("integrator", "stress", "headroom"), [
    (IntegratorConfig(), "off", 0.0),
    (IntegratorConfig(ensemble="npt"), "on", 0.1),
], ids=["fixed-cell", "variable-cell"])
def test_rule_10_stress_and_k_headroom_per_cell_mode(integrator, stress, headroom):
    got = plan("polar1-s", integrator=integrator)
    assert got.decision("compute_stress")[1] == stress and got.polar.kspace_headroom == headroom
    assert_conforms(got)


@pytest.mark.rule(10)
@pytest.mark.parametrize("cell_filter", [False, True], ids=["fixed-cell", "cell-filter"])
def test_rule_10_relaxations_take_the_stress_path_from_the_cell_filter(cell_filter):
    from fastmace import optimize

    optimizer = (optimize.LBFGS(cell_filter=optimize.FrechetCell()) if cell_filter else optimize.LBFGS()).to_config()
    got = plan(optimizer=optimizer)
    assert got.decision("compute_stress")[1] == ("on" if cell_filter else "off")
    numbered(got, "compute_stress", 10, "rule-10-optimizer")
    assert_conforms(got)


# --------------------------------------------------------------------------- rule 11


@pytest.mark.rule(11)
def test_rule_11_md_only_keeps_the_1a_skin_and_records_the_rules_value():
    got = plan(workload=Workload(max_temperature_K=2500.0))
    assert got.neighbours.skin == DEFAULT_SKIN_A
    reason = numbered(got, "NeighbourConfig.skin", 11, "rule-11-fixed")
    assert "rule 11 would give" in reason
    numbered(got, "workload.max_temperature_K", 11, "rule-11-md-only")
    numbered(got, "NeighbourConfig.headroom", 11, "rule-11-fixed")
    numbered(got, "EngineConfig.recapture_budget", 11, "rule-11")
    assert got.overrides() == ()
    assert_conforms(got)


@pytest.mark.rule(11)
def test_rule_11_sizes_mc_workloads():
    got = plan(workload=MC)
    assert got.neighbours.skin > DEFAULT_SKIN_A
    numbered(got, "NeighbourConfig.skin", 11, "rule-11")
    numbered(got, "NeighbourConfig.headroom", 11, "rule-11")
    assert got.overrides() == ()
    assert_conforms(got)


@pytest.mark.rule(11)
def test_rule_11_overrides_are_recorded():
    """An explicit skin, headroom and budget are kept, each as ``override; rule 11 would give X`` (choice 4)."""
    sized = plan(workload=MC)
    planner = tmp.minimal(neighbours=NeighbourConfig(skin=0.4, headroom="fixed", capacity_headroom=1.3),
                          engine=EngineConfig(recapture_budget=5))
    got = plan(workload=MC, planner=planner)
    assert (got.neighbours.skin, got.neighbours.capacity_headroom, got.engine.recapture_budget) == (0.4, 1.3, 5)
    overridden(got, "NeighbourConfig.skin", 11, "0.4 A", f"{sized.neighbours.skin:g} A")
    overridden(got, "NeighbourConfig.headroom", 11, "fixed",
               f"capacity_headroom {sized.neighbours.capacity_headroom:g}")
    overridden(got, "EngineConfig.recapture_budget", 11, "5", str(sized.engine.recapture_budget))
    requested = dict(got.requested)
    assert requested["NeighbourConfig.skin"] == "0.4" and requested["EngineConfig.recapture_budget"] == "5"
    assert len(got.overrides()) == 3
    assert_conforms(got)
    # The override survives the restart record (code and rule travel with the text).
    again = type(got).from_restart(got.to_restart(edge_bucket=64, neighbour_capacity=128))
    assert again.overrides() == got.overrides()


@pytest.mark.rule(11)
def test_rule_11_md_only_skin_override():
    got = plan(planner=tmp.minimal(neighbours=NeighbourConfig(skin=0.5)))
    assert got.neighbours.skin == 0.5
    overridden(got, "NeighbourConfig.skin", 11, "0.5 A", f"{DEFAULT_SKIN_A:g} A")
    assert_conforms(got)


@pytest.mark.rule(11)
def test_a_requested_value_equal_to_the_rules_is_no_override():
    got = plan(planner=tmp.minimal(neighbours=NeighbourConfig(skin=DEFAULT_SKIN_A),
                                   engine=EngineConfig(recapture_budget=64)))
    assert got.overrides() == ()
    reason = numbered(got, "NeighbourConfig.skin", 11, "rule-11-fixed")
    assert str(reason).startswith("requested")
    numbered(got, "EngineConfig.recapture_budget", 11, "rule-11")
    assert_conforms(got)


@pytest.mark.rule(11)
def test_rule_11_d3_skin_default_and_override():
    sized = plan(d3_cutoff_A=D3_CUTOFF)
    numbered(sized, "NeighbourConfig.d3_skin", 11, "rule-11-d3")
    got = plan(planner=tmp.minimal(neighbours=NeighbourConfig(d3_skin=2.0)), d3_cutoff_A=D3_CUTOFF)
    assert got.neighbours.d3_skin == 2.0
    overridden(got, "NeighbourConfig.d3_skin", 11, "2 A", f"{sized.neighbours.d3_skin:g} A")
    assert dict(got.requested)["NeighbourConfig.d3_skin"] == "2.0"
    assert_conforms(got)
    same = plan(planner=tmp.minimal(neighbours=NeighbourConfig(d3_skin=sized.neighbours.d3_skin)),
                d3_cutoff_A=D3_CUTOFF)
    assert same.overrides() == () and same.decision("NeighbourConfig.d3_skin")[2].code == "rule-11-d3"


@pytest.mark.rule(11)
def test_rule_11_relaxation_skin_override():
    from fastmace import optimize

    optimizer = optimize.LBFGS().to_config()
    sized = plan(optimizer=optimizer)
    numbered(sized, "NeighbourConfig.skin", 11, "rule-11-optimizer")
    got = plan(optimizer=optimizer, planner=tmp.minimal(neighbours=NeighbourConfig(skin=0.45)))
    overridden(got, "NeighbourConfig.skin", 11, "0.45 A", f"{sized.neighbours.skin:g} A")


# --------------------------------------------------------------------------- rule 12 (the D3 list, 3b-T02)


@pytest.mark.rule(12)
def test_rule_12_d3_list_fits():
    got = plan(d3_cutoff_A=D3_CUTOFF)
    assert got.decision("d3 list memory")[1] == "fits"
    numbered(got, "d3 list memory", 12, "rule-12-d3")
    assert_conforms(got)


@pytest.mark.rule(12)
def test_rule_12_d3_list_warns_near_the_gpu_memory():
    small = dataclasses.replace(tmp.GPU, memory_bytes=2**24)
    with pytest.warns(ResourceWarning, match="rule 12"):
        got = tmp.minimal().plan_traits(tmp.tp.traits("mh1-r2scan"), nacl(), IntegratorConfig(), WORKLOAD, small,
                                        precision="float32", d3_cutoff_A=D3_CUTOFF)
    assert got.decision("d3 list memory")[1] == "warn"
    numbered(got, "d3 list memory", 12, "rule-12-d3-warn")
    assert_conforms(got)


@pytest.mark.rule(12)
def test_rule_12_d3_list_refuses_above_the_free_memory():
    """Tested as it stands: §9.3 says only hard rules refuse, yet rule 12, a preference, refuses a D3 list whose
    buffers alone exceed the free memory (3b-T02). Owner decision D1 (pending) decides rule 12's action; the
    conductor's answer to 5-T02 question 4 keeps this behaviour until then."""
    tiny = dataclasses.replace(tmp.GPU, memory_bytes=2**18)
    planner = tmp.minimal()
    planner._free_memory_bytes = lambda hardware: hardware.memory_bytes  # as on CUDA before initialisation
    with pytest.warns(ResourceWarning):
        error = refused(planner.plan_traits, tmp.tp.traits("mh1-r2scan"), nacl(), IntegratorConfig(), WORKLOAD,
                        tiny, precision="float32", d3_cutoff_A=D3_CUTOFF)
    assert codes(error) == ["rule-12-d3-memory"] and error.reasons[0].rule == 12


# --------------------------------------------------------------------------- overrides in general


def test_every_preference_takes_an_override_and_nothing_else_does():
    assert [n for n in rules.NUMBERS if rules.kind(n) == "preference"] == [8, 9, 10, 11, 12]
    for number in rules.numbers("preference"):
        assert rules.check_reason(rules.override(number, "x")) == ()
    for number in (*rules.numbers("hard"), *rules.numbers("reaction")):
        with pytest.raises(ValueError, match="only preferences"):
            rules.override(number, "x")
