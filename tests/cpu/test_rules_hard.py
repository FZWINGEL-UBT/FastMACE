"""Conformance of the hard rules 1-7 of SPEC §9.3 (plan task 5-T02, phase 5; conductor's choice 1).

- Each hard rule refuses with its code and number, alone (one ``PlanRefused`` with that one reason).
- Several hard rules together give one ``PlanRefused`` with every reason, each with its number.
- Each ``allow_*`` flag lifts its own rule only, and is recorded (``Plan.allowances``, a rule-numbered
  decision); no flag lifts another rule.
- Every reason of every plan and refusal here uses a code of its rule (``rules.check_reason``).

The tiny model's traits (``test_planner.traits``) with injected ``SoftwareTraits`` and ``Hardware``: no GPU.
"""

import dataclasses

import pytest
import test_minimal_planner as tmp
from ase.constraints import FixAtoms, FixBondLength

from _support.rule_cases import assert_conforms, by_rule, codes, refused
from _support.structures import nacl
from fastmace.config import EngineConfig, IntegratorConfig
from fastmace.errors import SpeciesNotInSuperset
from fastmace.mc import Swap, Transmute
from fastmace.model.capabilities import RewriteSupport
from fastmace.planner import minimal as minimal_module
from fastmace.planner import reproducible, rules
from fastmace.planner.workload import Workload
from fastmace.rewrites import versions

NA, CL = 11, 17
WORKLOAD = tmp.WORKLOAD
REPRODUCIBLE = Workload(max_temperature_K=300.0, reproducible=True)
PINNED = {reproducible.CUEQ_OPS: reproducible.PINNED_CUEQ_OPS}
NAIVE = tmp.software(fused_cueq=(False, "cuequivariance_ops_torch cannot be imported"))


@pytest.fixture(autouse=True)
def _cuda_not_initialised(monkeypatch):
    monkeypatch.setattr(minimal_module, "_cuda_initialised", lambda: False)


def plan(label="mh1-r2scan", **kwargs):
    return tmp.plan_label(label, **kwargs)


def slab():
    atoms = nacl()
    atoms.pbc = (True, True, False)
    return atoms


def fixed_atom(atoms=None):
    atoms = atoms if atoms is not None else nacl()
    atoms.set_constraint(FixAtoms(indices=[0]))
    return atoms


def bond(atoms=None):
    atoms = atoms if atoms is not None else nacl()
    atoms.set_constraint(FixBondLength(0, 1))
    return atoms


def inexact(label="mh1-r2scan"):
    return dataclasses.replace(tmp.tp.traits(label), padding_is_exact=False)


def alone(error, rule, code):
    """``error`` holds exactly one reason: ``code`` of hard rule ``rule``."""
    assert codes(error) == [code], codes(error)
    assert error.reasons[0].rule == rule and rules.kind(rule) == "hard"
    assert rules.rule_of(code) == rule


# --------------------------------------------------------------------------- each hard rule alone


MC_SWAP = dict(species_changes=True, mc_moves=("swap",), min_mc_temperature_K=300.0)


@pytest.mark.rule(1)
@pytest.mark.parametrize(("code", "kwargs"), [
    ("non-3d-pbc", dict(atoms=slab())),
    ("constraints", dict(atoms=bond())),
    ("fix-atoms-with-fixcm", dict(atoms=fixed_atom())),  # the default fixcm=True (4-T04)
    ("variable-cell-without-barostat", dict(variable_cell=True)),
    ("mc-with-constraints", dict(atoms=fixed_atom(), integrator=IntegratorConfig(fixcm=False),
                                 workload=Workload(max_temperature_K=300.0, **MC_SWAP))),
])
def test_rule_1_refuses_a_workload_outside_section_4_4(code, kwargs):
    alone(refused(plan, **kwargs), 1, code)


@pytest.mark.rule(1)
def test_rule_1_refuses_a_polar_variable_cell_optimiser():
    from fastmace import optimize

    cell = optimize.LBFGS(cell_filter=optimize.FrechetCell()).to_config()
    alone(refused(plan, "polar1-s", optimizer=cell), 1, "polar-variable-cell-optimizer")


@pytest.mark.rule(1)
def test_rule_1_refuses_a_polar_model_on_an_unqualified_source(monkeypatch):
    """SPEC §9.3's rule 1 row names it (main b8ef2ae): PolarMACE without its hashed forward."""
    monkeypatch.setattr(versions, "polar_support", lambda: RewriteSupport(False, "unqualified source (test)"))
    alone(refused(plan, "polar1-s"), 1, "polar-unqualified-source")
    assert_conforms(plan("mh1-r2scan"))  # only PolarMACE asks


@pytest.mark.rule(2)
def test_rule_2_refuses_a_structure_species_outside_the_superset():
    error = refused(plan, species=["Na"])
    alone(error, 2, "species-not-in-superset")
    assert isinstance(error, SpeciesNotInSuperset) and "Cl" in str(error)


@pytest.mark.rule(2)
@pytest.mark.parametrize("move", [
    Swap([("Na", "K")], every=1, attempts=1),
    Transmute(["Na", "K"], every=1, attempts=1, velocities="redraw_all"),
], ids=["swap-pairs", "transmutable-set"])
def test_rule_2_refuses_a_move_set_species_outside_the_superset(move):
    """The move-set half of rule 2 (3-T09): swap pairs and transmutable sets."""
    error = refused(minimal_module.check_mc_moves, [move], superset=(NA, CL), charge_inputs=False, fixcm=False)
    alone(error, 2, "species-not-in-superset")
    assert isinstance(error, SpeciesNotInSuperset) and "K" in str(error)
    assert minimal_module.check_mc_moves([Swap([("Na", "Cl")], every=1, attempts=1)], superset=(NA, CL),
                                         charge_inputs=False, fixcm=False) == ()


@pytest.mark.rule(3)
def test_rule_3_refuses_polar_mace_on_plain_compiled_cueq():
    planner = tmp.minimal(engine=EngineConfig(backend="cueq"))
    alone(refused(plan, "polar1-s", planner=planner), 3, "polar-plain-compiled-cueq")
    # Hybrid, or eager plain cuEq: planned.
    assert_conforms(plan("polar1-s"))
    eager = plan("polar1-s", planner=tmp.minimal(engine=EngineConfig(backend="cueq", compile_mode="none")))
    assert eager.engine.backend == "cueq" and eager.engine.compile_mode == "none"
    assert_conforms(eager)


@pytest.mark.rule(4)
def test_rule_4_refuses_the_graph_tier_with_inexact_padding():
    error = refused(tmp.minimal().plan_traits, inexact(), nacl(), IntegratorConfig(), WORKLOAD, tmp.GPU,
                    precision="float32", mode="graph")
    alone(error, 4, "inexact-padding")
    # Without a mode, rule 4 plans the compiled tier with exact edge counts (3-T06).
    compiled = tmp.minimal().plan_traits(inexact(), nacl(), IntegratorConfig(), WORKLOAD, tmp.GPU, precision="float32")
    assert compiled.tier == "compiled"
    assert compiled.decision("mode")[2].code == "mode-compiled-rule-4" and compiled.decision("mode")[2].rule == 4
    assert compiled.decision("padding")[2].code == "compiled-exact-edges"
    assert_conforms(compiled)
    eager = tmp.minimal().plan_traits(inexact(), nacl(), IntegratorConfig(), WORKLOAD, tmp.GPU, precision="float32",
                                      mode="eager")
    assert eager.tier == "eager" and eager.decision("padding")[2].code == "eager-exact-edges"
    assert_conforms(eager)


@pytest.mark.rule(5)
def test_rule_5_refuses_an_unqualified_family():
    planner = tmp.minimal(versions=PINNED)
    alone(refused(plan, "mp0-medium", workload=REPRODUCIBLE, precision="float64", planner=planner), 5,
          "reproducible-family-not-qualified")


@pytest.mark.rule(5)
@pytest.mark.parametrize(("engine", "code"), [
    (EngineConfig(backend="hybrid"), "reproducible-backend"),
    (EngineConfig(backend="cueq", cueq_conv_fusion="on"), "reproducible-conv-fusion"),
])
def test_rule_5_refuses_an_explicit_conflict(engine, code):
    alone(refused(plan, workload=REPRODUCIBLE, precision="float64",
                  planner=tmp.minimal(versions=PINNED, engine=engine)), 5, code)


@pytest.mark.rule(5)
def test_rule_5_qualified_plan_records_its_rule():
    got = plan(workload=REPRODUCIBLE, precision="float64", planner=tmp.minimal(versions=PINNED))
    assert_conforms(got)
    for key, code in (("EngineConfig.backend", "backend-cueq-reproducible"), ("reproducible", "reproducible-qualified"),
                      ("reproducible.environment", "reproducible-environment")):
        reason = got.decision(key)[2]
        assert (reason.rule, reason.code) == (5, code), key


@pytest.mark.rule(6)
def test_rule_6_refuses_naive_cueq_kernels():
    planner = tmp.minimal(soft=NAIVE, engine=EngineConfig(backend="cueq"))
    alone(refused(plan, planner=planner), 6, "naive-cueq")
    fused = plan()
    assert fused.decision("allow_naive_cueq")[2].code == "fused-cueq" and fused.decision("allow_naive_cueq")[2].rule == 6


@pytest.mark.rule(7)
def test_rule_7_refuses_atom_count_changes():
    alone(refused(plan, workload=Workload(max_temperature_K=300.0, atom_count_changes=True)), 7,
          "atom-count-changes")


# --------------------------------------------------------------------------- together


@pytest.mark.rule(1)
@pytest.mark.rule(2)
@pytest.mark.rule(5)
@pytest.mark.rule(6)
@pytest.mark.rule(7)
def test_several_hard_rules_give_one_refusal_with_every_reason():
    """§9.3: when several hard constraints refuse, all reasons are reported together in one ``PlanRefused``."""
    planner = tmp.minimal(soft=NAIVE, versions=PINNED, engine=EngineConfig(backend="hybrid"))
    error = refused(plan, atoms=slab(), species=["Na"], planner=planner,
                    workload=Workload(max_temperature_K=300.0, reproducible=True, atom_count_changes=True))
    assert isinstance(error, SpeciesNotInSuperset)  # rule 2 is among them
    found = by_rule(error)
    assert found[1] == ["non-3d-pbc"] and found[2] == ["species-not-in-superset"]
    assert found[5] == ["reproducible-backend"] and found[6] == ["naive-cueq"] and found[7] == ["atom-count-changes"]
    assert set(found) <= set(rules.numbers("hard")), found


@pytest.mark.rule(3)
@pytest.mark.rule(4)
def test_rules_3_and_4_refuse_together():
    planner = tmp.minimal(engine=EngineConfig(backend="cueq"))
    error = refused(planner.plan_traits, inexact("polar1-s"), nacl(), IntegratorConfig(), WORKLOAD, tmp.GPU,
                    precision="float32", mode="graph")
    assert sorted(codes(error)) == ["inexact-padding", "polar-plain-compiled-cueq"]
    assert by_rule(error) == {3: ["polar-plain-compiled-cueq"], 4: ["inexact-padding"]}


# --------------------------------------------------------------------------- the allow_* flags


def test_only_rules_1_and_6_have_an_allow_flag():
    assert {n: rules.allow_flag(n) for n in rules.numbers("hard") if rules.allow_flag(n)} == {
        1: "allow_reference", 6: "allow_naive_cueq"}


@pytest.mark.rule(6)
def test_allow_naive_cueq_lifts_rule_6_only():
    planner = tmp.minimal(soft=NAIVE, engine=EngineConfig(backend="cueq"))
    lifted = plan(planner=planner, allow_naive_cueq=True)
    assert "allow_naive_cueq" in lifted.allowances
    choice, reason = lifted.decision("allow_naive_cueq")[1:]
    assert choice == "on" and (reason.rule, reason.code) == (6, "allow-naive-cueq")
    assert_conforms(lifted)
    # Every other hard rule still refuses.
    for kwargs, rule, code in ((dict(atoms=slab()), 1, "non-3d-pbc"), (dict(species=["Na"]), 2,
                                                                        "species-not-in-superset"),
                               (dict(workload=Workload(max_temperature_K=300.0, atom_count_changes=True)), 7,
                                "atom-count-changes")):
        alone(refused(plan, planner=planner, allow_naive_cueq=True, **kwargs), rule, code)
    alone(refused(plan, "polar1-s", planner=planner, allow_naive_cueq=True), 3, "polar-plain-compiled-cueq")
    alone(refused(plan, "mp0-medium", workload=REPRODUCIBLE, precision="float64", allow_naive_cueq=True,
                  planner=tmp.minimal(soft=NAIVE, versions=PINNED)), 5, "reproducible-family-not-qualified")


@pytest.mark.rule(1)
def test_allow_reference_is_recorded_and_lifts_no_other_rule():
    """``allow_reference`` lets the fallback reach the reference tier (rule 1's flag); it lifts no refusal of
    §4.4 and no other hard rule."""
    allowed = plan(allow_reference=True)
    assert "allow_reference" in allowed.allowances
    choice, reason = allowed.decision("allow_reference")[1:]
    assert choice == "on" and (reason.rule, reason.code) == (1, "allow-reference")
    assert_conforms(allowed)
    off = plan().decision("allow_reference")[2]
    assert (off.rule, off.code) == (1, "no-allow-reference")
    for kwargs, rule, code in ((dict(atoms=slab()), 1, "non-3d-pbc"), (dict(variable_cell=True), 1,
                                                                        "variable-cell-without-barostat"),
                               (dict(species=["Na"]), 2, "species-not-in-superset"),
                               (dict(workload=Workload(max_temperature_K=300.0, atom_count_changes=True)), 7,
                                "atom-count-changes")):
        alone(refused(plan, allow_reference=True, **kwargs), rule, code)
        alone(refused(plan, allow_reference=True, mode="reference", **kwargs), rule, code)
    alone(refused(plan, "polar1-s", allow_reference=True, planner=tmp.minimal(engine=EngineConfig(backend="cueq"))), 3,
          "polar-plain-compiled-cueq")
    alone(refused(plan, allow_reference=True, planner=tmp.minimal(soft=NAIVE, engine=EngineConfig(backend="cueq"))), 6,
          "naive-cueq")


def test_no_hard_rule_takes_an_override():
    """§9.3: a user override cannot lift a hard rule; only its ``allow_*`` flag does."""
    for number in rules.numbers("hard"):
        with pytest.raises(ValueError, match="only preferences"):
            rules.override(number, "anything")
