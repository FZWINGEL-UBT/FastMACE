"""The fallback search (SPEC §9.4 in full, §9.3 rules 13-14, §9.5; plan task 2-T15, decision 29).

- ``ddmin`` finds a 1-minimal failing subset of the rewrite set under the ``requires`` closure: a
  single culprit, an interacting pair, a culprit that others require (they go with it) and a
  culprit that requires another (only the culprit is blamed); each call counts against the budget;
- ``diagnose`` names the failing dimension: ``"mode"`` when eager passes, ``"rewrites"`` when eager
  fails and the empty rewrite set passes, ``None`` when the reduced gate cannot locate the fault;
- ``terminal_sequence`` is the same mode with every rewrite off, eager with every rewrite off, and
  the reference tier if ``reference_last`` allows it, without duplicates;
- ``Fallback.run`` on a tiny CPU model: a broken rewrite, a broken pair, a broken capture (graph,
  then compiled, then eager; 3-T06), a spent budget (the terminal sequence in order), exhaustion
  (``FallbackExhausted`` with the record), rule 13 on a self-check failure; a failing candidate is
  never activated, every candidate is built from the pristine model, and the budget counts reduced
  and full gate runs apart.

Faults are injected here only (monkeypatched builds and gates), never through a hook in the library.
The gate's numbers are stubbed in this file; ``test_fallback_simulation.py`` runs the real gate.
"""

from __future__ import annotations

import dataclasses

import pytest
import test_minimal_planner as tmp
import torch

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace.config import FallbackConfig, IntegratorConfig, RewriteConfig
from fastmace.errors import FallbackExhausted, GateFailed, PlanRefused, SelfCheckError
from fastmace.planner import fallback, minimal, selfcheck
from fastmace.planner.fallback import Budget, DiagnosisBudgetSpent

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

INTEGRATOR = IntegratorConfig(temperature_K=300.0)
NA_CL = (11, 17)


# --------------------------------------------------------------------------- ddmin on an oracle


def oracle(*culprits, calls=None):
    """``test(subset)``: fails when every member of one of ``culprits`` (sets) is in the subset."""
    groups = [frozenset(group) for group in culprits]

    def test(subset):
        if calls is not None:
            calls.append(frozenset(subset))
        return any(group <= subset for group in groups)

    return test


UNITS = tuple("abcdefgh")


def test_ddmin_finds_a_single_culprit_within_a_few_runs():
    calls, budget = [], Budget(limit=64)
    found = fallback.ddmin(UNITS, oracle({"f"}, calls=calls), requires={}, budget=budget)
    assert found == {"f"}
    assert budget.reduced_gate_runs == len(calls) <= 12
    assert len(set(calls)) == len(calls)  # no subset is tested twice


def test_ddmin_finds_a_pair_that_passes_alone():
    found = fallback.ddmin(UNITS, oracle({"b", "g"}), requires={}, budget=Budget(limit=64))
    assert found == {"b", "g"}


def test_ddmin_over_sixteen_records_fits_the_budget():
    units = tuple(f"r{i:02d}" for i in range(16))
    for culprits in ({"r03"}, {"r00", "r15"}, {"r07", "r08"}):
        budget = Budget(limit=64)
        assert fallback.ddmin(units, oracle(culprits), requires={}, budget=budget) == culprits
        assert budget.reduced_gate_runs <= 40


def test_ddmin_tests_closed_subsets_and_blames_the_requiring_culprit():
    """``b`` requires ``a``; the fault needs ``b`` (with ``a`` present): ``b`` is blamed, not ``a``."""
    requires = {"b": ("a",), "c": ("b",)}
    calls = []
    found = fallback.ddmin(tuple("abcd"), oracle({"b"}, calls=calls), requires=requires, budget=Budget(limit=64))
    assert found == {"b"}
    for subset in calls:  # every tested subset holds what its members require
        for unit in subset:
            assert set(requires.get(unit, ())) <= subset


def test_a_culprit_that_others_require_takes_them_along():
    requires = {"b": ("a",), "c": ("b",), "e": ("d",)}
    found = fallback.ddmin(tuple("abcde"), oracle({"a"}), requires=requires, budget=Budget(limit=64))
    assert found == {"a"}
    assert fallback.dependants(found, requires, tuple("abcde")) == ("b", "c")


def test_ddmin_stops_when_the_budget_is_spent():
    budget = Budget(limit=3)
    with pytest.raises(DiagnosisBudgetSpent):
        fallback.ddmin(tuple(f"r{i}" for i in range(16)), oracle({"r9"}), requires={}, budget=budget)
    assert budget.reduced_gate_runs == 3


def test_ddmin_without_a_budget_counts_nothing():
    assert fallback.ddmin(UNITS, oracle({"c"}), requires={}, budget=None) == {"c"}


def test_ddmin_of_one_unit_is_that_unit():
    calls = []
    assert fallback.ddmin(("a",), oracle({"a"}, calls=calls), requires={}, budget=Budget(limit=1)) == {"a"}
    assert calls == []  # the caller knows the whole set fails


# --------------------------------------------------------------------------- diagnose, terminal sequence


def test_diagnose_names_the_dimension():
    graph = minimal_plan_stub("graph")
    assert fallback.diagnose(graph, None, test=lambda mode, units: False) == "mode"
    assert fallback.diagnose(graph, None, test=lambda mode, units: bool(units)) == "rewrites"
    assert fallback.diagnose(graph, None, test=lambda mode, units: True) is None
    eager = minimal_plan_stub("eager")
    assert fallback.diagnose(eager, None, test=lambda mode, units: False) is None  # the reduced gate cannot see it


def minimal_plan_stub(tier):
    from fastmace.planner.plan import Plan

    return Plan(tier=tier, rewrites=RewriteConfig(block_linear="on", grouped_linear="on"))


@pytest.mark.parametrize(("tier", "reference", "expected"), [
    ("graph", True, ("graph", "eager", "reference")),
    ("graph", False, ("graph", "eager")),
    ("eager", True, ("eager", "reference")),
    ("reference", True, ("reference", "eager")),
])
def test_terminal_sequence_in_order_without_duplicates(tier, reference, expected):
    steps = fallback.terminal_sequence(minimal_plan_stub(tier), reference=reference)
    assert tuple(mode for mode, _ in steps) == expected
    assert all(label for _, label in steps)


# --------------------------------------------------------------------------- Fallback.run on a tiny model

# factor_species and block_linear on the CPU switch on their chains (registry ``requires``):
# fold_cutoff, density_first, joint_radial_input, fused_folded_backward, grouped_linear,
# merged_residual, plus the defaults single_graph_strain and polar_uvu.
REWRITES = RewriteConfig(factor_species="on", block_linear="on")


@pytest.fixture
def pristine(tmp_path, quiet_licences):
    from fastmace.model import loading

    model = tiny_mace_model(list(NA_CL), **FAMILIES["mh1"])
    path = tmp_path / "tiny-mh1.model"
    torch.save(model, path)
    return loading.load_pristine(str(path), trusted=True)


quiet_licences = tmp.quiet_licences


class Faults:
    """Injected faults: rewrite sets and modes the stubbed gate fails, and builds that raise.

    ``bad``: frozensets of rewrite fields; the gate fails (force) on an engine whose plan has all of
    one set on. ``bad_modes``: tiers whose build raises a capture error. ``built``: every plan built.
    """

    def __init__(self, monkeypatch, *, bad=(), bad_modes=(), fail_all=False):
        self.bad = [frozenset(group) for group in bad]
        self.bad_modes = set(bad_modes)
        self.fail_all = fail_all
        self.built, self.gates, self.engines = [], [], []
        original = minimal.MinimalPlanner.build
        faults = self

        def build(planner, plan, force_field, atoms, **kwargs):
            faults.built.append(plan)
            if plan.tier in faults.bad_modes:
                raise RuntimeError("CUDA graph capture of stage B failed (injected)")
            engine = original(planner, plan, force_field, atoms, **kwargs)
            engine._test_plan = plan
            faults.engines.append(engine)
            return engine

        def production(engine, references, *, check, quantities=None):
            plan = engine._test_plan
            faults.gates.append((plan, tuple(quantities or references.quantities), len(references.copies)))
            on = frozenset(fallback.enabled_rewrites(plan))
            if faults.fail_all or any(group <= on for group in faults.bad):
                record = selfcheck.GateRecord(status="failed", check=check, reason="force error 3e-3 eV/Å above "
                                              "the limit (injected)",
                                              quantities={"force_eV_per_A": {"passed": False}})
                raise GateFailed("the gate on this structure failed (injected)", gate_record=record)
            return selfcheck.GateRecord(status="passed", check=check, reason=None)

        monkeypatch.setattr(minimal.MinimalPlanner, "build", build)
        monkeypatch.setattr(selfcheck, "production", production)


def stub_references(check="gate", copies=2):
    return selfcheck.References(check=check, copies=[object()] * copies, values=[{}] * copies,
                                reference_copies=[object()] * copies, ran="in-process")


def setup_case(pristine, rewrites=REWRITES):
    from fastmace.model.forcefield import ForceField

    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    ff = ForceField(pristine, atoms, precision="float64", rewrites=rewrites)
    planner = tmp.cpu_minimal()
    plan = planner.plan(ff, atoms, INTEGRATOR, tmp.WORKLOAD, minimal_hardware())
    return planner, ff, atoms, plan


def minimal_hardware():
    import test_planner as tp

    return tp.CPU


def gate_failure(plan, quantity="force_eV_per_A"):
    record = selfcheck.GateRecord(status="failed", check="gate", reason="force error above the limit",
                                  quantities={quantity: {"passed": False}})
    error = GateFailed("the gate on this structure failed", gate_record=record)
    return fallback.Failure(plan=plan, stage="gate", error=error, gate=record)


def run_fallback(planner, ff, atoms, failure, references=None, config=None):
    search = fallback.Fallback(planner, ff, atoms, INTEGRATOR, references or stub_references(), config)
    return search, search.run(failure)


def off_fields(plan):
    return {f.name for f in dataclasses.fields(plan.rewrites) if getattr(plan.rewrites, f.name) in ("off", False)}


def test_the_rewrite_set_and_its_requires(pristine):
    planner, ff, atoms, plan = setup_case(pristine)
    units = fallback.enabled_rewrites(plan)
    assert {"factor_species", "fold_cutoff", "block_linear", "grouped_linear", "merged_residual"} <= set(units)
    requires = fallback.field_requires(units)
    assert requires["fold_cutoff"] == ("factor_species",)
    assert set(requires["merged_residual"]) == {"block_linear", "density_first"}


def test_a_broken_rewrite_is_found_and_disabled_with_its_dependants(pristine, monkeypatch):
    planner, ff, atoms, plan = setup_case(pristine)
    faults = Faults(monkeypatch, bad=[{"block_linear"}])
    search, result = run_fallback(planner, ff, atoms, gate_failure(plan))
    active = result.plan
    assert active.revision == plan.revision + 1 and active.predecessor is plan and active.tier == "graph"
    assert {"block_linear", "grouped_linear", "merged_residual"} <= off_fields(active)
    assert "factor_species" not in off_fields(active) and "fold_cutoff" not in off_fields(active)
    assert active.reason.startswith("fallback: rewrites ") and "ddmin" in active.reason
    assert result.engine is faults.engines[-1] and result.gate.passed
    # the reduced gate judged one copy and the failing quantity; the activation the full gate
    reduced = [entry for entry in faults.gates[:-1]]
    assert all(quantities == ("force_eV_per_A",) and copies == 1 for _, quantities, copies in reduced)
    assert faults.gates[-1][1:] == (selfcheck.QUANTITIES, 2)
    assert result.budget.full_gate_runs == 1 and 0 < result.budget.reduced_gate_runs <= 64
    assert result.budget.reduced_gate_runs == sum(t.reduced_gate_runs for t in active.fallbacks)
    assert active.fallbacks[-1].outcome == "activated" and active.fallbacks[-1].revision == active.revision
    assert [t.outcome for t in active.fallbacks[:-1]] and all(
        t.outcome in ("passed", "failed") for t in active.fallbacks[:-1])
    # every candidate was built in the mode its test needs, the diagnosis in eager
    assert {p.tier for p in faults.built[:-1]} == {"eager"} and faults.built[-1].tier == "graph"


@pytest.mark.slow
def test_a_broken_pair_that_passes_alone_is_found(pristine, monkeypatch):
    planner, ff, atoms, plan = setup_case(pristine)
    Faults(monkeypatch, bad=[{"joint_radial_input", "grouped_linear"}])
    _, result = run_fallback(planner, ff, atoms, gate_failure(plan))
    off = off_fields(result.plan)
    assert {"joint_radial_input", "grouped_linear"} <= off
    assert "block_linear" not in off and "fold_cutoff" not in off and "factor_species" not in off
    assert "joint_radial_input, grouped_linear" in result.plan.reason or "grouped_linear, joint_radial_input" in \
        result.plan.reason


@pytest.mark.slow
def test_a_culprit_required_by_others_takes_them_along(pristine, monkeypatch):
    planner, ff, atoms, plan = setup_case(pristine)
    Faults(monkeypatch, bad=[{"fold_cutoff"}])
    _, result = run_fallback(planner, ff, atoms, gate_failure(plan))
    off = off_fields(result.plan)
    assert {"fold_cutoff", "density_first", "joint_radial_input", "fused_folded_backward", "merged_residual"} <= off
    assert "factor_species" not in off and "block_linear" not in off and "grouped_linear" not in off


def test_a_broken_capture_drops_the_graph_mode(pristine, monkeypatch):
    """3-T06: the mode dimension is graph -> compiled -> eager (§9.4 step 2): compiled first."""
    planner, ff, atoms, plan = setup_case(pristine)
    faults = Faults(monkeypatch, bad_modes={"graph"})
    error = RuntimeError("CUDA graph capture of stage B failed (injected)")
    _, result = run_fallback(planner, ff, atoms, fallback.Failure(plan=plan, stage="capture", error=error))
    active = result.plan
    assert active.tier == "compiled" and active.revision == plan.revision + 1
    assert active.reason.startswith("fallback: mode compiled: ") and fallback.MODE_WHY in active.reason
    assert active.rewrites == plan.rewrites
    assert [t.dimension for t in active.fallbacks] == ["mode", "mode"]
    assert [t.outcome for t in active.fallbacks] == ["passed", "activated"]
    assert result.budget.reduced_gate_runs == 1 and result.budget.full_gate_runs == 1
    assert [p.tier for p in faults.built] == ["eager", "compiled"]


def test_a_broken_compilation_drops_to_eager_after_compiled(pristine, monkeypatch):
    planner, ff, atoms, plan = setup_case(pristine)
    faults = Faults(monkeypatch, bad_modes={"graph", "compiled"})
    error = RuntimeError("CUDA graph capture of stage B failed (injected)")
    _, result = run_fallback(planner, ff, atoms, fallback.Failure(plan=plan, stage="capture", error=error))
    active = result.plan
    assert active.tier == "eager" and active.reason.startswith("fallback: mode eager: ")
    assert [p.tier for p in faults.built] == ["eager", "compiled", "eager"]
    assert [(t.dimension, t.outcome) for t in active.fallbacks] == [
        ("mode", "passed"), ("mode", "failed"), ("mode", "activated")]
    assert result.budget.reduced_gate_runs == 1


def test_a_failing_compiled_plan_drops_to_eager(pristine, monkeypatch):
    planner, ff, atoms, _ = setup_case(pristine)
    plan = planner.plan(ff, atoms, INTEGRATOR, tmp.WORKLOAD, minimal_hardware(), mode="compiled")
    faults = Faults(monkeypatch, bad_modes={"compiled"})
    error = RuntimeError("BackendCompilerFailed: backend='inductor' raised (injected)")
    _, result = run_fallback(planner, ff, atoms, fallback.Failure(plan=plan, stage="compile", error=error))
    assert result.plan.tier == "eager"
    assert [p.tier for p in faults.built] == ["eager", "eager"]


def test_a_spent_budget_runs_the_terminal_sequence_in_order(pristine, monkeypatch):
    planner, ff, atoms, plan = setup_case(pristine)
    faults = Faults(monkeypatch, bad=[{"merged_residual"}], bad_modes={"graph"})
    _, result = run_fallback(planner, ff, atoms, gate_failure(plan), config=FallbackConfig(diagnosis_budget=2))
    # the eager re-run fails (merged_residual), the empty set passes, ddmin has no budget left:
    # terminal step 1 (graph, rewrites off) fails at capture, step 2 (eager, rewrites off) passes
    active = result.plan
    assert active.tier == "eager"
    assert not fallback.enabled_rewrites(active) or set(fallback.enabled_rewrites(active)) <= set()
    assert result.budget.reduced_gate_runs == 2
    terminal = [t for t in active.fallbacks if t.full_gate_runs or t.stage == "build"]
    assert [t.outcome for t in terminal] == ["failed", "activated"]
    assert terminal[0].stage == "build" and "capture" in terminal[0].reason
    built_tiers = [p.tier for p in faults.built]
    assert built_tiers[-2:] == ["graph", "eager"]


def test_nothing_passes_raises_fallback_exhausted_with_the_record(pristine, monkeypatch):
    planner, ff, atoms, plan = setup_case(pristine)
    faults = Faults(monkeypatch, fail_all=True)
    with pytest.raises(FallbackExhausted) as caught:
        run_fallback(planner, ff, atoms, gate_failure(plan))
    error = caught.value
    assert isinstance(error, PlanRefused)
    assert error.revisions[0] is plan and len(error.revisions) >= 2
    assert error.tried and all(t.outcome != "activated" for t in error.tried)
    assert any("reference" in t.change for t in error.tried)  # the reference tier was tried or refused
    assert "force error" in str(error)
    assert error.budget.reduced_gate_runs >= 1
    assert all(not getattr(e, "_active", False) for e in faults.engines)


def test_the_reference_tier_ends_the_sequence_only_if_it_fits(pristine, monkeypatch):
    planner, ff, atoms, plan = setup_case(pristine)
    faults = Faults(monkeypatch, bad_modes={"graph", "eager"})
    error = RuntimeError("CUDA graph capture of stage B failed (injected)")
    _, result = run_fallback(planner, ff, atoms, fallback.Failure(plan=plan, stage="capture", error=error))
    assert result.plan.tier == "reference" and faults.built[-1].tier == "reference"
    assert not fallback.enabled_rewrites(result.plan)
    assert "terminal sequence: the reference tier" in result.plan.reason

    monkeypatch.setattr(type(planner), "reference_fits", lambda self, *args: (False, "18.6 GB > 16 GB (injected)"))
    with pytest.raises(FallbackExhausted) as caught:
        run_fallback(planner, ff, atoms, fallback.Failure(plan=plan, stage="capture", error=error))
    last = caught.value.tried[-1]
    assert (last.change, last.outcome) == ("mode reference", "skipped") and "does not fit: 18.6 GB" in last.reason


def test_a_failing_candidate_is_never_activated(pristine, monkeypatch):
    planner, ff, atoms, plan = setup_case(pristine)
    Faults(monkeypatch, bad=[{"block_linear"}], bad_modes={"graph"})
    calls = []
    original = fallback.activate

    def spy(candidate, *args, **kwargs):
        activation = original(candidate, *args, **kwargs)
        calls.append(activation)
        return activation

    monkeypatch.setattr(fallback, "activate", spy)
    _, result = run_fallback(planner, ff, atoms, gate_failure(plan))
    # ddmin's candidate (graph, block_linear and its dependants off) fails at capture and is never
    # activated; the next round finds eager passing and activates compiled first (3-T06)
    assert [a.activated for a in calls] == [False, True]
    assert calls[0].engine is None and calls[0].tried.stage == "build" and result.plan.tier == "compiled"
    assert {"block_linear", "grouped_linear", "merged_residual"} <= off_fields(result.plan)
    assert result.plan.revision == calls[-1].plan.revision and result.engine is calls[-1].engine


def test_candidates_are_built_from_the_pristine_model(pristine, monkeypatch):
    planner, ff, atoms, plan = setup_case(pristine)
    before = tmp._parameters(ff.model)
    faults = Faults(monkeypatch, bad_modes={"graph"})
    error = RuntimeError("CUDA graph capture of stage B failed (injected)")
    run_fallback(planner, ff, atoms, fallback.Failure(plan=plan, stage="capture", error=error))
    assert len(faults.engines) == 2
    models = [engine.force_field.source_model for engine in faults.engines]
    assert len({id(model) for model in models}) == len(models)
    assert all(model is not ff.model.model for model in models)
    after = tmp._parameters(ff.model)
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_a_self_check_failure_is_rule_13(pristine, monkeypatch):
    planner, ff, atoms, plan = setup_case(pristine)
    faults = Faults(monkeypatch)
    from fastmace.rewrites import registry as reg

    reaction = reg.default_registry().disable_with_dependants(
        _resolution_stub(plan), "block_linear", "relative error 3e-3 > 1e-5")
    error = SelfCheckError("rewrite block_linear failed its self-check: relative error 3e-3 > 1e-5")
    error.reaction = reaction
    _, result = run_fallback(planner, ff, atoms, fallback.Failure(plan=plan, stage="self-check", error=error),
                             references=stub_references("skip", copies=0))
    active = result.plan
    assert {"block_linear", "grouped_linear", "merged_residual"} <= off_fields(active)
    assert "rule 13" in active.reason
    assert result.budget.reduced_gate_runs == 0 and result.budget.full_gate_runs == 1
    assert faults.gates[-1][2] == 1  # validated by a reduced gate (one copy) under check="skip"


def _resolution_stub(plan):
    from fastmace.rewrites import registry as reg

    registry = reg.default_registry()
    enabled = set(fallback.enabled_rewrites(plan))
    decisions = tuple(reg.Decision(name=name, requested="auto", enabled=registry.config_field(name) in enabled,
                                   reason="stub") for name in registry.order())
    return reg.Resolution(decisions)


@pytest.fixture
def skip_references(monkeypatch):
    calls = []

    def references(plan, force_field, atoms, integrator, *, check, seed, gate=None, count=2, quantities=None):
        calls.append((check, count))
        return stub_references(check, copies=count)

    monkeypatch.setattr(selfcheck, "references", references)
    return calls


def test_under_check_skip_the_references_are_computed_once(pristine, monkeypatch, skip_references):
    planner, ff, atoms, plan = setup_case(pristine)
    Faults(monkeypatch, bad_modes={"graph"})
    error = RuntimeError("CUDA graph capture of stage B failed (injected)")
    _, result = run_fallback(planner, ff, atoms, fallback.Failure(plan=plan, stage="capture", error=error),
                             references=selfcheck.References(check="skip"))
    assert result.plan.tier == "compiled"
    assert skip_references == [("gate", 1)]


def test_refuse_raises_the_original_error(pristine, monkeypatch):
    planner, ff, atoms, plan = setup_case(pristine)
    failure = gate_failure(plan)
    with pytest.raises(GateFailed) as caught:
        run_fallback(planner, ff, atoms, failure, config=FallbackConfig(fallback="refuse"))
    assert caught.value is failure.error
