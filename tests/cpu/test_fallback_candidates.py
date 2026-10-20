"""Fallback groundwork: pristine candidates and validated activation (SPEC §9.4 steps 1 and 4, §9.5; 2-T06).

- ``candidate(plan, changes, force_field, atoms, *, planner=None)`` is a revision of ``plan`` with
  one dimension changed (the rewrite set, the mode or the backend), never the physics; the mode and
  backend re-plan through the planner, keeping the rewrites earlier fallbacks switched off.
- ``activate(candidate, planner, force_field, atoms, *, references, ...)`` builds the candidate from
  the pristine model in the mode it will run in (the self-checks run in the build), runs the gate on
  that engine, and only then makes it active; a failing candidate is recorded, never activated.
- Every tried candidate is a ``TriedCandidate`` in ``Plan.fallbacks``; ``PLAN_RECORD_VERSION`` 2
  encodes them, and version 1 records still read.

Faults are injected here only (monkeypatched builds and gates), never through a hook in the library.
"""

import dataclasses
import json

import pytest
import test_minimal_planner as tmp
import test_planner as tp
import torch

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace.config import IntegratorConfig
from fastmace.engine import constraints
from fastmace.errors import GateFailed, NonFiniteError, PlanRefused, ProcessConstraintError, SelfCheckError
from fastmace.planner import fallback, minimal, selfcheck
from fastmace.planner.fallback import Budget, TriedCandidate
from fastmace.planner.plan import PLAN_RECORD_VERSION, Plan

NA_CL = (11, 17)
INTEGRATOR = IntegratorConfig(temperature_K=300.0)


class StubForceField:
    """What ``MinimalPlanner.plan`` reads from an ``fm.ForceField``, for plan-level tests (no model)."""

    def __init__(self, label="mh1-r2scan", precision="float32", rewrites="auto"):
        self.traits = tp.traits(label)
        self.precision = precision
        self.rewrites = rewrites
        self.model = None

    def superset_for(self, atoms):
        return tuple(sorted({int(z) for z in atoms.numbers}))


def gpu_planner(process=None):
    planner = tmp.minimal(process=process or constraints.ProcessRecord())
    planner.hardware = lambda: tp.GPU
    return planner


def gpu_plan(planner=None, ff=None, atoms=None):
    planner = planner or gpu_planner()
    return planner, planner.plan(ff or StubForceField(), atoms or nacl(), INTEGRATOR, tmp.WORKLOAD, tp.GPU)


def enabled_fields(plan):
    return [f.name for f in dataclasses.fields(plan.rewrites) if getattr(plan.rewrites, f.name) in ("on", True)]


# --------------------------------------------------------------------------- candidate: the rewrite set


def test_rewrite_candidate_is_a_revision_with_the_fields_off():
    _, plan = gpu_plan()
    first, second = enabled_fields(plan)[:2]
    before = plan.to_restart(edge_bucket=1024, neighbour_capacity=2048)
    cand = fallback.candidate(plan, {"rewrites": {first: "off", second: "off"}, "reason": "ddmin culprit pair"},
                              StubForceField(), nacl())
    assert (cand.revision, cand.predecessor, cand.step) == (plan.revision + 1, plan, 0)
    assert cand.reason == f"fallback: rewrites {first}, {second} off: ddmin culprit pair"
    assert getattr(cand.rewrites, first) == "off" and getattr(cand.rewrites, second) == "off"
    key, choice, reason = cand.decision(f"RewriteConfig.{first}")
    assert choice == "off" and reason.code == fallback.REWRITE_OFF_CODE and "ddmin culprit pair" in str(reason)
    for field in enabled_fields(plan)[2:]:
        assert getattr(cand.rewrites, field) == getattr(plan.rewrites, field)
    for part in ("tier", "engine", "precision", "neighbours", "polar", "integrator", "workload", "superset",
                 "runner", "fallbacks"):
        assert getattr(cand, part) == getattr(plan, part), part
    assert plan.to_restart(edge_bucket=1024, neighbour_capacity=2048) == before  # the plan never changes


def test_rewrite_candidate_takes_a_rewrite_config():
    _, plan = gpu_plan()
    field = enabled_fields(plan)[0]
    off = False if isinstance(getattr(plan.rewrites, field), bool) else "off"
    cand = fallback.candidate(plan, {"rewrites": dataclasses.replace(plan.rewrites, **{field: off}),
                                     "reason": "terminal"}, StubForceField(), nacl())
    assert getattr(cand.rewrites, field) == off
    assert cand.reason == f"fallback: rewrites {field} off: terminal"


@pytest.mark.parametrize(("changes", "match"), [
    ({"reason": "x"}, "one of"),
    ({"rewrites": {"fold_cutoff": "off"}, "mode": "eager", "reason": "x"}, "one dimension"),
    ({"rewrites": {"fold_cutoff": "off"}}, "reason"),
    ({"precision": "float64", "reason": "x"}, "precision"),
    ({"rewrites": {"no_such_rewrite": "off"}, "reason": "x"}, "no_such_rewrite"),
    ({"rewrites": {"selfcheck_cache": "off"}, "reason": "x"}, "not a rewrite switch"),
    ({"rewrites": {"fold_cutoff": "auto"}, "reason": "x"}, "resolved"),
    ({"rewrites": {}, "reason": "x"}, "changes nothing"),
    ({"mode": "warp", "reason": "x"}, "mode"),
    ({"runner": "subprocess", "reason": "x"}, "backend"),
])
def test_candidate_changes_exactly_one_dimension(changes, match):
    _, plan = gpu_plan()
    with pytest.raises(ValueError, match=match):
        fallback.candidate(plan, changes, StubForceField(), nacl(), planner=gpu_planner())


def test_rewrite_candidate_of_an_unchanged_field_is_refused():
    _, plan = gpu_plan()
    field = enabled_fields(plan)[0]
    with pytest.raises(ValueError, match="changes nothing"):
        fallback.candidate(plan, {"rewrites": {field: getattr(plan.rewrites, field)}, "reason": "x"},
                           StubForceField(), nacl())


# --------------------------------------------------------------------------- candidate: mode and backend


def test_mode_and_backend_candidates_need_the_planner():
    _, plan = gpu_plan()
    for changes in ({"mode": "eager", "reason": "x"}, {"backend": "cueq", "reason": "x"}):
        with pytest.raises(ValueError, match="planner"):
            fallback.candidate(plan, changes, StubForceField(), nacl())


def test_mode_candidate_replans_in_that_mode_and_keeps_earlier_fallbacks(monkeypatch):
    """The mode re-plans through the planner (2-T02's eager mode), with the predecessor's off rewrites kept."""
    planner, plan = gpu_plan()
    field = enabled_fields(plan)[0]
    off = fallback.candidate(plan, {"rewrites": {field: "off"}, "reason": "ddmin"}, StubForceField(), nacl())
    calls = []
    original = type(planner).plan

    def eager_as_graph(self, force_field, atoms, integrator, workload, hardware=None, **flags):
        calls.append((integrator, workload, dict(flags)))  # stands in for 2-T02's eager mode on any main
        flags["mode"] = "graph"
        planned = original(self, force_field, atoms, integrator, workload, hardware, **flags)
        return dataclasses.replace(planned, tier="eager")

    monkeypatch.setattr(type(planner), "plan", eager_as_graph)
    cand = fallback.candidate(off, {"mode": "eager", "reason": "eager passes, capture fails"}, StubForceField(),
                              nacl(), planner=planner)
    (integrator, workload, flags), = calls
    assert integrator == plan.integrator and workload == plan.workload and flags["mode"] == "eager"
    assert cand.tier == "eager" and cand.predecessor is off and cand.revision == off.revision + 1
    assert cand.reason == "fallback: mode eager: eager passes, capture fails"
    assert getattr(cand.rewrites, field) == "off"
    assert cand.decision(f"RewriteConfig.{field}") == off.decision(f"RewriteConfig.{field}")
    assert cand.superset == plan.superset and cand.precision.dtype == plan.precision.dtype


def test_mode_candidate_uses_the_planners_eager_mode():
    planner, plan = gpu_plan()
    changes = {"mode": "eager", "reason": "capture failed"}
    if "eager" in minimal.MODE_REFUSALS:  # before 2-T02 the planner refuses the mode; the refusal propagates
        with pytest.raises(PlanRefused) as caught:
            fallback.candidate(plan, changes, StubForceField(), nacl(), planner=planner)
        assert "mode-eager" in tmp.codes(caught.value)
    else:
        cand = fallback.candidate(plan, changes, StubForceField(), nacl(), planner=planner)
        assert cand.tier == "eager" and cand.decision("mode")[1] == "eager"


def test_backend_candidate_in_a_child_claims_no_family():
    """§9.4 step 3: plain cuEq after a hybrid compile runs in a child (``runner="subprocess"``)."""
    process = constraints.ProcessRecord()
    planner, plan = gpu_plan(gpu_planner(process))
    assert plan.engine.backend == "hybrid" and process.family == constraints.OEQ_FAMILY
    cand = fallback.candidate(plan, {"backend": "cueq", "runner": "subprocess", "reason": "hybrid capture fails"},
                              StubForceField(), nacl(), planner=planner)
    assert cand.engine.backend == "cueq" and cand.runner == "subprocess"
    assert cand.decision("runner")[1] == "subprocess"
    assert cand.reason == "fallback: backend cueq (subprocess): hybrid capture fails"
    assert process.family == constraints.OEQ_FAMILY  # the child plans for itself
    assert planner.engine.backend == "auto"  # the caller's planner is not changed


def test_backend_candidate_in_process_respects_the_family():
    planner, plan = gpu_plan()
    try:
        cand = fallback.candidate(plan, {"backend": "cueq", "reason": "x"}, StubForceField(), nacl(),
                                  planner=planner)
    except ProcessConstraintError:  # main before 2-T07: the error goes to 2-T15, which routes it to the runner
        return
    assert cand.runner == "subprocess"  # 2-T07's planner offers the runner itself


def test_unchanged_backend_or_mode_is_refused():
    planner, plan = gpu_plan()
    with pytest.raises(ValueError, match="changes nothing"):
        fallback.candidate(plan, {"backend": "hybrid", "reason": "x"}, StubForceField(), nacl(), planner=planner)
    with pytest.raises(ValueError, match="changes nothing"):
        fallback.candidate(plan, {"mode": "graph", "reason": "x"}, StubForceField(), nacl(), planner=planner)


def test_a_replanned_candidate_never_changes_the_physics(monkeypatch):
    planner, plan = gpu_plan()
    original = type(planner).plan

    def other_physics(self, *args, **kwargs):
        planned = original(self, *args, **{**kwargs, "mode": "graph"})
        return dataclasses.replace(planned, tier="eager", integrator=IntegratorConfig(temperature_K=500.0))

    monkeypatch.setattr(type(planner), "plan", other_physics)
    with pytest.raises(ValueError, match="physics"):
        fallback.candidate(plan, {"mode": "eager", "reason": "x"}, StubForceField(), nacl(), planner=planner)


# --------------------------------------------------------------------------- activate


@pytest.fixture
def pristine(tmp_path, quiet_licences):
    from fastmace.model import loading

    model = tiny_mace_model(list(NA_CL), **FAMILIES["mh1"])
    path = tmp_path / "tiny-mh1.model"
    torch.save(model, path)
    return loading.load_pristine(str(path), trusted=True)


quiet_licences = tmp.quiet_licences


@pytest.fixture
def cpu_case(pristine):
    pytest.importorskip("cuequivariance_torch")
    from fastmace.model.forcefield import ForceField

    atoms = nacl(rattle=0.05, seed=1)
    ff = ForceField(pristine, atoms, precision="float64")
    planner = tmp.cpu_minimal()
    plan = planner.plan(ff, atoms, INTEGRATOR, tmp.WORKLOAD, tp.CPU)
    field = enabled_fields(plan)[0]
    cand = fallback.candidate(plan, {"rewrites": {field: "off"}, "reason": "ddmin"}, ff, atoms)
    return planner, ff, atoms, plan, cand


def passed_record(check="gate"):
    return selfcheck.GateRecord(status="passed", check=check, reason="stub gate passed")


def references(check="gate"):
    return selfcheck.References(check=check, copies=[object()], ran="in-process")


def gate_calls(monkeypatch, result=None, error=None):
    calls = []

    def production(engine, refs, *, check, **kwargs):
        calls.append((engine, refs, check, kwargs))
        if error is not None:
            raise error
        return result or passed_record(check)

    monkeypatch.setattr(selfcheck, "production", production)
    return calls


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_activate_builds_from_the_pristine_model_and_gates_before_activation(cpu_case, monkeypatch):
    planner, ff, atoms, plan, cand = cpu_case
    before = tmp._parameters(ff.model)
    failed_engine = planner.build(plan, ff, atoms)
    calls = gate_calls(monkeypatch)
    refs = references()
    budget = Budget(limit=64)
    activation = fallback.activate(cand, planner, ff, atoms, references=refs, budget=budget)
    assert activation.activated and activation.error is None
    engine = activation.engine
    assert calls == [(engine, refs, "gate", {})]
    # a candidate never shares modules with its predecessor (decision 24, the pristine-copy test of 1b-T12)
    assert engine.force_field is not failed_engine.force_field
    assert engine.force_field.source_model is not failed_engine.force_field.source_model
    assert engine.force_field.source_model is not ff.model.model
    ours = {id(module) for module in engine.force_field.source_model.modules()}
    assert not ours & {id(module) for module in failed_engine.force_field.source_model.modules()}
    after = tmp._parameters(ff.model)
    assert all(torch.equal(before[name], after[name]) for name in before)
    active = activation.plan
    assert active.revision == cand.revision and active.predecessor is plan
    assert active.rewrites == cand.rewrites
    assert active.fallbacks == (TriedCandidate(dimension="rewrites", change=cand.reason.split(": ")[1],
                                               outcome="activated", reason="ddmin", revision=cand.revision,
                                               full_gate_runs=1),)
    assert engine.plan == active and activation.gate.passed
    assert (budget.full_gate_runs, budget.reduced_gate_runs) == (1, 0)
    engine.run(2)
    engine.flush()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_failing_gate_never_activates(cpu_case, monkeypatch):
    planner, ff, atoms, plan, cand = cpu_case
    record = selfcheck.GateRecord(status="failed", check="gate", reason="forces 3x the yardstick")
    gate_calls(monkeypatch, error=GateFailed("the gate failed", gate_record=record))
    budget = Budget(limit=64)
    activation = fallback.activate(cand, planner, ff, atoms, references=references(), budget=budget, reduced=True,
                                   quantities=("forces",))
    assert not activation.activated and activation.engine is None
    assert isinstance(activation.error, GateFailed) and activation.gate is record
    assert activation.candidate is cand
    tried = activation.tried
    assert (tried.outcome, tried.stage, tried.revision) == ("failed", "gate", None)
    assert "forces 3x the yardstick" in tried.reason
    assert (tried.reduced_gate_runs, tried.full_gate_runs) == (1, 0)
    assert (budget.reduced_gate_runs, budget.full_gate_runs) == (1, 0)
    # the next candidate starts from the predecessor, with this one recorded
    base = activation.plan
    assert base.revision == plan.revision and base.rewrites == plan.rewrites and base.fallbacks == (tried,)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_reduced_gate_passes_its_quantities(cpu_case, monkeypatch):
    planner, ff, atoms, plan, cand = cpu_case
    calls = gate_calls(monkeypatch)
    activation = fallback.activate(cand, planner, ff, atoms, references=references(), reduced=True,
                                   quantities=("energy",))
    assert activation.activated and calls[0][3] == {"quantities": ("energy",)}
    assert activation.tried.reduced_gate_runs == 1 and activation.tried.full_gate_runs == 0


def test_a_failing_self_check_never_activates(cpu_case, monkeypatch):
    planner, ff, atoms, plan, cand = cpu_case
    calls = gate_calls(monkeypatch)

    def broken(self, *args, **kwargs):
        raise SelfCheckError("self-check of FOLD_RADIAL_CUTOFF failed: 3e-3 > 1e-5")

    monkeypatch.setattr(type(planner), "build", broken)
    budget = Budget(limit=64)
    activation = fallback.activate(cand, planner, ff, atoms, references=references(), budget=budget)
    assert not activation.activated and activation.engine is None and calls == []
    assert (activation.tried.stage, activation.tried.outcome) == ("self-check", "failed")
    assert "FOLD_RADIAL_CUTOFF" in activation.tried.reason
    assert (budget.reduced_gate_runs, budget.full_gate_runs) == (0, 0)
    assert activation.tried.full_gate_runs == 0


def test_other_build_errors_are_build_failures_with_the_original_error(cpu_case, monkeypatch):
    planner, ff, atoms, plan, cand = cpu_case
    gate_calls(monkeypatch)
    error = RuntimeError("capture of stage B failed")
    monkeypatch.setattr(type(planner), "build", lambda self, *a, **k: (_ for _ in ()).throw(error))
    activation = fallback.activate(cand, planner, ff, atoms, references=references())
    assert activation.error is error and activation.tried.stage == "build"
    assert "capture of stage B failed" in activation.tried.reason


@pytest.mark.parametrize("error", [ProcessConstraintError("one compile family per process"),
                                   torch.cuda.OutOfMemoryError("CUDA out of memory"),
                                   NonFiniteError("forces are not finite")])
def test_constraint_memory_and_nonfinite_errors_propagate(cpu_case, monkeypatch, error):
    planner, ff, atoms, plan, cand = cpu_case
    gate_calls(monkeypatch)
    monkeypatch.setattr(type(planner), "build", lambda self, *a, **k: (_ for _ in ()).throw(error))
    with pytest.raises(type(error)):
        fallback.activate(cand, planner, ff, atoms, references=references())


def test_activation_needs_references(cpu_case, monkeypatch):
    planner, ff, atoms, plan, cand = cpu_case
    gate_calls(monkeypatch)
    with pytest.raises(ValueError, match="references"):
        fallback.activate(cand, planner, ff, atoms, references=references("skip"))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_an_unavailable_gate_does_not_validate(cpu_case, monkeypatch):
    planner, ff, atoms, plan, cand = cpu_case
    record = selfcheck.GateRecord(status="unavailable", check="allow_unavailable", reason="no memory")
    gate_calls(monkeypatch, result=record)
    activation = fallback.activate(cand, planner, ff, atoms, references=references("allow_unavailable"))
    assert not activation.activated and activation.tried.stage == "gate" and activation.gate is record


def test_a_subprocess_candidate_is_validated_in_its_child(cpu_case, monkeypatch):
    planner, ff, atoms, plan, cand = cpu_case
    calls = gate_calls(monkeypatch)
    child = dataclasses.replace(cand, runner="subprocess")
    with pytest.raises(ProcessConstraintError, match="subprocess"):
        fallback.activate(child, planner, ff, atoms, references=references())
    assert calls == []


def test_activate_takes_only_fallback_candidates(cpu_case, monkeypatch):
    planner, ff, atoms, plan, cand = cpu_case
    gate_calls(monkeypatch)
    with pytest.raises(ValueError, match="candidate"):
        fallback.activate(plan, planner, ff, atoms, references=references())


# --------------------------------------------------------------------------- records


def test_fallbacks_round_trip_through_the_restart_record():
    _, plan = gpu_plan()
    tried = (TriedCandidate("rewrites", "rewrites fold_cutoff off", "failed", "gate failed: forces", stage="gate",
                            reduced_gate_runs=1),
             TriedCandidate("mode", "mode eager", "activated", "capture failed", revision=1, full_gate_runs=1))
    active = plan.revise("fallback: mode eager: capture failed", 0, fallbacks=tried)
    record = active.to_restart(edge_bucket=1024, neighbour_capacity=2048)
    assert PLAN_RECORD_VERSION == 2 and record["plan_record_version"] == 2
    assert json.loads(json.dumps(record)) == record
    assert record["fallbacks"][0] == {"dimension": "rewrites", "change": "rewrites fold_cutoff off",
                                      "outcome": "failed", "reason": "gate failed: forces", "stage": "gate",
                                      "revision": None, "reduced_gate_runs": 1, "full_gate_runs": 0}
    again = Plan.from_restart(record)
    assert again.fallbacks == tried
    assert again.to_restart(edge_bucket=1024, neighbour_capacity=2048) == record


def test_version_1_restart_records_still_read():
    _, plan = gpu_plan()
    record = plan.to_restart(edge_bucket=1024, neighbour_capacity=2048)
    record["plan_record_version"] = 1
    del record["fallbacks"]
    again = Plan.from_restart(record)
    assert again.fallbacks == () and again.rewrites == plan.rewrites
    record["plan_record_version"] = 3
    with pytest.raises(ValueError, match="plan record version"):
        Plan.from_restart(record)


def test_describe_lists_the_tried_candidates():
    _, plan = gpu_plan()
    tried = (TriedCandidate("rewrites", "rewrites fold_cutoff off", "failed", "gate failed: forces", stage="gate"),
             TriedCandidate("mode", "mode eager", "activated", "capture failed", revision=1, full_gate_runs=1))
    text = str(plan.revise("fallback: mode eager: capture failed", 0, fallbacks=tried))
    assert "  fallback tried: rewrites fold_cutoff off: failed at gate (gate failed: forces)" in text
    assert "  fallback tried: mode eager: activated as revision 1 (capture failed)" in text
