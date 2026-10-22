"""The minimal planner and the plan-free ``fm.ForceField`` (plan task 1b-T12; SPEC §9.1-§9.5, §7.1, §7.6, §8.7).

- For each of the 19 labels and every ``main.py`` argument set of 1a-T15, mapped onto 1b inputs (the
  rewrites and the precision on the ``ForceField``, the rest on the planner), the minimal plan's
  rewrite set and engine options equal ``FixedPlanner``'s, refusals included.
- Each hard rule of §9.3 refuses with its rule and code; several give one ``PlanRefused`` with every
  reason; rule 2 gives ``SpeciesNotInSuperset``. Owner decisions D3 (superset = structure species)
  and D6 are tested; 3-T06 plans the compiled MD tier for rule 4 and rule 9 when no mode is given.
- The backend rule (hybrid when OEq builds and MACE was not imported first) and the process family
  (§8.7) are enforced at plan time.
- ``Plan.revise`` keeps the chain; ``Plan.to_restart`` is JSON with the edge and capacity buckets.
- ``ForceField`` is plan-free: two plans give two execution copies, the pristine model is bitwise
  unchanged; ``evaluate`` plans a one-off compiled evaluation; a superset species without an
  embedding raises ``ModelElementError``. On the CPU (tiny models, plain cuEq, eager stages) the
  minimal plan's engine equals ``FixedPlanner``'s.
"""

import dataclasses
import json

import numpy as np
import pytest
import test_planner as tp
import torch
from ase.constraints import FixAtoms

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace.config import DEFAULT_SKIN_A, EngineConfig, IntegratorConfig, PrecisionConfig, RewriteConfig
from fastmace.engine import constraints
from fastmace.errors import ModelElementError, PlanRefused, ProcessConstraintError, Reason, SpeciesNotInSuperset
from fastmace.planner import fixed
from fastmace.planner.minimal import MinimalPlanner, SoftwareTraits
from fastmace.planner.plan import Plan
from fastmace.planner.workload import Workload

GPU, CPU = tp.GPU, tp.CPU
NA_CL = (11, 17)
WORKLOAD = Workload(max_temperature_K=300.0)


def software(**changes):
    """Software traits with everything available (the GPU machines of 1a), changed by ``changes``."""
    base = dict(oeq=(True, "openequivariance is installed"), triton=(True, "triton is importable"),
                nvrtc=(True, "NVRTC loads"), fused_cueq=(True, "cuequivariance_ops_torch imports"),
                mace_imported_first=())
    base.update(changes)
    return SoftwareTraits(**base)


def minimal(process=None, soft=None, **configs):
    configs.setdefault("environ", {})
    configs.setdefault("torch_version", tp.TORCH_2_14)
    return MinimalPlanner(software=soft or software(), process=process or constraints.ProcessRecord(), **configs)


def plan_label(label, atoms=None, integrator=None, workload=WORKLOAD, *, precision="float32", rewrites="auto",
               species=None, planner=None, **flags):
    planner = planner or minimal()
    return planner.plan_traits(tp.traits(label), atoms if atoms is not None else nacl(), integrator or
                               IntegratorConfig(), workload, GPU, precision=precision, rewrites=rewrites,
                               species=species, **flags)


def codes(error):
    return [reason.code for reason in error.reasons]


# --------------------------------------------------------------------------- equal to FixedPlanner


@pytest.mark.parametrize("label", tp.LABELS)
def test_minimal_plans_equal_fixed_plans(label):
    """19 labels x 1a-T15's 31 main.py argument sets: same rewrites, engine, neighbours, polar, integrator."""
    record = tp.traits(label)
    compared = 0
    for argv in tp.ARGV_CASES:
        configs = tp.configs_from_argv(argv)
        integrator = configs.pop("integrator")
        try:
            want = tp.planner(**configs).plan(record, nacl(), integrator, GPU)
        except PlanRefused:
            want = None
        rewrites = configs.pop("rewrites")
        precision = configs.pop("precision")
        dtype = precision.dtype if precision.dtype != "auto" else (want.precision.dtype if want else "float32")
        try:
            got = minimal(**configs).plan_traits(record, nacl(), integrator, WORKLOAD, GPU, precision=dtype,
                                                 rewrites=rewrites)
        except PlanRefused as error:
            assert want is None, f"{label} {argv}: minimal refused ({error}), fixed planned"
            assert all(isinstance(reason, Reason) for reason in error.reasons)
            continue
        assert want is not None, f"{label} {argv}: fixed refused, minimal planned"
        for part in ("rewrites", "engine", "neighbours", "polar", "integrator", "precision"):
            assert getattr(got, part) == getattr(want, part), (label, argv, part)
        assert got.auto_options_off == want.auto_options_off, (label, argv)
        assert got.tier == "graph"
        compared += 1
    assert compared >= len(tp.ARGV_CASES) - 8


def test_float64_energy_passes_through():
    plan = plan_label("mh1-r2scan", planner=minimal(precision=PrecisionConfig(float64_energy=False)))
    assert plan.precision.float64_energy is False
    assert plan_label("mh1-r2scan").precision.float64_energy is True


def test_precision_is_owned_by_the_force_field():
    with pytest.raises(ValueError, match="ForceField"):
        minimal(precision=PrecisionConfig(dtype="float64"))
    assert plan_label("mh1-r2scan", precision="float64").precision.dtype == "float64"


# --------------------------------------------------------------------------- hard rules


def _with_constraint(atoms):
    atoms.set_constraint(FixAtoms(indices=[0]))
    return atoms


def _slab():
    atoms = nacl()
    atoms.pbc = (True, True, False)
    return atoms


@pytest.mark.parametrize(("rule", "code", "kwargs"), [
    (1, "non-3d-pbc", dict(atoms=_slab())),
    (1, "constraints", dict(atoms=_with_constraint(nacl()))),
    (1, "variable-cell-without-barostat", dict(variable_cell=True)),
    (2, "species-not-in-superset", dict(species=["Na"])),
    (5, "reproducible-family-not-qualified", dict(label="mp0-medium",
                                                  workload=Workload(max_temperature_K=300.0, reproducible=True),
                                                  planner=minimal(versions={"cuequivariance_ops": "0.12.0"}))),
    (7, "atom-count-changes", dict(workload=Workload(max_temperature_K=300.0, atom_count_changes=True))),
])
def test_each_hard_rule_refuses_with_its_reason(rule, code, kwargs):
    with pytest.raises(PlanRefused) as caught:
        plan_label(kwargs.pop("label", "mh1-r2scan"), **kwargs)
    assert codes(caught.value) == [code]
    assert caught.value.reasons[0].rule == rule
    assert isinstance(caught.value, SpeciesNotInSuperset) == (rule == 2)


def test_rule_3_polar_on_plain_compiled_cueq():
    with pytest.raises(PlanRefused) as caught:
        plan_label("polar1-s", planner=minimal(engine=EngineConfig(backend="cueq")))
    assert codes(caught.value) == ["polar-plain-compiled-cueq"] and caught.value.reasons[0].rule == 3
    plan = plan_label("polar1-s", planner=minimal(engine=EngineConfig(backend="cueq", compile_mode="none")))
    assert plan.engine.backend == "cueq"


def test_rule_4_inexact_padding_refuses_the_graph_tier():
    """3-T06: without a mode, inexact padding plans the compiled tier (exact edge counts); an explicit
    ``mode="graph"`` is refused, naming ``mode="compiled"`` (rule 4 is hard)."""
    record = dataclasses.replace(tp.traits("mh1-r2scan"), padding_is_exact=False)
    with pytest.raises(PlanRefused) as caught:
        minimal().plan_traits(record, nacl(), IntegratorConfig(), WORKLOAD, GPU, precision="float32", mode="graph")
    assert codes(caught.value) == ["inexact-padding"] and caught.value.reasons[0].rule == 4
    assert 'mode="compiled"' in caught.value.reasons[0] and "phase 3" not in caught.value.reasons[0]
    plan = minimal().plan_traits(record, nacl(), IntegratorConfig(), WORKLOAD, GPU, precision="float32")
    assert plan.tier == "compiled"
    choice, reason = plan.decision("mode")[1:]
    assert choice == "compiled" and reason.code == "mode-compiled-rule-4" and reason.rule == 4
    # The one-off evaluation uses exact edge counts: allowed.
    plan = minimal().plan_evaluation_traits(record, nacl(), GPU, precision="float32")
    assert plan.tier == "compiled"


def test_rule_6_naive_cueq_refused_unless_allowed():
    planner = minimal(soft=software(fused_cueq=(False, "cuequivariance_ops_torch cannot be imported")),
                      engine=EngineConfig(backend="cueq"))
    with pytest.raises(PlanRefused) as caught:
        plan_label("mh1-r2scan", planner=planner)
    assert codes(caught.value) == ["naive-cueq"] and caught.value.reasons[0].rule == 6
    plan = plan_label("mh1-r2scan", planner=planner, allow_naive_cueq=True)
    assert "allow_naive_cueq" in plan.allowances
    assert plan.decision("allow_naive_cueq")[1] == "on"
    # The hybrid uses no cuEq convolution, but its symmetric contraction is cuEq: refused too.
    with pytest.raises(PlanRefused, match="naive"):
        plan_label("mh1-r2scan", planner=minimal(soft=software(fused_cueq=(False, "no ops"))))


def test_three_violated_rules_give_one_refusal_with_three_reasons():
    with pytest.raises(PlanRefused) as caught:
        plan_label("mh1-r2scan", atoms=_with_constraint(_slab()),
                   workload=Workload(max_temperature_K=300.0, reproducible=True),
                   planner=minimal(engine=EngineConfig(backend="hybrid")))
    assert sorted(codes(caught.value)) == ["constraints", "non-3d-pbc", "reproducible-backend"]
    assert not isinstance(caught.value, SpeciesNotInSuperset)


def test_superset_violation_is_species_not_in_superset_with_every_reason():
    with pytest.raises(SpeciesNotInSuperset) as caught:
        plan_label("mh1-r2scan", atoms=_slab(), species=["Cl"])
    assert sorted(codes(caught.value)) == ["non-3d-pbc", "species-not-in-superset"]
    assert "Na" in str(caught.value)


def test_d3_superset_beyond_the_structure_is_phase_3():
    with pytest.raises(PlanRefused) as caught:
        plan_label("mh1-r2scan", species=["Na", "Cl", "O"])
    assert codes(caught.value) == ["superset-beyond-structure"]
    assert "phase 3 (§8.5)" in caught.value.reasons[0]
    # Sets, not sequences.
    assert plan_label("mh1-r2scan", species=["Cl", "Na", "Na"]).superset == NA_CL
    assert plan_label("mh1-r2scan", species=[17, 11]).superset == NA_CL


def test_mode_compiled_plans_the_compiled_tier():  # 3-T06 lifted the phase 3 refusal
    plan = plan_label("mh1-r2scan", mode="compiled")
    assert plan.tier == "compiled" and plan.decision("mode")[2].code == "mode-compiled-tier"


@pytest.mark.parametrize("label", tp.LABELS)
def test_mode_reference_plans_the_reference_tier_for_every_label(label):
    """2-T03: no rewrites, the stock uncompiled cuEq backend, no capture; padding and polar refusals lifted."""
    plan = plan_label(label, mode="reference")
    assert plan.tier == "reference"
    assert (plan.engine.backend, plan.engine.compile_mode, plan.engine.pipelined) == ("cueq", "none", "off")
    assert not [name for name in fixed.REWRITE_FIELDS if getattr(plan.rewrites, name) == "on"]
    assert plan.decision("capture")[1] == "off" and plan.decision("tier")[1] == "reference"
    assert {choice for key, choice, _ in plan.decisions if key.startswith("RewriteConfig.")} <= {"off"}


def test_unknown_mode_is_a_value_error():
    with pytest.raises(ValueError, match="mode"):
        plan_label("mh1-r2scan", mode="fast")


# --------------------------------------------------------------------------- preferences in 1b


def test_rule_9_frequent_host_callbacks_plan_the_compiled_tier():
    """3-T06: below ``RULE_9_HOST_CALLBACKS`` steps between callbacks, a plan without a mode is compiled;
    an explicit ``mode="graph"`` overrides the preference and keeps the graph tier with a host sync."""
    from fastmace.planner import minimal as minimal_module

    assert minimal_module.RULE_9_HOST_CALLBACKS == 10
    frequent = Workload(max_temperature_K=300.0, host_callbacks_every=2)
    plan = plan_label("mh1-r2scan", workload=frequent)
    assert plan.tier == "compiled"
    mode = plan.decision("mode")
    assert mode[2].code == "mode-compiled-rule-9" and mode[2].rule == 9 and "10" in mode[2]
    assert plan.decision("workload.host_callbacks_every")[2].code == "host-callbacks-compiled"
    graph = plan_label("mh1-r2scan", workload=frequent, mode="graph")
    assert graph.tier == "graph"
    key, choice, reason = graph.decision("workload.host_callbacks_every")
    assert choice == "graph, host sync every 2 steps" and reason.code == "host-callbacks-graph-sync"
    for every in (10, 1000):
        rare = plan_label("mh1-r2scan", workload=Workload(max_temperature_K=300.0, host_callbacks_every=every))
        assert rare.tier == "graph" and rare.decision("mode")[2].code == "mode-graph"


def test_rule_10_stress_and_headroom_follow_the_integrator():
    nvt = plan_label("polar1-s")
    npt = plan_label("polar1-s", integrator=IntegratorConfig(ensemble="npt"))
    assert nvt.decision("compute_stress")[1] == "off" and npt.decision("compute_stress")[1] == "on"
    assert (nvt.polar.kspace_headroom, npt.polar.kspace_headroom) == (0.0, 0.1)


def test_rule_11_records_max_temperature_and_keeps_the_1a_skin():
    plan = plan_label("mh1-r2scan", workload=Workload(max_temperature_K=2500.0))
    assert plan.workload.max_temperature_K == 2500.0
    assert plan.neighbours.skin == DEFAULT_SKIN_A  # NeighbourConfig().skin is "auto" since 3-T01
    assert "phase 3" in plan.decision("workload.max_temperature_K")[2]


# --------------------------------------------------------------------------- backend and process family


def test_backend_is_hybrid_when_oeq_builds():
    plan = plan_label("mh1-r2scan")
    assert plan.engine.backend == "hybrid"
    assert plan.decision("EngineConfig.backend")[2].code == "backend-hybrid"


@pytest.mark.parametrize(("soft", "code"), [
    (dict(oeq=(False, "openequivariance is not installed")), "backend-cueq-oeq-unavailable"),
    (dict(mace_imported_first=("mace",)), "backend-cueq-mace-imported-first"),
])
def test_backend_falls_back_to_plain_cueq_with_the_reason(soft, code):
    plan = plan_label("mh1-r2scan", planner=minimal(soft=software(**soft)), precision="float64")
    assert plan.engine.backend == "cueq"
    key, choice, reason = plan.decision("EngineConfig.backend")
    assert choice == "cueq" and reason.code == code
    # The rewrite set is plain cuEq's, as FixedPlanner's with --backend cueq.
    want = tp.planner(engine=EngineConfig(backend="cueq")).plan(tp.traits("mh1-r2scan"), nacl(), IntegratorConfig(),
                                                                 GPU)
    assert plan.rewrites == want.rewrites and plan.neighbours == want.neighbours


def test_explicit_hybrid_without_oeq_is_refused():
    planner = minimal(soft=software(oeq=(False, "openequivariance is not installed")),
                      engine=EngineConfig(backend="hybrid"))
    with pytest.raises(PlanRefused) as caught:
        plan_label("mh1-r2scan", planner=planner)
    assert codes(caught.value) == ["hybrid-unavailable"]


def test_cpu_device_plans_plain_cueq():
    planner = minimal(engine=EngineConfig(device="cpu", compile_mode="none"))
    plan = planner.plan_traits(tp.traits("mh1-r2scan"), nacl(), IntegratorConfig(), WORKLOAD, CPU, precision="float64")
    assert plan.engine.backend == "cueq" and plan.decision("EngineConfig.backend")[2].code == "backend-cueq-cpu"


def test_the_process_family_is_enforced_at_plan_time():
    record = constraints.ProcessRecord()
    hybrid = plan_label("mh1-r2scan", planner=minimal(process=record))
    assert hybrid.engine.backend == "hybrid" and record.family == constraints.OEQ_FAMILY
    # An MD plan of the other family runs in a child through the subprocess runner (2-T07); the
    # process keeps its family.
    plan = plan_label("mh1-r2scan", planner=minimal(process=record, engine=EngineConfig(backend="cueq")))
    assert plan.runner == "subprocess" and plan.engine.backend == "cueq"
    assert plan.decision("runner")[1] == "subprocess"
    assert plan.decision("runner")[2].code == "runner-subprocess-family"
    assert record.family == constraints.OEQ_FAMILY
    # A one-off evaluation has no runner: refused, with the way out named.
    with pytest.raises(ProcessConstraintError) as caught:
        minimal(process=record, engine=EngineConfig(backend="cueq")).plan_evaluation_traits(
            tp.traits("mh1-r2scan"), nacl(), GPU, precision="float32")
    assert "subprocess runner" in str(caught.value) and "phase 2" not in str(caught.value)
    # Uncompiled plain cuEq compiles nothing: allowed. Auto respects the family.
    plan_label("mh1-r2scan", planner=minimal(process=record, engine=EngineConfig(backend="cueq",
                                                                                  compile_mode="none")))
    other = constraints.ProcessRecord(family=constraints.AUTOGRAD_FAMILY, family_owner="an earlier plan")
    plan = plan_label("mh1-r2scan", planner=minimal(process=other), precision="float64")
    assert plan.engine.backend == "cueq"
    assert plan.decision("EngineConfig.backend")[2].code == "backend-cueq-process-family"


def test_reproducible_after_cuda_was_initialised_offers_the_runner(monkeypatch):
    """§8.7: the deterministic settings must precede CUDA's initialisation, so a ``reproducible=True``
    plan in a process that already initialised CUDA runs in a child (2-T07). Rule 5 still refuses
    every reproducible plan until 2-T16, so the trigger is tested on ``_claim_family`` itself."""
    from fastmace.planner import minimal as minimal_module

    planner = minimal()
    plan = plan_label("mh1-r2scan", planner=planner)
    reproducible = dataclasses.replace(plan, workload=dataclasses.replace(plan.workload, reproducible=True))
    monkeypatch.setattr(minimal_module, "_cuda_initialised", lambda: True)
    offered = planner._claim_family(reproducible, "a test plan", offer_runner=True)
    assert offered.runner == "subprocess"
    assert offered.decision("runner")[2].code == "runner-subprocess-reproducible"
    assert planner._claim_family(plan, "a test plan", offer_runner=True).runner == "in-process"
    monkeypatch.setattr(minimal_module, "_cuda_initialised", lambda: False)
    assert planner._claim_family(reproducible, "a test plan", offer_runner=True).runner == "in-process"


def test_default_plans_have_no_runner_decision():
    plan = plan_label("mh1-r2scan")
    assert plan.runner == "in-process"
    with pytest.raises(KeyError):
        plan.decision("runner")


# --------------------------------------------------------------------------- the Plan record


def test_decision_keys_are_unique_and_reasons_have_codes():
    for label in ("mh1-r2scan", "mp0-medium", "polar1-s"):
        for integrator in (IntegratorConfig(), IntegratorConfig(ensemble="npt")):
            plan = plan_label(label, integrator=integrator)
            keys = [key for key, _, _ in plan.decisions]
            assert len(keys) == len(set(keys)), label
            for key in ("mode", "tier", "EngineConfig.backend", "workload.max_temperature_K", "superset",
                        "software.oeq", "software.triton", "software.nvrtc", "software.fused_cueq",
                        "software.mace_imported_first", "compute_stress", "integrator"):
                assert key in keys, (label, key)
            assert plan.decision("superset")[1] == "Na Cl"


def test_software_traits_are_recorded():
    plan = plan_label("mh1-r2scan", planner=minimal(soft=software(triton=(False, "triton is not importable"))))
    assert plan.software.triton == (False, "triton is not importable")
    assert plan.decision("software.triton")[1] == "off"


def test_revise_keeps_the_chain_and_the_original():
    plan = plan_label("mh1-r2scan")
    before = plan.to_restart(edge_bucket=1024, neighbour_capacity=2048)
    engine = dataclasses.replace(plan.engine, pipelined="off")
    first = plan.revise("pipelining off after a replay check", 500, engine=engine)
    second = first.revise("restart with a new plan", 1000)
    assert (first.revision, first.predecessor, first.step) == (1, plan, 500)
    assert (second.revision, second.predecessor.predecessor) == (2, plan)
    assert first.engine.pipelined == "off" and plan.engine.pipelined == "on"
    assert plan.revision == 0 and plan.predecessor is None
    assert plan.to_restart(edge_bucket=1024, neighbour_capacity=2048) == before
    assert "predecessor: revision 1" in str(second)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.revision = 3
    with pytest.raises(ValueError, match="revision"):
        plan.revise("x", 1, revision=5)


def test_to_restart_is_json_with_the_buckets_and_round_trips():
    from fastmace.io import restart

    plan = plan_label("mh1-r2scan", integrator=IntegratorConfig(ensemble="npt", external_stress_GPa=(1, 2, 3, 0, 0, 0)))
    plan = plan.revise("a revision", 10)
    record = plan.to_restart(edge_bucket=4096, neighbour_capacity=8192)
    assert json.loads(json.dumps(record)) == record
    assert {key: record[key] for key in restart.PLAN_BUCKET_KEYS} == {"edge_bucket": 4096, "neighbour_capacity": 8192}
    assert record["revision"] == 1
    assert record["chain"] == [{"revision": 0, "step": None, "reason": plan.predecessor.reason}]
    again = Plan.from_restart(record)
    assert again.to_restart(edge_bucket=4096, neighbour_capacity=8192) == record
    assert again.rewrites == plan.rewrites and again.integrator == plan.integrator and again.workload == plan.workload
    assert again.decision("EngineConfig.backend")[2].code == "backend-hybrid"


def test_evaluation_plan_is_compiled_with_stress_and_no_workload():
    plan = minimal().plan_evaluation_traits(tp.traits("mh1-r2scan"), nacl(), GPU, precision="float32")
    assert plan.tier == "compiled" and plan.workload is None
    assert plan.decision("compute_stress")[1] == "on" and plan.decision("capture")[1] == "off"
    assert plan.engine.pipelined == "off" and plan.engine.async_output == "off"
    graph = plan_label("mh1-r2scan")
    assert plan.rewrites == graph.rewrites and plan.neighbours == graph.neighbours
    with pytest.raises(PlanRefused) as caught:
        minimal().plan_evaluation_traits(tp.traits("mh1-r2scan"), _slab(), GPU, precision="float32")
    assert codes(caught.value) == ["non-3d-pbc"]


def test_fixed_planner_refusals_carry_codes():
    with pytest.raises(PlanRefused) as caught:
        tp.plan_for("mh1-r2scan", atoms=_with_constraint(_slab()), rewrites=RewriteConfig(inplace_block_linear=True))
    assert sorted(codes(caught.value)) == ["constraints", "inplace-block-linear-not-carried-over", "non-3d-pbc"]
    assert fixed.PlanRefused is PlanRefused


def test_integrator_keywords_read_the_new_fields():
    from ase import units

    for pressure in (0.0, 1.5, -2.0):
        config = IntegratorConfig(ensemble="npt", pressure_GPa=pressure)
        keywords = fixed.integrator_keywords(config)
        assert keywords["externalstress"] == -pressure * units.GPa  # bitwise as 1a (baoab_settings)
        assert keywords["disable_cell_langevin"] is False
    config = IntegratorConfig(ensemble="npt", external_stress_GPa=(1.0, 2.0, 3.0, 0.5, 0.25, 0.125),
                              disable_cell_langevin=True)
    keywords = fixed.integrator_keywords(config)
    np.testing.assert_array_equal(keywords["externalstress"], np.array([1.0, 2.0, 3.0, 0.5, 0.25, 0.125]) * units.GPa)
    assert keywords["disable_cell_langevin"] is True
    assert fixed.integrator_keywords(IntegratorConfig()) == {}


# --------------------------------------------------------------------------- ForceField (CPU, tiny models)


@pytest.fixture
def pristine(tmp_path, quiet_licences):
    from fastmace.model import loading

    def make(family="mh1", z=NA_CL):
        model = tiny_mace_model(list(z), **FAMILIES[family])
        path = tmp_path / f"tiny-{family}-{'-'.join(map(str, z))}.model"
        torch.save(model, path)
        return loading.load_pristine(str(path), trusted=True)

    return make


@pytest.fixture
def quiet_licences(monkeypatch):
    from fastmace.model import loading

    monkeypatch.setattr(loading, "_LICENCES_SHOWN", set())


def cpu_minimal(process=None):
    return MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ={},
                          process=process or constraints.ProcessRecord())


def _parameters(model):
    return {name: tensor.detach().clone() for name, tensor in model.model.state_dict().items()}


def test_force_field_needs_a_loaded_model_or_a_label():
    from fastmace.model.forcefield import ForceField

    with pytest.raises(TypeError, match="fm.load"):
        ForceField(object())


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_force_field_superset_and_identity(pristine):
    from fastmace.model.forcefield import ForceField
    from fastmace.state import ForceFieldIdentity

    loaded = pristine()
    ff = ForceField(loaded)
    assert ff.species is None and ff.identity is None  # fixed at first use
    assert ff.superset_for(nacl()) == NA_CL and ff.species == NA_CL
    identity = ff.identity
    assert isinstance(identity, ForceFieldIdentity)
    assert (identity.checkpoint_sha256, identity.species, identity.force_precision) == (loaded.sha256, NA_CL,
                                                                                        "float32")
    assert ForceField(loaded, nacl(), precision="float64").identity.digest != identity.digest
    assert ForceField(loaded, species=["Cl", "Na"]).species == NA_CL
    with pytest.raises(ValueError, match="precision"):
        ForceField(loaded, precision="float16")
    with pytest.raises(ModelElementError, match="O"):
        ForceField(loaded, species=["Na", "Cl", "O"])
    with pytest.raises(ModelElementError):
        ForceField(loaded).superset_for(nacl().__class__("NaClO", positions=[[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                                                         cell=[5, 5, 5], pbc=True))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_force_field_is_plan_free(pristine):
    pytest.importorskip("cuequivariance_torch")
    from fastmace.model.forcefield import ForceField

    loaded = pristine()
    before = _parameters(loaded)
    ff = ForceField(loaded, nacl(), precision="float64")
    planner = cpu_minimal()
    atoms = nacl(rattle=0.05, seed=1)
    engines = []
    for integrator in (IntegratorConfig(temperature_K=300.0), IntegratorConfig(ensemble="npt", temperature_K=300.0)):
        plan = planner.plan(ff, atoms, integrator, WORKLOAD, CPU)
        engines.append(planner.build(plan, ff, atoms))
    first, second = (engine.force_field for engine in engines)
    assert first is not second and first.source_model is not second.source_model
    assert first.source_model is not loaded.model
    for engine in engines:
        engine.run(3)
        engine.flush()
    after = _parameters(loaded)
    assert before.keys() == after.keys()
    for name in before:
        assert torch.equal(before[name], after[name]), name
    assert loaded.backend == "e3nn" and loaded.dtype == "float64"


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(("family", "npt"), [("mh1", False), ("density", False), ("mh1", True)])
def test_minimal_engine_equals_the_fixed_engine_on_the_cpu(pristine, tmp_path, family, npt):
    pytest.importorskip("cuequivariance_torch")
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField

    loaded = pristine(family)
    atoms = nacl(rattle=0.05, seed=1)
    integrator = IntegratorConfig(ensemble="npt", pressure_GPa=0.5) if npt else IntegratorConfig()
    integrator = dataclasses.replace(integrator, temperature_K=300.0)
    planner = cpu_minimal()
    plan = planner.plan(ForceField(loaded, atoms, precision="float64"), atoms, integrator, WORKLOAD, CPU)
    engine = planner.build(plan, ForceField(loaded, atoms, precision="float64"), atoms)

    reference = fixed.FixedPlanner(engine=EngineConfig(device="cpu", backend="cueq", compile_mode="none"), environ={})
    converted = loading.load(str(loaded.path), backend="cueq", device="cpu", trusted=True)
    want = reference.plan(converted, atoms, integrator, CPU)
    for part in ("rewrites", "engine", "neighbours", "precision", "integrator"):
        assert getattr(plan, part) == getattr(want, part), part
    manual = reference.build(want, converted, atoms, temperature_K=300.0, constraints=constraints.ProcessRecord())
    for target in (engine, manual):
        target.run(6)
        target.flush()
    np.testing.assert_array_equal(engine.state.positions.numpy(), manual.state.positions.numpy())
    assert engine.potential_energy() == manual.potential_energy()
    if npt:
        np.testing.assert_array_equal(engine.state.cell.numpy(), manual.state.cell.numpy())


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_force_field_evaluate_plans_a_one_off_evaluation(pristine, monkeypatch):
    pytest.importorskip("cuequivariance_torch")
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal as minimal_module

    loaded = pristine()
    record = constraints.ProcessRecord()
    monkeypatch.setattr(minimal_module, "default_planner",
                        lambda force_field: cpu_minimal(record))
    ff = ForceField(loaded, precision="float64")
    atoms = nacl(rattle=0.05, seed=2)
    energy, forces, stress = ff.evaluate(atoms, stress=True)
    assert np.isfinite(energy) and forces.shape == (len(atoms), 3) and stress.shape == (3, 3)
    assert ff.evaluate(atoms)[2] is None
    assert ff.evaluation_plan(atoms).tier == "compiled"
    # MACE's own calculator on the pristine e3nn model, float64.
    probe = atoms.copy()
    probe.calc = loaded.calculator
    np.testing.assert_allclose(forces, probe.get_forces(), rtol=0, atol=1e-10)
    np.testing.assert_allclose(energy, probe.get_potential_energy(), rtol=1e-12, atol=1e-10)
    with pytest.raises(PlanRefused):
        ff.evaluate(_slab())
