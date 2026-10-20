"""Unknown MACE versions, the always-on treatments and the self-check reaction (plan task 2-T04).

SPEC §6.3 (versioned registry), §9.3 rule 13, §12.1 and §18 decision 2: 0.3.16 is the only
qualified MACE. On another version every record or patch whose pinned source does not hash is
off with its reason, the always-on treatments follow the rule of 2-T04, and nothing is patched
that was not hashed. Fake versions and fake hashes stand in for another MACE here; the real one
is ``test_unknown_mace.py`` in ``~/environments/fastmace_mace_other``.
"""

from __future__ import annotations

import dataclasses
import importlib
import types

import pytest

from fastmace import _patches
from fastmace.config import RewriteConfig
from fastmace.errors import Reason, SelfCheckError
from fastmace.model import energy
from fastmace.model.capabilities import RewriteSupport
from fastmace.planner import reactions
from fastmace.planner.plan import Plan
from fastmace.rewrites import engine, versions
from fastmace.rewrites import registry as reg
from fastmace.rewrites.registry import CheckResult, Registry, Rewrite

pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    # Fake hashes against the qualified sources; the real other MACE is test_unknown_mace.py.
    pytest.mark.skipif(reg.installed_mace_version() not in versions.QUALIFIED_MACE_VERSIONS,
                       reason="needs the qualified MACE 0.3.16"),
]

PATCHES = ("capture_safe_mask_head", "first_derivative_only", "single_graph_strain", "unfuse_full_addmm")
SETTERS = {"capture_safe_mask_head": _patches.set_capture_safe_mask_head,
           "first_derivative_only": _patches.set_first_derivative_only,
           "single_graph_strain": _patches.set_single_graph_strain,
           "unfuse_full_addmm": _patches.set_unfuse_full_addmm,
           "cueq_conv_fusion": _patches.set_cueq_conv_fusion}
WRONG = "0" * 64
# The 0.3.16 reference record ids of 1b (main at b0f24d5).
MH1_ID = "fastmace.reference/0.3.16/mh1/763155208cd1ef8c"
POLAR_ID = "fastmace.reference/0.3.16/polar/8e216bde0c3515e9"


@pytest.fixture(autouse=True)
def _undo_patches():
    yield
    for name in PATCHES:
        _patches.restore(name)
    _patches.set_cueq_conv_fusion(None)


def _unpin(monkeypatch, name):
    """Pretend ``name``'s sources are another version's: every pin is wrong."""
    monkeypatch.setitem(_patches.SOURCE_PINS, name, {target: WRONG for target in _patches.SOURCE_PINS[name]})


def _current(target):
    module, attribute = target.split(":", 1)
    return getattr(importlib.import_module(module), attribute, None)


# --------------------------------------------------------------------------- hash gates on every patch


def _applies(name):
    """Whether patch ``name`` has anything to do on this torch (unfuse_full_addmm: not on torch 2.13)."""
    return _patches.patch_applicable(name)[0]


def test_every_patch_is_pinned_and_qualified_here():
    assert set(_patches.SOURCE_PINS) == set(SETTERS)
    for name in SETTERS:
        if _applies(name):
            assert _patches.patch_problems(name) == [], name


def test_only_the_addmm_patch_can_be_inapplicable():
    for name in SETTERS:
        if name != "unfuse_full_addmm":
            assert _applies(name), name
    assert _applies("unfuse_full_addmm") == (_patches._torch_release() >= _patches.ADDMM_TEST_SINCE)


def test_a_missing_addmm_test_on_torch_2_14_is_a_problem_not_not_needed(monkeypatch):
    """On torch >= 2.14 a moved or renamed bias-like test is an unqualified source, never "not needed"."""
    from torch._inductor.fx_passes import post_grad

    monkeypatch.setattr(_patches, "_torch_release", lambda: (2, 15))
    monkeypatch.delattr(post_grad, "_is_bias_like_addmm_input", raising=False)
    assert _applies("unfuse_full_addmm")
    problems = _patches.patch_problems("unfuse_full_addmm")
    assert any("_is_bias_like_addmm_input" in problem and "AttributeError" in problem for problem in problems)
    with pytest.raises(_patches.UnqualifiedPatchError):
        _patches.set_unfuse_full_addmm(True)
    assert _patches.record_of("unfuse_full_addmm") is None
    assert _patches.install_if_qualified("unfuse_full_addmm") is None  # skipped with its reason logged
    treatment = versions.treatment("unfuse_full_addmm")
    assert not treatment.qualified and treatment.decision == "not installed"


def test_an_older_torch_does_not_need_the_addmm_patch(monkeypatch):
    monkeypatch.setattr(_patches, "_torch_release", lambda: (2, 13))
    applicable, reason = _patches.patch_applicable("unfuse_full_addmm")
    assert not applicable and "before 2.14" in reason
    assert _patches.set_unfuse_full_addmm(True) is None and _patches.record_of("unfuse_full_addmm") is None
    assert versions.treatment("unfuse_full_addmm").decision == "not needed"


def test_without_inductor_the_addmm_patch_is_not_needed(monkeypatch):
    import sys

    from torch._inductor import fx_passes

    monkeypatch.delattr(fx_passes, "post_grad", raising=False)
    monkeypatch.setitem(sys.modules, "torch._inductor.fx_passes.post_grad", None)  # the import raises ImportError
    applicable, reason = _patches.patch_applicable("unfuse_full_addmm")
    assert not applicable and "Inductor is not importable" in reason


@pytest.mark.parametrize("name", sorted(SETTERS))
def test_an_unhashed_source_is_never_patched(monkeypatch, name):
    _unpin(monkeypatch, name)
    before = {target: _current(target) for target in _patches.SOURCE_PINS[name]}
    if not _applies(name):  # torch 2.13: the setter has nothing to do and patches nothing
        assert SETTERS[name](True) is None and _patches.install_if_qualified(name) is None
        assert {target: _current(target) for target in before} == before
        assert _patches.record_of(name) is None
        return
    with pytest.raises(_patches.UnqualifiedPatchError) as caught:
        SETTERS[name](True)
    assert caught.value.patch == name and caught.value.problems
    assert "unqualified source" in str(caught.value) and "Why:" in str(caught.value)
    assert {target: _current(target) for target in before} == before
    assert _patches.record_of(name) is None


def test_a_missing_source_is_a_problem(monkeypatch):
    monkeypatch.setitem(_patches.SOURCE_PINS, "first_derivative_only",
                        {"mace.modules.utils:no_such_function": WRONG})
    assert "AttributeError" in _patches.patch_problems("first_derivative_only")[0]


def test_engine_install_skips_unqualified_patches_with_their_reason(monkeypatch, caplog):
    from fastmace.config import EngineConfig

    _unpin(monkeypatch, "first_derivative_only")
    _unpin(monkeypatch, "single_graph_strain")
    records = engine.install(EngineConfig(), RewriteConfig())
    names = {record.name for record in records}
    assert "capture_safe_mask_head" in names
    assert not names & {"first_derivative_only", "single_graph_strain"}
    assert _patches.record_of("first_derivative_only") is None
    assert "first_derivative_only not installed" in caplog.text


# --------------------------------------------------------------------------- the always-on treatments


def test_treatments_on_the_qualified_version():
    found = {t.name: t for t in versions.treatments()}
    assert set(found) == set(versions.TREATMENTS)
    assert all(t.qualified for t in found.values())
    assert found["float64_energy"].decision == "patched"
    assert found["capture_safe_mask_head"].decision == "installed"
    assert found["unfuse_full_addmm"].decision == ("installed" if _applies("unfuse_full_addmm") else "not needed")


def test_treatments_on_unhashed_sources(monkeypatch):
    for name in _patches.SOURCE_PINS:
        _unpin(monkeypatch, name)
    monkeypatch.setattr(energy, "SOURCE_HASHES", {key: WRONG for key in energy.SOURCE_HASHES})
    from fastmace.longrange import polar

    monkeypatch.setattr(polar, "SOURCE_HASHES", {key: WRONG for key in polar.SOURCE_HASHES})
    found = {t.name: t for t in versions.treatments()}
    # torch 2.13: the addmm patch has nothing to do, so it is "not needed" whatever its pins say.
    inapplicable = {name for name in _patches.SOURCE_PINS if not _applies(name)}
    for name in inapplicable:
        assert found.pop(name).decision == "not needed"
    assert not any(t.qualified for t in found.values())
    assert found["float64_energy"].decision == "native float64"
    assert found["polar_forward"].decision == "refuse PolarMACE"
    assert found["capture_safe_mask_head"].decision == "no capture for multi-head models"
    for name in {"first_derivative_only", "single_graph_strain", "unfuse_full_addmm"} - inapplicable:
        assert found[name].decision == "not installed"
    assert found["cueq_conv_fusion"].decision == "MACE's own rule"
    assert all(t.reason and "sha256" in t.reason for t in found.values())


@pytest.mark.parametrize(("hashed", "requested", "expected"), [
    (True, "float32", ("float32", None)),
    (True, "float64", ("float64", None)),
    (False, "float64", ("float64", None)),
    (False, "float32", ("float64", "precision-promoted-unqualified-energy")),
])
def test_precision_promotion(monkeypatch, hashed, requested, expected):
    if not hashed:
        monkeypatch.setattr(energy, "SOURCE_HASHES", {key: WRONG for key in energy.SOURCE_HASHES})
    precision, reason = versions.effective_precision(requested, polar_model=False)
    assert (precision, None if reason is None else reason.code) == expected
    if reason is not None:
        assert "float64_energy" in reason and "float32" in reason


def test_no_promotion_without_float64_energy_or_for_polar(monkeypatch):
    monkeypatch.setattr(energy, "SOURCE_HASHES", {key: WRONG for key in energy.SOURCE_HASHES})
    assert versions.effective_precision("float32", polar_model=False, float64_energy=False) == ("float32", None)
    # PolarMACE has its own forward and is refused when that does not hash (polar_support).
    assert versions.effective_precision("float32", polar_model=True) == ("float32", None)


def test_polar_support(monkeypatch):
    from fastmace.longrange import polar

    assert versions.polar_support().supported
    monkeypatch.setattr(polar, "SOURCE_HASHES", {key: WRONG for key in polar.SOURCE_HASHES})
    support = versions.polar_support()
    assert not support.supported and "stress" in support.reason and "sha256" in support.reason


def test_capture_support(monkeypatch):
    single = types.SimpleNamespace(heads=["default"])
    multi = types.SimpleNamespace(heads=["a", "b"])
    assert versions.capture_support(multi).supported
    _unpin(monkeypatch, "capture_safe_mask_head")
    assert versions.capture_support(single).supported  # MACE masks only with several heads
    support = versions.capture_support(multi)
    assert not support.supported and "eager" in support.reason


# --------------------------------------------------------------------------- versioned records


def _toy(name, requires=(), selfcheck_passes=True, sources=None, field="block_linear"):
    def check(original, rewritten, context):
        return CheckResult(selfcheck_passes, 0.0 if selfcheck_passes else 1.0, 1e-10, 1,
                           "ok" if selfcheck_passes else "broken on purpose", "float64")

    return Rewrite(name=name, applies_to=lambda model: RewriteSupport(True, "ok"), requires=requires, conflicts=(),
                   mace_versions=">=0.3.16,<0.3.17", source_hashes=sources or {}, exactness="rounding",
                   default="on", workload_constraints=(), apply=lambda model, context: model.log.append(name),
                   selfcheck=check), field


def _registry(*records):
    registry = Registry()
    for rewrite, field in records:
        registry.register(rewrite, config_field=field)
    return registry


def test_a_fake_version_with_matching_hashes_stays_enabled_and_says_so():
    registry = reg.default_registry()
    name = "single_graph_strain"
    assert registry.qualification(name, "0.3.16").reason == "ok"
    other = registry.qualification(name, "0.3.99")
    assert other.supported and "outside" in other.reason and "every pinned source matches" in other.reason


def test_a_fake_hash_disables_the_record_and_its_dependants(caplog):
    here = "fastmace.rewrites.versions:treatments"
    registry = _registry(_toy("a", sources={here: WRONG}), _toy("b", requires=("a",), field="grouped_linear"),
                         _toy("c", field="merged_residual"))
    resolution = registry.resolve(types.SimpleNamespace(), mace_version="0.3.99")
    assert resolution.enabled == ("c",)
    assert "differs on MACE 0.3.99" in resolution.decision("a").reason
    assert resolution.decision("b").reason == "requires a, which is off"
    assert "rewrite a disabled" in caplog.text


# --------------------------------------------------------------------------- rule 13: the self-check reaction


def _chain():
    """a <- b <- c, d independent, e requires d."""
    return _registry(_toy("a"), _toy("b", requires=("a",), field="grouped_linear"),
                     _toy("c", requires=("b",), field="merged_residual"), _toy("d", field="fold_cutoff"),
                     _toy("e", requires=("d",), field="factor_species"))


def test_dependants_are_transitive():
    registry = _chain()
    assert registry.dependants("a") == ("b", "c")
    assert registry.dependants("d") == ("e",)
    assert registry.dependants("c") == ()


def test_disable_with_dependants():
    registry = _chain()
    resolution = registry.resolve(types.SimpleNamespace())
    assert resolution.enabled == ("a", "b", "c", "d", "e")
    reaction = registry.disable_with_dependants(resolution, "b", "relative error 1 > 1e-10")
    assert reaction.disabled == ("b", "c")
    assert reaction.resolution.enabled == ("a", "d", "e")
    assert "self-check failed" in reaction.resolution.decision("b").reason
    assert "relative error 1" in reaction.resolution.decision("b").reason
    assert "requires b" in reaction.resolution.decision("c").reason
    assert reaction.fields == ("grouped_linear", "merged_residual")
    assert resolution.enabled == ("a", "b", "c", "d", "e")  # the old resolution is unchanged


def test_a_failing_selfcheck_raises_with_the_reaction():
    registry = _registry(_toy("a"), _toy("b", requires=("a",), selfcheck_passes=False, field="grouped_linear"),
                         _toy("c", requires=("b",), field="merged_residual"), _toy("d", field="fold_cutoff"))
    resolution = registry.resolve(types.SimpleNamespace())
    model = types.SimpleNamespace(log=[])
    with pytest.raises(SelfCheckError) as caught:
        registry.apply(model, resolution, original=types.SimpleNamespace(log=[]))
    error = caught.value
    assert error.rewrite == "b" and not error.result.passed and error.applied == ("a",)
    assert error.reaction.disabled == ("b", "c") and error.reaction.resolution.enabled == ("a", "d")
    assert "broken on purpose" in str(error)


def test_a_failing_strain_record_leaves_the_process_patch(monkeypatch):
    """Since 2-T08b no registry record installs a process-wide patch, so a failure restores none."""
    registry = reg.default_registry()
    rewrite = registry["single_graph_strain"]
    broken = dataclasses.replace(rewrite, selfcheck=lambda o, r, c: CheckResult(False, 1.0, 1e-10, 1, "no", "float64"))
    monkeypatch.setitem(registry._records, "single_graph_strain", broken)
    resolution = reg.Resolution((reg.Decision("single_graph_strain", "on", True, "on"),))
    _patches.set_single_graph_strain(True)  # the reference calculators' patch, say
    with pytest.raises(SelfCheckError):
        registry.apply(types.SimpleNamespace(), resolution, original=types.SimpleNamespace())
    assert _patches.record_of("single_graph_strain") is not None


def _plan(registry, resolution):
    fields = {registry.config_field(name) for name in registry.names()}
    rewrites = RewriteConfig(**{field: "on" for field in fields})
    decisions = tuple((f"RewriteConfig.{field}", "on", "default") for field in sorted(fields))
    decisions += (("force_precision", "float32", "the ForceField's precision"),)
    return Plan(tier="graph", rewrites=rewrites, decisions=decisions)


def test_the_reaction_revises_the_plan():
    registry = _chain()
    resolution = registry.resolve(types.SimpleNamespace())
    plan = _plan(registry, resolution)
    reaction = registry.disable_with_dependants(resolution, "b", "relative error 1 > 1e-10")
    revised = reactions.selfcheck_revision(plan, reaction)
    assert revised.revision == plan.revision + 1 and revised.predecessor is plan and revised.step == 0
    assert revised.reason.startswith("rule 13: rewrite b failed its self-check")
    assert "b, c off" in revised.reason
    assert revised.rewrites.grouped_linear == "off" and revised.rewrites.merged_residual == "off"
    assert revised.rewrites.block_linear == "on" and revised.rewrites.fold_cutoff == "on"
    key, choice, reason = revised.decision("RewriteConfig.grouped_linear")
    assert choice == "off" and isinstance(reason, Reason) and reason.code == "selfcheck-failed" and reason.rule == 13
    assert revised.decision("RewriteConfig.merged_residual")[2].code == "selfcheck-dependant"
    assert revised.decision("RewriteConfig.block_linear") == plan.decision("RewriteConfig.block_linear")
    # Built again from the revised plan, the registry leaves the reaction's records off.
    again = registry.resolve(types.SimpleNamespace(), registry.requested(revised.rewrites))
    assert again.enabled == ("a", "d", "e")


def test_the_reaction_turns_a_bool_field_off():
    registry = _registry(_toy("u", field="polar_uvu"))
    resolution = registry.resolve(types.SimpleNamespace())
    plan = Plan(tier="graph", rewrites=RewriteConfig(polar_uvu=True),
                decisions=(("RewriteConfig.polar_uvu", "on", "default"),))
    revised = reactions.selfcheck_revision(plan, registry.disable_with_dependants(resolution, "u", "broken"))
    assert revised.rewrites.polar_uvu is False and revised.decision("RewriteConfig.polar_uvu")[1] == "off"


def test_the_reaction_accepts_the_error():
    registry = _registry(_toy("a", selfcheck_passes=False))
    resolution = registry.resolve(types.SimpleNamespace())
    with pytest.raises(SelfCheckError) as caught:
        registry.apply(types.SimpleNamespace(log=[]), resolution, original=types.SimpleNamespace(log=[]))
    revised = reactions.selfcheck_revision(_plan(registry, resolution), caught.value)
    assert revised.rewrites.block_linear == "off"


def test_selfcheck_records():
    result = CheckResult(True, 1e-12, 1e-10, 3, "ok", "float64")
    applied = (reg.AppliedRewrite("a", result), reg.AppliedRewrite("b", None))
    assert reg.selfcheck_records(applied) == (
        {"name": "a", "passed": True, "max_relative_error": 1e-12, "threshold": 1e-10, "modules": 3,
         "dtype": "float64", "detail": "ok"},
        {"name": "b", "passed": None, "max_relative_error": None, "threshold": None, "modules": 0,
         "dtype": None, "detail": "not checked (check=False)"},
    )


# --------------------------------------------------------------------------- the reference record of another MACE


def test_the_qualified_records_are_unchanged():
    from fastmace.reference import records

    # Ids of 1b (restart files and metadata carry them): 2-T04 must not change them.
    assert records.record_for("0.3.16", "mh1").id == MH1_ID
    assert records.record_for("0.3.16", "polar").id == POLAR_ID
    assert records.record_for("0.3.16", "mh1").qualified


def test_an_unqualified_version_gets_stock_equivalence_only():
    from fastmace.reference import records

    record = records.record_for("0.3.15", "mh1")
    assert not record.qualified and record.id.startswith("fastmace.reference/0.3.15/mh1/")
    assert record.upstream_fixes == () and record.optional_physics == ()
    assert record.stock == records.record_for("0.3.16", "mh1").stock  # every pinned source hashes here
    assert [entry["name"] for entry in record.numerical] == ["float64_energy_accumulation"]
    assert record.content_digest() == records.record_for("0.3.16", "mh1").content_digest()
    assert record.digest != records.record_for("0.3.16", "mh1").digest


def test_an_unqualified_record_follows_the_installed_sources(monkeypatch):
    from fastmace.reference import records

    _unpin(monkeypatch, "single_graph_strain")
    monkeypatch.setattr(energy, "SOURCE_HASHES", {key: WRONG for key in energy.SOURCE_HASHES})
    record = records.record_for("0.3.15", "mh1")
    assert record.stock["single_graph_strain"] is False
    assert record.numerical == (records.FLOAT32_SUM_UNQUALIFIED,)
    assert record.content_digest() != records.record_for("0.3.16", "mh1").content_digest()
    # The qualified record never changes with the sources: on 0.3.16 they hash by definition.
    assert records.record_for("0.3.16", "mh1").id == MH1_ID


def test_polar_is_refused_on_an_unqualified_version():
    from fastmace.reference import records

    with pytest.raises(records.ReferencePolicyError, match="PolarMACE"):
        records.record_for("0.3.15", "polar")


def test_content_digest_ignores_only_the_version():
    from fastmace.reference import records

    record = records.record_for("0.3.16", "mh1")
    moved = dataclasses.replace(record, mace_version="0.3.17")
    assert moved.content_digest() == record.content_digest() and moved.id != record.id
    changed = dataclasses.replace(record, numerical=())
    assert changed.content_digest() != record.content_digest()


# --------------------------------------------------------------------------- the force field's energy treatment


def test_native_float64_energy_is_mace_own(monkeypatch):
    import copy

    import numpy as np
    from mace.calculators import MACECalculator

    from _support.structures import nacl
    from _support.tiny_models import FAMILIES, tiny_mace_model
    from fastmace.config import EngineConfig, NeighbourConfig
    from fastmace.model.forces import MACEForceField

    model = tiny_mace_model([11, 17], **FAMILIES["mh1"])
    atoms = nacl(repeat=2, rattle=0.1, seed=8)

    def force_field(dtype):
        calc = MACECalculator(models=[copy.deepcopy(model)], device="cpu", default_dtype=dtype)
        return MACEForceField(calc, atoms, engine=EngineConfig(compile_mode="none"),
                              neighbours=NeighbourConfig(skin=0.5, sort_receivers="off", gpu_neighbour_list="off"))

    assert force_field("float64").energy_treatment == "patched"
    monkeypatch.setattr(energy, "SOURCE_HASHES", {key: WRONG for key in energy.SOURCE_HASHES})
    monkeypatch.setattr(energy, "_FORWARDS", {})
    native = force_field("float64")
    assert native.energy_treatment == "native" and native.info()["energy_treatment"] == "native"
    stock = atoms.copy()
    stock.calc = MACECalculator(models=[copy.deepcopy(model)], device="cpu", default_dtype="float64")
    e, f, _ = native.evaluate(atoms)
    assert e == pytest.approx(stock.get_potential_energy(), rel=1e-12)  # other edge order: summation order
    np.testing.assert_allclose(f, stock.get_forces(), rtol=0, atol=1e-10)
    with pytest.raises(ValueError, match="qualified float64 energy forward"):
        native.per_atom_energies(atoms)
    with pytest.raises(energy.EnergyReductionError):
        force_field("float32")


def test_force_field_identity_takes_the_effective_precision(monkeypatch, tmp_path):
    pytest.importorskip("cuequivariance_torch")
    import torch

    from _support.structures import nacl
    from _support.tiny_models import FAMILIES, tiny_mace_model
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField

    path = tmp_path / "tiny-mh1-11-17.model"
    torch.save(tiny_mace_model([11, 17], **FAMILIES["mh1"]), path)
    loading._LICENCES_SHOWN.add(str(path))
    atoms = nacl(repeat=1)
    ff = ForceField(loading.load_pristine(str(path), trusted=True), atoms, precision="float32")
    assert ff.effective_precision() == ("float32", None) and ff.identity.force_precision == "float32"
    monkeypatch.setattr(energy, "SOURCE_HASHES", {key: WRONG for key in energy.SOURCE_HASHES})
    precision, reason = ff.effective_precision()
    assert precision == "float64" and reason.code == versions.PROMOTION_CODE
    assert ff.identity.force_precision == "float64" and ff.precision == "float32"
    assert ff.effective_precision(float64_energy=False) == ("float32", None)


# --------------------------------------------------------------------------- planner, gate yardstick, report


@pytest.fixture
def tiny_mh1(tmp_path):
    pytest.importorskip("cuequivariance_torch")
    import torch

    from _support.tiny_models import FAMILIES, tiny_mace_model
    from fastmace.model import loading

    path = tmp_path / "tiny-mh1-11-17.model"
    torch.save(tiny_mace_model([11, 17], **FAMILIES["mh1"]), path)
    loading._LICENCES_SHOWN.add(str(path))
    return loading.load_pristine(str(path), trusted=True)


def _cpu_planner():
    from fastmace.config import EngineConfig
    from fastmace.engine import constraints
    from fastmace.planner import minimal

    return minimal.MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ={},
                                  process=constraints.ProcessRecord())


def _langevin():
    from fastmace.config import IntegratorConfig

    return IntegratorConfig(temperature_K=300.0)


def _md_plan(planner, loaded, precision="float32"):
    from _support.structures import nacl
    from fastmace.model.forcefield import ForceField
    from fastmace.planner.workload import Workload

    atoms = nacl(repeat=1)
    ff = ForceField(loaded, atoms, precision=precision)
    return planner.plan(ff, atoms, _langevin(), Workload(max_temperature_K=300.0)), ff, atoms


def test_the_default_never_promotes(tiny_mh1):
    """MACE 0.3.16, every hash matches: the plan keeps the ForceField's precision and records no request."""
    assert energy.source_problems() == []
    planner = _cpu_planner()
    for precision in ("float32", "float64"):
        plan, ff, atoms = _md_plan(planner, tiny_mh1, precision)
        key, choice, reason = plan.decision("force_precision")
        assert choice == precision and getattr(reason, "code", None) is None
        assert "force_precision" not in dict(plan.requested)
        assert planner.plan_evaluation(ff, atoms).decision("force_precision")[1] == precision


def test_the_planner_promotes_on_an_unhashed_energy_forward(monkeypatch, tiny_mh1):
    from fastmace import report

    monkeypatch.setattr(energy, "SOURCE_HASHES", {key: WRONG for key in energy.SOURCE_HASHES})
    monkeypatch.setattr(energy, "_FORWARDS", {})
    planner = _cpu_planner()
    plan, ff, atoms = _md_plan(planner, tiny_mh1, "float32")
    key, choice, reason = plan.decision("force_precision")
    assert choice == "float64" and reason.code == versions.PROMOTION_CODE
    assert dict(plan.requested)["force_precision"] == "float32"
    assert plan.precision.dtype == "float64" and plan.revision == 0
    assert ff.identity.force_precision == "float64"
    assert versions.PROMOTION_CODE in report.DOWNGRADES
    lines = report.Report(model="tiny", structure="NaCl", n_atoms=len(atoms), plan=plan)._why()
    assert any("promoted from float32" in text for _, text in lines)
    assert planner.plan_evaluation(ff, atoms).decision("force_precision")[1] == "float64"
    engine = planner.build(plan, ff, atoms, check=False)
    assert engine.force_field.energy_treatment == "native"


def test_polar_is_refused_by_the_planner_without_its_forward(monkeypatch):
    pytest.importorskip("cuequivariance_torch")
    from _support.structures import nacl
    from _support.tiny_models import FAMILIES, polar_available, tiny_mace_model

    if not polar_available():
        pytest.skip("PolarMACE needs graph_longrange")
    from fastmace.errors import PlanRefused
    from fastmace.longrange import polar
    from fastmace.model import capabilities
    from fastmace.planner.workload import Workload

    traits = capabilities.model_traits(tiny_mace_model([11, 17], **FAMILIES["polar"]))
    planner = _cpu_planner()
    arguments = dict(precision="float32")
    planner.plan_traits(traits, nacl(repeat=1), _langevin(), Workload(max_temperature_K=300.0), planner.hardware(),
                        **arguments)
    monkeypatch.setattr(polar, "SOURCE_HASHES", {key: WRONG for key in polar.SOURCE_HASHES})
    with pytest.raises(PlanRefused) as caught:
        planner.plan_traits(traits, nacl(repeat=1), _langevin(), Workload(max_temperature_K=300.0),
                            planner.hardware(), **arguments)
    assert "polar-unqualified-source" in [getattr(r, "code", None) for r in caught.value.reasons]


def test_build_records_the_selfchecks(tiny_mh1):
    planner = _cpu_planner()
    plan, ff, atoms = _md_plan(planner, tiny_mh1)
    engine = planner.build(plan, ff, atoms)
    built = engine.plan
    assert built == dataclasses.replace(plan, selfchecks=built.selfchecks)
    assert built.selfchecks and plan.selfchecks == ()
    assert [check.name for check in built.selfchecks] == [entry.name for entry in engine.force_field.applied]
    assert all(check.passed for check in built.selfchecks)
    assert f"self-checks: {len(built.selfchecks)} records" in built.describe()
    hash(built)  # still a hashable frozen record


def test_the_unqualified_yardstick_keeps_mace_float32_sum(monkeypatch):
    pytest.importorskip("cuequivariance_torch")
    import copy

    from mace.calculators import MACECalculator
    from mace.cli.convert_e3nn_cueq import run

    from _support.structures import nacl
    from _support.tiny_models import FAMILIES, tiny_mace_model
    from fastmace.model import compile as fc
    from fastmace.reference import calculators, records

    model = tiny_mace_model([11, 17], **FAMILIES["mh1"])
    atoms = nacl(repeat=2, rattle=0.1, seed=8)

    def calculator():
        calc = MACECalculator(models=[copy.deepcopy(model)], device="cpu", default_dtype="float32")
        with fc.naive_cueq():
            calc.models = [run(calc.models[0], device="cpu", return_model=True)]
        return calc

    qualified = calculators.ReferenceCalculator.from_calculator(calculator(), records.record_for("0.3.15", "mh1"),
                                                                atoms)
    assert qualified.info()["energy_bound"] == "float64 accumulation"
    monkeypatch.setattr(energy, "SOURCE_HASHES", {key: WRONG for key in energy.SOURCE_HASHES})
    monkeypatch.setattr(energy, "_FORWARDS", {})
    record = records.record_for("0.3.15", "mh1")
    yardstick = calculators.ReferenceCalculator.from_calculator(calculator(), record, atoms)
    info = yardstick.info()
    assert info["role"] == "yardstick" and info["float64_energy"] is False
    assert info["energy_bound"] == records.ENERGY_BOUND_UNQUALIFIED
    stock = atoms.copy()
    stock.calc = calculator()
    assert yardstick.evaluate(atoms)[0] == pytest.approx(stock.get_potential_energy(), rel=1e-6)


def test_the_report_names_the_looser_energy_bound():
    from fastmace import report
    from fastmace.reference import records

    gate = {"status": "passed", "copies": "2 rattled copies", "energy_meV_per_atom": 0.1,
            "references": {"yardstick": {"energy_bound": records.ENERGY_BOUND_UNQUALIFIED}}}
    text = report.Report(model="tiny", structure="NaCl", n_atoms=8, plan=Plan(tier="graph"), gate=gate)._gate()
    assert "energy bound looser" in text
    gate["references"]["yardstick"]["energy_bound"] = "float64 accumulation"
    assert "energy bound" not in report.Report(model="tiny", structure="NaCl", n_atoms=8, plan=Plan(tier="graph"),
                                               gate=gate)._gate()


def test_conv_fusion_scope_on_an_unhashed_converter(monkeypatch):
    _unpin(monkeypatch, "cueq_conv_fusion")
    # MACE's own rule fuses only for the device string "cuda": asking for that needs no patch.
    with _patches.cueq_conv_fusion(False, "cpu") as record:
        assert record is None and _patches.record_of("cueq_conv_fusion") is None
    with _patches.cueq_conv_fusion(True, "cuda") as record:
        assert record is None
    with pytest.raises(_patches.UnqualifiedPatchError):
        with _patches.cueq_conv_fusion(True, "cuda:0"):
            pass
    with pytest.raises(_patches.UnqualifiedPatchError):
        with _patches.cueq_conv_fusion(False):
            pass
