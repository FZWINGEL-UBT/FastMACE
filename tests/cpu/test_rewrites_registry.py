"""The rewrite registry (SPEC §6.1, §6.3, §6.4; plan task 1a-T06) on toy records.

Toy records act on a toy model (a SimpleNamespace that logs what was applied), so the rules
are tested without MACE: application order, ``requires`` and ``conflicts``, the version and
source-hash qualification, the never-default rule for ``approximate`` rewrites, the self-check
runner's thresholds, and the registration of the polar records.
"""

import copy
import hashlib
import inspect
import logging
import types

import pytest

from fastmace.config import RewriteConfig
from fastmace.model.capabilities import RewriteSupport
from fastmace.rewrites import registry as reg
from fastmace.rewrites.registry import (
    CheckResult,
    Context,
    Registry,
    Rewrite,
    RewriteRefusedError,
    SelfCheckError,
)

MACE = "0.3.16"


def _here_sha256():
    return hashlib.sha256(inspect.getsource(_toy_target).encode()).hexdigest()


def _toy_target():
    """A function whose source the toy records pin."""
    return 1


TOY_QUALIFIED = f"{__name__}:_toy_target"


def toy(name, config_field="block_linear", **fields):
    """A toy record that appends its name to model.log when applied; self-check always passes."""
    values = dict(
        name=name,
        applies_to=lambda model: RewriteSupport(True, "ok"),
        requires=(),
        conflicts=(),
        mace_versions=">=0.3.16,<0.3.17",
        source_hashes={},
        exactness="rounding",
        default="on",
        workload_constraints=(),
        apply=lambda model, context: model.log.append(name),
        selfcheck=lambda original, rewritten, context: CheckResult(True, 0.0, 1e-10, 1, dtype="float64"),
    )
    values.update(fields)
    return Rewrite(**values), config_field


def registry_of(*records):
    registry = Registry()
    for rewrite, field in records:
        registry.register(rewrite, config_field=field)
    return registry


def model():
    return types.SimpleNamespace(log=[])


# --------------------------------------------------------------------------- the record


def test_record_has_exactly_the_spec_fields():
    names = [field.name for field in reg.dataclasses.fields(Rewrite)]
    assert names == ["name", "applies_to", "requires", "conflicts", "mace_versions", "source_hashes",
                     "exactness", "default", "workload_constraints", "apply", "selfcheck"]  # SPEC §6.1


def test_record_normalises_and_validates():
    rewrite, _ = toy("a", requires=["b"], mace_versions=">=0.3.16")
    assert rewrite.requires == ("b",)
    assert str(rewrite.mace_versions) == ">=0.3.16"
    with pytest.raises(ValueError, match="exactness"):
        toy("a", exactness="close")
    with pytest.raises(ValueError, match="default"):
        toy("a", default="yes")
    with pytest.raises(ValueError, match="itself"):
        toy("a", requires=("a",))


def test_an_approximate_record_cannot_be_a_default():
    for default in ("on", "auto"):
        with pytest.raises(ValueError, match="approximate"):
            toy("a", exactness="approximate", default=default)
    rewrite, _ = toy("a", exactness="approximate", default="off")
    assert rewrite.default == "off"


def test_record_from_dict_takes_the_spec_fields_and_config_field():
    rewrite, _ = toy("a")
    fields = {field.name: getattr(rewrite, field.name) for field in reg.dataclasses.fields(Rewrite)}
    registry = Registry()
    registry.register({**fields, "config_field": "RewriteConfig.grouped_linear"})
    assert registry.config_field("a") == "grouped_linear"
    with pytest.raises(ValueError, match="unknown"):
        Registry().register({**fields, "colour": "red"})


def test_register_refuses_duplicates_and_unknown_config_fields():
    registry = registry_of(toy("a"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(toy("a")[0], config_field="block_linear")
    with pytest.raises(ValueError, match="RewriteConfig"):
        registry.register(toy("b")[0], config_field="no_such_field")


# --------------------------------------------------------------------------- order


def test_order_is_registration_order_with_requirements_first():
    registry = registry_of(toy("c", requires=("b",)), toy("a"), toy("b", requires=("a",)), toy("d"))
    assert registry.order() == ("a", "b", "c", "d")
    assert [rewrite.name for rewrite in registry] == ["a", "b", "c", "d"]


def test_order_refuses_a_cycle():
    registry = registry_of(toy("a", requires=("b",)), toy("b", requires=("a",)))
    with pytest.raises(ValueError, match="cycle"):
        registry.order()


def test_apply_runs_in_order_and_passes_what_was_applied():
    seen = []

    def remember(name):
        def apply(model, context):
            seen.append((name, context.applied))
            model.log.append(name)
        return apply

    registry = registry_of(toy("b", requires=("a",), apply=remember("b")), toy("a", apply=remember("a")))
    target = model()
    resolution = registry.resolve(target, {}, mace_version=MACE)
    applied = registry.apply(target, resolution, original=model())
    assert target.log == ["a", "b"] and [entry.name for entry in applied] == ["a", "b"]
    assert seen == [("a", ()), ("b", ("a",))]


# --------------------------------------------------------------------------- resolution


def test_off_and_unsupported_records_are_disabled_with_reasons():
    registry = registry_of(toy("a"), toy("b", applies_to=lambda m: RewriteSupport(False, "no such layer")))
    resolution = registry.resolve(model(), {"a": "off"}, mace_version=MACE)
    assert resolution.enabled == ()
    assert resolution.decision("a").reason == "off"
    assert resolution.decision("b").reason == "no such layer"


def test_on_for_an_unsupported_record_is_refused():
    registry = registry_of(toy("b", applies_to=lambda m: RewriteSupport(False, "no such layer")))
    with pytest.raises(RewriteRefusedError, match="no such layer"):
        registry.resolve(model(), {"b": "on"}, mace_version=MACE)
    lenient = registry.resolve(model(), {"b": "on"}, mace_version=MACE, strict=False)
    assert lenient.enabled == () and lenient.decision("b").refused


def test_requires_disables_auto_and_refuses_on():
    registry = registry_of(toy("a"), toy("b", requires=("a",)), toy("c", requires=("missing",)))
    resolution = registry.resolve(model(), {"a": "off"}, mace_version=MACE)
    assert resolution.enabled == ()
    assert "requires a" in resolution.decision("b").reason
    assert "missing" in resolution.decision("c").reason and "not registered" in resolution.decision("c").reason
    with pytest.raises(RewriteRefusedError, match="requires a"):
        registry.resolve(model(), {"a": "off", "b": "on"}, mace_version=MACE)
    assert registry.resolve(model(), {}, mace_version=MACE, strict=False).enabled == ("a", "b")


def test_conflicts_are_symmetric():
    registry = registry_of(toy("a"), toy("b", conflicts=("a",)), toy("c"), toy("d"))
    resolution = registry.resolve(model(), {}, mace_version=MACE)
    assert resolution.enabled == ("a", "c", "d")
    assert "conflicts with a" in resolution.decision("b").reason
    with pytest.raises(RewriteRefusedError, match="conflicts"):
        registry.resolve(model(), {"b": "on"}, mace_version=MACE)
    # a conflict declared by the earlier record counts as well
    registry = registry_of(toy("a", conflicts=("b",)), toy("b"))
    assert registry.resolve(model(), {}, mace_version=MACE).enabled == ("a",)
    assert registry.resolve(model(), {"a": "off"}, mace_version=MACE).enabled == ("b",)


def test_approximate_is_never_enabled_by_default():
    registry = registry_of(toy("a", exactness="approximate", default="off"))
    resolution = registry.resolve(model(), {"a": "auto"}, mace_version=MACE)
    assert resolution.enabled == () and "default" in resolution.decision("a").reason
    assert registry.resolve(model(), {"a": "on"}, mace_version=MACE).enabled == ("a",)


def test_default_off_needs_an_explicit_on():
    registry = registry_of(toy("a", default="off"), toy("b", default="auto"))
    assert registry.resolve(model(), {}, mace_version=MACE).enabled == ("b",)
    assert registry.resolve(model(), {"a": "on"}, mace_version=MACE).enabled == ("a", "b")


def test_requested_reads_rewrite_config_fields():
    registry = registry_of(toy("a", config_field="grouped_linear"), toy("p", config_field="polar_uvu"))
    config = RewriteConfig(grouped_linear="off", polar_uvu=True)
    assert registry.requested(config) == {"a": "off", "p": "auto"}
    assert registry.requested(RewriteConfig(polar_uvu=False))["p"] == "off"
    resolution = registry.resolve(model(), config, mace_version=MACE)
    assert resolution.enabled == ("p",)


def test_requested_rejects_unknown_names_and_values():
    registry = registry_of(toy("a"))
    with pytest.raises(ValueError, match="unknown rewrite"):
        registry.resolve(model(), {"zz": "on"}, mace_version=MACE)
    with pytest.raises(ValueError, match="on"):
        registry.resolve(model(), {"a": "yes"}, mace_version=MACE)


def test_candidates_are_the_planner_input():
    registry = registry_of(toy("a", workload_constraints=("single_graph",)),
                           toy("b", requires=("a",), applies_to=lambda m: RewriteSupport(False, "nope")))
    candidates = registry.candidates(model(), mace_version=MACE)
    assert [c.name for c in candidates] == ["a", "b"]
    a, b = candidates
    assert a.available and a.workload_constraints == ("single_graph",) and a.config_field == "block_linear"
    assert a.default == "on" and a.exactness == "rounding"
    assert not b.available and b.support == RewriteSupport(False, "nope") and b.requires == ("a",)


# --------------------------------------------------------------------------- versions and source hashes


def test_source_sha256_is_inspect_getsource_as_the_research_engine():
    assert reg.source_sha256(TOY_QUALIFIED) == _here_sha256()


def test_matching_hashes_qualify_on_any_version():
    rewrite, field = toy("a", source_hashes={TOY_QUALIFIED: _here_sha256()})
    registry = registry_of((rewrite, field))
    assert registry.qualification("a", mace_version="0.3.16").supported
    other = registry.qualification("a", mace_version="0.3.17")
    assert other.supported and "0.3.17" in other.reason


def test_a_hash_mismatch_disables_with_a_logged_reason(caplog):
    registry = registry_of(toy("a", source_hashes={TOY_QUALIFIED: "0" * 64}), toy("b"))
    with caplog.at_level(logging.WARNING, logger="fastmace.rewrites"):
        resolution = registry.resolve(model(), {}, mace_version="0.3.17")
    assert resolution.enabled == ("b",)
    reason = resolution.decision("a").reason
    assert "source" in reason and TOY_QUALIFIED in reason
    assert any(TOY_QUALIFIED in record.getMessage() for record in caplog.records)
    with pytest.raises(RewriteRefusedError):
        registry.resolve(model(), {"a": "on"}, mace_version="0.3.17")


def test_a_hash_mismatch_disables_inside_the_version_range_too():
    registry = registry_of(toy("a", source_hashes={TOY_QUALIFIED: "0" * 64}))
    assert not registry.qualification("a", mace_version=MACE).supported


def test_an_unresolvable_source_disables():
    registry = registry_of(toy("a", source_hashes={"fastmace_no_such_module:f": "0" * 64}))
    qualification = registry.qualification("a", mace_version=MACE)
    assert not qualification.supported and "fastmace_no_such_module" in qualification.reason


def test_installed_mace_version_is_used_by_default():
    pytest.importorskip("mace")
    assert reg.installed_mace_version() == "0.3.16"


# --------------------------------------------------------------------------- self-check runner


def test_thresholds_are_the_spec_values():
    assert dict(reg.THRESHOLDS) == {"float64": 1e-10, "float32": 1e-4}  # SPEC §6.4
    torch = pytest.importorskip("torch")
    assert reg.threshold_for(torch.float64) == 1e-10 and reg.threshold_for("float32") == 1e-4
    with pytest.raises(ValueError):
        reg.threshold_for(torch.float16)


def test_check_tensors_measures_relative_error():
    torch = pytest.importorskip("torch")
    expected = torch.tensor([1.0, -2.0, 4.0], dtype=torch.float64)
    passed = reg.check_tensors([("m", expected, expected + 1e-12)])
    assert passed.passed and passed.modules == 1 and passed.dtype == "float64"
    assert passed.max_relative_error == pytest.approx(0.25e-12)
    failed = reg.check_tensors([("m", expected, expected * (1 + 1e-9))])
    assert not failed.passed and "m:" in failed.detail
    loose = reg.check_tensors([("m", expected.float(), expected.float() * (1 + 1e-6))])
    assert loose.passed and loose.threshold == 1e-4


def test_runner_enforces_the_spec_threshold():
    lying, _ = toy("a", selfcheck=lambda o, r, c: CheckResult(True, 1e-6, 1e-3, 1, dtype="float64"))
    registry = registry_of((lying, "block_linear"))
    result = registry.selfcheck("a", model(), model(), Context())
    assert not result.passed and result.threshold == 1e-10
    with pytest.raises(SelfCheckError, match="a"):
        registry.apply(model(), registry.resolve(model(), {}, mace_version=MACE), original=model())


def test_runner_accepts_dict_results_and_refuses_loose_thresholds():
    as_dict, _ = toy("a", selfcheck=lambda o, r, c: {"passed": True, "max_relative_error": 0.0,
                                                    "threshold": 1e-10, "modules": 2, "detail": "ok"})
    loose, _ = toy("b", selfcheck=lambda o, r, c: {"passed": True, "max_relative_error": 1e-3,
                                                  "threshold": 1e-2, "modules": 1, "detail": "ok"})
    registry = registry_of((as_dict, "block_linear"), (loose, "block_linear"))
    assert registry.selfcheck("a", model(), model(), Context()).passed
    assert not registry.selfcheck("b", model(), model(), Context()).passed


def test_apply_without_original_checks_against_a_copy():
    originals = []

    def check(original, rewritten, context):
        originals.append(list(original.log))
        return CheckResult(True, 0.0, 1e-10, 1, dtype="float64")

    registry = registry_of(toy("a", selfcheck=check), toy("b", selfcheck=check))
    target = model()
    registry.apply(target, registry.resolve(target, {}, mace_version=MACE))
    assert originals == [[], []] and target.log == ["a", "b"]  # both against the model before any rewrite
    target = model()
    registry.apply(target, registry.resolve(target, {}, mace_version=MACE), check=False)
    assert target.log == ["a", "b"]


def test_apply_records_the_check_and_what_apply_returned():
    registry = registry_of(toy("a", apply=lambda model, context: ["x", "y"]))
    target = model()
    (entry,) = registry.apply(target, registry.resolve(target, {}, mace_version=MACE), original=copy.copy(target))
    assert entry.name == "a" and entry.result.passed and entry.detail == ["x", "y"]


# --------------------------------------------------------------------------- the default registry


def test_default_registry_holds_the_linear_and_polar_records():
    registry = reg.default_registry()
    assert {"block_linear", "grouped_linear", "merged_residual", "polar_uvu"} <= set(registry.names())
    assert registry.config_field("polar_uvu") == "polar_uvu"
    order = registry.order()
    assert order.index("block_linear") < order.index("grouped_linear") < order.index("merged_residual")
    assert reg.default_registry() is not registry  # a fresh registry each time, no shared state


def test_polar_record_resolves():
    registry = reg.default_registry()
    plain = types.SimpleNamespace(named_modules=lambda: iter(()))
    resolution = registry.resolve(plain, {name: "off" for name in registry.names() if name != "polar_uvu"},
                                  mace_version=MACE)
    decision = resolution.decision("polar_uvu")
    assert not decision.enabled and "not PolarMACE" in decision.reason
    rewrite = registry["polar_uvu"]
    assert rewrite.exactness == "rounding" and rewrite.default == "on"
    assert set(rewrite.source_hashes) == {"mace.modules.field_blocks:SparseUvuTensorProduct.forward",
                                          "mace.modules.field_blocks:SparseUvuTensorProduct.__init__"}


def test_polar_source_hashes_qualify_on_the_installed_mace():
    pytest.importorskip("mace.modules.field_blocks")
    assert reg.default_registry().qualification("polar_uvu").supported


# The research engine's source pins (langevin/gpu_langevin.py: _NONLINEAR_FORWARD_SHA256 and
# GENERIC_INTERACTIONS) at tag research-engine-baseline-20261017; plan tasks 1a-T07 and 1a-T08
# carry them in their records' source_hashes.
RESEARCH_ENGINE_PINS = {
    "mace.modules.blocks:RealAgnosticResidualNonLinearInteractionBlock.forward":
        "727c37b796de7d3b33793ae6317a6d09c1f4c8677b28c04963ffbb34899b0b44",
    "mace.modules.blocks:RealAgnosticResidualInteractionBlock.forward":
        "5148c60b11fd7dcb8fca624aa0447b9e508fcb4d7869a2b3d79e1ac14f768b62",
    "mace.modules.blocks:RealAgnosticDensityInteractionBlock.forward":
        "80fd7bc7ea53c17b3212db3ff42ac87b6e42ee8e986c9b408aec06922ad91dad",
    "mace.modules.blocks:RealAgnosticDensityResidualInteractionBlock.forward":
        "228610bc97daa842d6ecc8770142d1ad4bce24cc3060fecebc13e3df726a8316",
}


@pytest.mark.parametrize("qualified", sorted(RESEARCH_ENGINE_PINS))
def test_source_sha256_reproduces_the_research_engine_pins(qualified):
    pytest.importorskip("mace.modules.blocks")
    assert reg.source_sha256(qualified) == RESEARCH_ENGINE_PINS[qualified]


def test_polar_record_applies_through_the_registry():
    from _support.tiny_models import polar_available, tiny_family_calculator
    if not polar_available():
        pytest.skip("graph_longrange is not installed")
    from fastmace.rewrites.polar import uvu
    model = tiny_family_calculator("polar", [11, 17]).models[0]
    registry = reg.default_registry()
    rewritten = copy.deepcopy(model)
    resolution = registry.resolve(rewritten, RewriteConfig(), strict=False)
    assert "polar_uvu" in resolution.enabled
    only_polar = reg.Resolution(tuple(d for d in resolution.decisions if d.name == "polar_uvu"))
    (entry,) = registry.apply(rewritten, only_polar, original=model)
    assert entry.result.passed and uvu.is_applied(rewritten) and not uvu.is_applied(model)


# --------------------------------------------------------------------------- _patches: polar_longrange


def test_patches_report_the_polar_longrange_record(monkeypatch):
    import sys

    from fastmace import _patches
    fields = {"name": "polar_longrange", "value": True, "targets": ("mace.modules.extensions:PolarMACE.forward",),
              "detail": "class-level PolarMACE wrappers"}
    fake = types.SimpleNamespace(installed=lambda: dict(fields))
    monkeypatch.setitem(sys.modules, "fastmace.longrange.polar", fake)
    record = _patches.record_of("polar_longrange")
    assert record == _patches.PatchRecord("polar_longrange", True, fields["targets"], fields["detail"])
    assert record in _patches.applied()
    assert {"name": "polar_longrange", "value": True, "targets": list(fields["targets"]),
            "detail": fields["detail"]} in _patches.provenance()
    with pytest.raises(RuntimeError, match="its own module"):
        _patches.restore("polar_longrange")
    fake.installed = lambda: None
    assert _patches.record_of("polar_longrange") is None and _patches.restore("polar_longrange") is False


def test_patches_read_the_real_polar_module():
    from fastmace import _patches
    from fastmace.longrange import polar
    assert polar.PATCH_NAME == "polar_longrange"
    expected = polar.installed()
    record = _patches.record_of("polar_longrange")
    assert (record is None) == (expected is None)
    if record is not None:
        assert record.targets == tuple(expected["targets"])
