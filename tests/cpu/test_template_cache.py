"""The converted-model template cache of ``ForceField.execution_model`` (opt-1 OPT-T06).

Design: ``docs/design/opt-T06-template-cache.md``. On CUDA with the cuEq or hybrid backend, a plan's
execution model after the first is a private copy of a cached template instead of a reload and
conversion. These tests run on the CPU with a fake converted model: ``loading.load`` is replaced by
a counter that returns a ``LoadedModel`` around a small torch module, and the CUDA rule
(``templates.applies``) is forced on where a test needs the cache. The real conversion and the
bitwise equality of copy and load are the GPU test's (``tests/gpu/test_template_cache_gpu.py``).
"""

from __future__ import annotations

import dataclasses
import gc
import types
import weakref
from pathlib import Path

import pytest
import torch

from fastmace import _patches
from fastmace.config import EngineConfig
from fastmace.model import forcefield as forcefield_module
from fastmace.model import loading, templates
from fastmace.model.forcefield import ForceField


class FakeCalculator:
    """The parts of a MACE calculator a plan reads, around one model."""

    def __init__(self, model):
        self.models = [model]
        self.results = {"energy": 0.0}
        self.r_max = 4.0
        self.head = None


def fake_loaded(seed=0, *, backend="hybrid", dtype="float32", device="cuda:0"):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.SiLU(), torch.nn.Linear(8, 1))
    model.register_buffer("scale", torch.ones(1))
    return loading.LoadedModel(
        calculator=FakeCalculator(model), path=Path("fake.model"), sha256="0" * 64, trust="sha256",
        checkpoint_source="explicit", label=None, head=None, heads=(), model_type="MACE", family=None,
        functional=None, licence=None, backend=backend, dtype=dtype, device=device, cueq_conv_fusion=None,
        fused_cueq={}, embedding_specs=None, charge=None, spin_multiplicity=None)


@pytest.fixture
def loads(monkeypatch):
    """loading.load replaced by a counter; returns the list of the keyword arguments of each call."""
    calls = []

    def load(source, head=None, **options):
        calls.append(dict(options, source=source, head=head))
        return fake_loaded(backend=options["backend"], dtype=options["dtype"], device=options["device"])

    monkeypatch.setattr(loading, "load", load)
    return calls


@pytest.fixture
def cuda_rule(monkeypatch):
    """The cache's CUDA cuEq/hybrid rule forced on, so the CPU can exercise it."""
    monkeypatch.setattr(templates, "applies", lambda backend, device: True)


def pristine():
    return dataclasses.replace(fake_loaded(), backend="e3nn", dtype="float64", device="cpu", label=None)


def plan(backend="hybrid", dtype="float32", device="cuda:0", cache=True, fusion="auto"):
    engine = EngineConfig(backend=backend, device=device, model_template_cache=cache, cueq_conv_fusion=fusion)
    return types.SimpleNamespace(engine=engine, precision=types.SimpleNamespace(dtype=dtype), allowances=())


def tensors(model):
    return list(model.parameters()) + list(model.buffers())


def storages(loaded):
    return {t.untyped_storage().data_ptr() for t in tensors(loaded.model)}


# --------------------------------------------------------------------------- the rule and the field


def test_the_field_is_on_by_default():
    assert EngineConfig().model_template_cache is True


@pytest.mark.parametrize(("backend", "device", "expected"), [
    ("hybrid", "cuda:0", True), ("cueq", "cuda", True), ("cueq", "cuda:1", True),
    ("cueq", "cpu", False), ("hybrid", "cpu", False), ("e3nn", "cuda:0", False), ("e3nn", "cpu", False)])
def test_templates_apply_to_cuda_cueq_and_hybrid_only(backend, device, expected):
    # On the CPU a deepcopy of a cuEq model rebuilds its naive kernels (41 s, as long as the reload), and
    # e3nn has no conversion to save (copy 0.74 s against reload 0.39 s): opt-T06 CPU benchmark.
    assert templates.applies(backend, device) is expected


def test_no_template_where_the_rule_says_no(loads):
    ff = ForceField(pristine())
    for _ in range(3):
        ff.execution_model(plan(device="cpu"))
    assert len(loads) == 3 and ff.template_count() == 0


def test_the_opt_out_reloads_every_plan(loads, cuda_rule):
    ff = ForceField(pristine())
    for _ in range(3):
        ff.execution_model(plan(cache=False))
    assert len(loads) == 3 and ff.template_count() == 0


# --------------------------------------------------------------------------- first plan, later plans


def test_first_plan_gets_the_load_and_later_plans_a_copy(loads, cuda_rule):
    ff = ForceField(pristine())
    first = ff.execution_model(plan())
    second = ff.execution_model(plan())
    third = ff.execution_model(plan())
    assert len(loads) == 1  # one conversion for three plans
    assert ff.template_count() == 1
    assert first.model is not second.model is not third.model
    for later in (second, third):
        assert torch.equal(torch.cat([t.flatten() for t in tensors(later.model)]),
                           torch.cat([t.flatten() for t in tensors(first.model)]))
        assert later.sha256 == first.sha256 and later.backend == first.backend and later.dtype == first.dtype


def test_copies_share_no_storage_module_or_results(loads, cuda_rule):
    ff = ForceField(pristine())
    plans = [ff.execution_model(plan()) for _ in range(3)]
    template = ff._templates.template(next(iter(ff._templates.keys())))
    everyone = plans + [template]
    for i, a in enumerate(everyone):
        for b in everyone[i + 1:]:
            assert not storages(a) & storages(b)
            assert not {id(m) for m in a.model.modules()} & {id(m) for m in b.model.modules()}
            assert a.calculator is not b.calculator
            assert a.calculator.results is not b.calculator.results
            assert a.calculator.models is not b.calculator.models


def test_a_rewritten_plan_does_not_leak_into_the_next(loads, cuda_rule):
    # A plan's rewrites act on its own model (here: weights changed in place and a module replaced);
    # the next plan, a fallback candidate for example, starts from the pristine template.
    ff = ForceField(pristine())
    first = ff.execution_model(plan())
    reference = [t.detach().clone() for t in tensors(first.model)]
    with torch.no_grad():
        for t in tensors(first.model):
            t.mul_(3.0)
    first.model[1] = torch.nn.Identity()
    second = ff.execution_model(plan())
    assert isinstance(second.model[1], torch.nn.SiLU)
    assert all(torch.equal(a, b) for a, b in zip(tensors(second.model), reference))


# --------------------------------------------------------------------------- the key


@pytest.mark.parametrize(("base", "change"), [
    ({}, dict(dtype="float64")), ({}, dict(backend="cueq")), ({}, dict(device="cuda:1")),
    (dict(backend="cueq"), dict(backend="cueq", fusion="on"))])  # conv fusion is a plain cuEq option
def test_each_conversion_option_is_its_own_template(loads, cuda_rule, base, change):
    ff = ForceField(pristine())
    ff.execution_model(plan(**base))
    ff.execution_model(plan(**change))
    assert len(loads) == 2 and ff.template_count() == 2
    ff.execution_model(plan(**change))
    assert len(loads) == 2


@pytest.mark.parametrize("name", templates.CONVERSION_PATCHES)
def test_a_changed_conversion_patch_misses(loads, cuda_rule, monkeypatch, name):
    ff = ForceField(pristine())
    ff.execution_model(plan())
    applied = _patches.applied()
    record = _patches.PatchRecord(name, "another value", (), "installed since the first plan")
    monkeypatch.setattr(_patches, "applied", lambda: tuple(applied) + (record,))
    ff.execution_model(plan())
    assert len(loads) == 2


@pytest.mark.parametrize("name", ["capture_safe_mask_head", "first_derivative_only", "unfuse_full_addmm",
                                  "polar_longrange"])
def test_an_engine_patch_installed_by_the_first_build_still_hits(loads, cuda_rule, monkeypatch, name):
    # The first plan's build installs process-wide engine patches (compile.install_engine_patches). They act
    # when MACE's functions are called, not on the converted modules, so the next plan copies the template
    # (murphy, opt-T06: with every patch in the key, a second Simulation missed and kept 2 templates).
    ff = ForceField(pristine())
    ff.execution_model(plan())
    applied = _patches.applied()
    record = _patches.PatchRecord(name, True, ("mace.modules.blocks.mask_head",), "installed by a build")
    monkeypatch.setattr(_patches, "applied", lambda: tuple(applied) + (record,))
    ff.execution_model(plan())
    assert len(loads) == 1 and ff.template_count() == 1


def test_naive_cueq_allowance_is_in_the_key(loads, cuda_rule):
    ff = ForceField(pristine())
    ff.execution_model(plan())
    allowed = plan()
    allowed.allowances = ("allow_naive_cueq",)
    ff.execution_model(allowed)
    assert len(loads) == 2
    assert loads[1]["allow_naive_cueq"] is True


def test_a_template_is_never_shared_between_force_fields(loads, cuda_rule):
    one, two = ForceField(pristine()), ForceField(pristine())
    one.execution_model(plan())
    two.execution_model(plan())
    assert len(loads) == 2


# --------------------------------------------------------------------------- bound, release, lifetime


def test_the_cache_is_bounded_least_recently_used_out(loads, cuda_rule):
    ff = ForceField(pristine())
    ff.execution_model(plan(dtype="float32"))
    ff.execution_model(plan(dtype="float64"))
    ff.execution_model(plan(dtype="float32"))  # hit: float32 is now the most recently used
    ff.execution_model(plan(backend="cueq"))  # a third key evicts float64
    assert ff.template_count() == templates.MAX_TEMPLATES == 2
    assert len(loads) == 3
    ff.execution_model(plan(dtype="float32"))
    assert len(loads) == 3
    ff.execution_model(plan(dtype="float64"))
    assert len(loads) == 4


def test_release_templates_empties_the_cache_and_reports_bytes(loads, cuda_rule):
    ff = ForceField(pristine())
    ff.execution_model(plan())
    expected = sum(t.numel() * t.element_size() for t in tensors(fake_loaded().model))
    assert ff.release_templates() == expected
    assert ff.template_count() == 0 and ff.release_templates() == 0
    ff.execution_model(plan())
    assert len(loads) == 2


def test_templates_die_with_their_force_field(loads, cuda_rule):
    ff = ForceField(pristine())
    ff.execution_model(plan())
    template = ff._templates.template(next(iter(ff._templates.keys())))
    alive = weakref.ref(template.model)
    del ff, template
    gc.collect()
    assert alive() is None


# --------------------------------------------------------------------------- the mutation guard


def test_a_template_changed_in_place_raises(loads, cuda_rule):
    ff = ForceField(pristine())
    ff.execution_model(plan())
    template = ff._templates.template(next(iter(ff._templates.keys())))
    with torch.no_grad():
        next(template.model.parameters()).add_(1.0)
    with pytest.raises(templates.TemplateChangedError, match="changed"):
        ff.execution_model(plan())


def test_a_template_tensor_replaced_raises(loads, cuda_rule):
    ff = ForceField(pristine())
    ff.execution_model(plan())
    template = ff._templates.template(next(iter(ff._templates.keys())))
    template.model[0].weight = torch.nn.Parameter(template.model[0].weight.detach().clone())
    with pytest.raises(templates.TemplateChangedError):
        ff.execution_model(plan())


# --------------------------------------------------------------------------- what reaches loading.load


def test_the_load_gets_the_same_options_as_before(loads, cuda_rule):
    # execution_model's call into loading.load is unchanged by the cache (decision 24's per-plan conversion).
    ff = ForceField(pristine())
    ff.execution_model(plan(backend="cueq", dtype="float64", fusion="on"))
    call = loads[0]
    assert call["backend"] == "cueq" and call["dtype"] == "float64" and call["device"] == "cuda:0"
    assert call["cueq_conv_fusion"] is True and call["sha256"] == "0" * 64 and call["allow_naive_cueq"] is False


def test_forcefield_module_keeps_no_cache_state():
    # No module-level state (AGENTS.md): the cache belongs to each ForceField.
    assert not any(isinstance(value, templates.TemplateCache) for value in vars(forcefield_module).values())
    assert not any(isinstance(value, templates.TemplateCache) for value in vars(templates).values())
