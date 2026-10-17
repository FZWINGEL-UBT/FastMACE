"""The linear family: block_linear, grouped_linear, merged_residual (SPEC §6.2; plan task 1a-T06).

The module tests are ported from the research engine's ``tests/core/test_block_linear.py``
(ResearchModule tag research-engine-baseline-20261017) with their irreps and tolerances; the
in-place variant (``INPLACE_BLOCK_LINEAR``, opt-in and slower) is not ported. The tiny-model
tests convert each family of ``tests/_support/tiny_models.py`` to cuEquivariance on the CPU,
rewrite it through the registry and compare energy and forces with the unrewritten cuEq model
at the SPEC §6.4 threshold. The baseline tests run the research engine's own
``_block_linear`` and ``_ResidualBlockLinear`` from the export named by ``FASTMACE_BASELINE``
and skip when it is unset: same operations in the same order, so they agree bitwise.
"""

import copy
import importlib
import os
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import torch
from packaging.specifiers import SpecifierSet

from fastmace.rewrites import linear
from fastmace.rewrites import registry as reg
from fastmace.rewrites.linear import modules as lm

cue = pytest.importorskip("cuequivariance")
cuet = pytest.importorskip("cuequivariance_torch")

Z_VALUES = [11, 17]
FAMILIES = ("mh1", "omol", "residual", "density", "off")


def cueq_linear(irreps_in, irreps_out, layout, dtype=torch.float64):
    torch.manual_seed(2)
    return cuet.Linear(cue.Irreps("O3", irreps_in), cue.Irreps("O3", irreps_out), layout=layout,
                       shared_weights=True, internal_weights=True, dtype=dtype, method="naive")


def values_and_input_gradients(module, *inputs, grad):
    out = module(*inputs)
    return (out.detach(), *torch.autograd.grad(out, inputs, grad))


# --------------------------------------------------------------------------- ported module tests

LAYOUT_CASES = (("8x0e+4x1o+3x2e", "5x0e+6x1o+2x2e"),
                ("4x0e+2x1o+3x0e", "3x1o+5x0e"),
                ("6x1o", "2x1o+3x2e"))


@pytest.mark.parametrize("grouped", [False, True])
def test_values_and_input_gradients_match_for_both_layouts(grouped):
    for layout in (cue.ir_mul, cue.mul_ir):
        for irreps_in, irreps_out in LAYOUT_CASES:
            original = cueq_linear(irreps_in, irreps_out, layout)
            replacement = lm.block_linear(original, grouped=grouped)
            assert isinstance(replacement, lm.BlockLinear) and replacement.grouped is grouped
            wide = torch.randn((11, original.irreps_in.dim + 5), dtype=torch.float64)
            x = wide[:, 2:2 + original.irreps_in.dim].requires_grad_()  # strided rows
            grad = torch.randn((11, original.irreps_out.dim), dtype=torch.float64)
            new = values_and_input_gradients(replacement, x, grad=grad)
            old = values_and_input_gradients(original, x, grad=grad)
            for a, b in zip(new, old):
                torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_grouped_puts_same_irrep_inputs_of_one_output_in_one_product():
    original = cueq_linear("2x1o+3x1o+4x0e+1x1o", "5x1o+2x0e", cue.ir_mul)
    replacement = lm.block_linear(original, grouped=True)
    assert sorted(len(i) if isinstance(i, tuple) else 1 for i, _ in replacement.block_ends) == [1, 3]


def test_regrouping_equals_building_grouped():
    original = cueq_linear("2x1o+3x1o+4x0e+1x1o", "5x1o+2x0e", cue.ir_mul)
    built = lm.block_linear(original, grouped=True)
    regrouped = lm.regroup(lm.block_linear(original))
    assert regrouped.grouped and regrouped.block_ends == built.block_ends
    for (name, a), (other, b) in zip(built.named_buffers(), regrouped.named_buffers()):
        assert name == other and torch.equal(a, b)
    assert lm.regroup(built) is built


def test_merged_residual_matches_the_sum_of_both_layers():
    for grouped in (False, True):
        first = lm.block_linear(cueq_linear("4x0e+2x1o+3x0e+2x1o", "5x0e+3x1o", cue.ir_mul), grouped=grouped)
        residual = lm.block_linear(cueq_linear("2x0e+1x1o", "5x0e+3x1o", cue.ir_mul), grouped=grouped)
        merged = lm.ResidualBlockLinear(first, residual)
        assert len(merged.block_ends) == 2  # one product per output segment
        x = torch.randn((9, 19), dtype=torch.float64, requires_grad=True)
        y = torch.randn((9, 5), dtype=torch.float64, requires_grad=True)
        grad = torch.randn((9, 14), dtype=torch.float64)
        out = first(x) + residual(y)
        old = (out.detach(), *torch.autograd.grad(out, (x, y), grad))
        new = values_and_input_gradients(merged, x, y, grad=grad)
        for a, b in zip(new, old):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_merged_residual_refuses_different_outputs():
    first = lm.block_linear(cueq_linear("4x0e", "5x0e", cue.ir_mul))
    residual = lm.block_linear(cueq_linear("4x0e", "3x0e", cue.ir_mul))
    with pytest.raises(ValueError, match="different outputs"):
        lm.ResidualBlockLinear(first, residual)


def test_block_linears_replaces_nested_layers_and_skips_others():
    model = torch.nn.Module()
    model.inner = torch.nn.Module()
    model.inner.linear = cueq_linear("4x0e+2x1o", "3x0e+2x1o", cue.ir_mul)
    model.plain = torch.nn.Linear(3, 3)
    assert linear.block_linears(model) == ["inner.linear"]
    assert isinstance(model.inner.linear, lm.BlockLinear)
    assert isinstance(model.plain, torch.nn.Linear)
    assert linear.block_linears(model) == []  # nothing left to replace


def test_block_linear_skips_external_weights():
    external = cuet.Linear(cue.Irreps("O3", "4x0e"), cue.Irreps("O3", "3x0e"), layout=cue.ir_mul,
                           shared_weights=True, internal_weights=False, dtype=torch.float64, method="naive")
    assert lm.block_linear(external) is None


# --------------------------------------------------------------------------- records and self-checks


def wrapped(*linears):
    model = torch.nn.Module()
    for index, module in enumerate(linears):
        setattr(model, f"l{index}", module)
    return model


def test_records_have_the_spec_fields():
    assert set(linear.RECORDS) == {"block_linear", "grouped_linear", "merged_residual"}
    for name, rewrite in linear.RECORDS.items():
        assert isinstance(rewrite, reg.Rewrite) and rewrite.name == name
        assert rewrite.exactness == "rounding"  # SPEC §6.2
        assert rewrite.mace_versions == SpecifierSet(">=0.3.16,<0.3.17")
    assert linear.RECORDS["grouped_linear"].requires == ("block_linear",)
    assert linear.RECORDS["merged_residual"].requires == ("block_linear", "density_first")


def test_block_linear_applies_to_cueq_models_only():
    rewrite = linear.RECORDS["block_linear"]
    assert rewrite.applies_to(wrapped(cueq_linear("4x0e", "3x0e", cue.ir_mul))) == (True, "ok")
    supported, reason = rewrite.applies_to(wrapped(torch.nn.Linear(3, 3)))
    assert not supported and "cuEquivariance Linear" in reason


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_block_linear_selfcheck_passes_at_the_spec_threshold(dtype):
    original = wrapped(cueq_linear("8x0e+4x1o+3x2e", "5x0e+6x1o+2x2e", cue.ir_mul, dtype),
                       cueq_linear("6x1o", "2x1o+3x2e", cue.mul_ir, dtype))
    rewritten = copy.deepcopy(original)
    linear.RECORDS["block_linear"].apply(rewritten, reg.Context())
    result = reg.Registry([linear.RECORDS["block_linear"]]).selfcheck("block_linear", original, rewritten,
                                                                     reg.Context())
    assert result.passed and result.modules == 2
    assert result.threshold == reg.THRESHOLDS[str(dtype).removeprefix("torch.")]


def test_block_linear_selfcheck_catches_a_wrong_weight():
    original = wrapped(cueq_linear("8x0e+4x1o", "5x0e+6x1o", cue.ir_mul))
    rewritten = copy.deepcopy(original)
    linear.block_linears(rewritten)
    with torch.no_grad():
        rewritten.l0.weight_0.mul_(1 + 1e-8)
    result = linear.RECORDS["block_linear"].selfcheck(original, rewritten, reg.Context())
    assert not result.passed and "l0" in result.detail


def test_grouped_linear_selfcheck():
    original = wrapped(cueq_linear("2x1o+3x1o+4x0e+1x1o", "5x1o+2x0e", cue.ir_mul))
    rewritten = copy.deepcopy(original)
    linear.block_linears(rewritten)
    assert linear.group_linears(rewritten) == ["l0"]
    assert rewritten.l0.grouped
    result = linear.RECORDS["grouped_linear"].selfcheck(original, rewritten, reg.Context())
    assert result.passed and result.modules == 1


# --------------------------------------------------------------------------- tiny models


def tiny_cueq(family):
    from mace.cli.convert_e3nn_cueq import run

    from _support.structures import nacl
    from _support.tiny_models import FAMILIES as TINY
    from _support.tiny_models import tiny_mace_calculator

    calc = tiny_mace_calculator(Z_VALUES, **TINY[family])
    batch = calc._atoms_to_batch(nacl(repeat=1, rattle=0.05, seed=1))
    model = run(calc.models[0], device="cpu", return_model=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, batch


def energy_and_forces(model, batch):
    out = model(batch.clone().to_dict(), training=False, compute_force=True)
    return out["energy"].detach(), out["forces"].detach()


def relative(got, expected):
    return float((got - expected).abs().max()) / max(float(expected.abs().max()), 1e-300)


@pytest.mark.parametrize("family", FAMILIES)
def test_rewritten_tiny_models_agree_with_the_originals(family):
    model, batch = tiny_cueq(family)
    registry = reg.default_registry()
    rewritten = copy.deepcopy(model)
    resolution = registry.resolve(rewritten, {"block_linear": "on", "grouped_linear": "on"}, strict=False)
    assert resolution.enabled[:2] == ("block_linear", "grouped_linear")
    merged = resolution.decision("merged_residual")
    assert not merged.enabled
    assert ("requires density_first" if family in ("mh1", "omol") else "linear_res") in merged.reason
    applied = registry.apply(rewritten, resolution, original=model)
    assert [entry.name for entry in applied] == ["block_linear", "grouped_linear"]
    assert all(entry.result.passed and entry.result.max_relative_error <= 1e-10 for entry in applied)
    assert sum(isinstance(m, lm.BlockLinear) for m in rewritten.modules()) == applied[0].result.modules > 0
    energy_and_forces(model, batch)  # throwaway evaluation (docs/development.md)
    energy, forces = energy_and_forces(model, batch)
    new_energy, new_forces = energy_and_forces(rewritten, batch)
    assert relative(new_energy, energy) <= 1e-10 and relative(new_forces, forces) <= 1e-10


def _marked_forward(self, *args, **kwargs):
    return type(self).forward(self, *args, **kwargs)


_marked_forward.fastmace_residual_linear = True  # what 1a-T07's folded forward declares


def test_merged_residual_on_a_tiny_mh1_model():
    model, _ = tiny_cueq("mh1")
    rewritten = copy.deepcopy(model)
    linear.block_linears(rewritten)
    assert linear.merge_residual_linears(rewritten) == []  # no forward reads residual_linear yet
    for interaction in rewritten.interactions:
        interaction.density_before_linear = True
        interaction.forward = types.MethodType(_marked_forward, interaction)
    assert linear.merge_residual_linears(rewritten) == ["interactions.0", "interactions.1"]
    result = reg.Registry([linear.RECORDS["merged_residual"]]).selfcheck("merged_residual", model, rewritten,
                                                                        reg.Context())
    assert result.passed and result.modules == 2 and result.max_relative_error <= 1e-10
    with torch.no_grad():
        rewritten.interactions[1].residual_linear.weight_0.mul_(1 + 1e-8)
    assert not linear.RECORDS["merged_residual"].selfcheck(model, rewritten, reg.Context()).passed


def test_merged_residual_applies_to_mh1_interactions():
    rewrite = linear.RECORDS["merged_residual"]
    assert rewrite.applies_to(tiny_cueq("mh1")[0]).supported
    supported, reason = rewrite.applies_to(tiny_cueq("residual")[0])
    assert not supported and "RealAgnosticResidualNonLinearInteractionBlock" in reason


# --------------------------------------------------------------------------- baseline (bitwise)


def _baseline(name):
    root = os.environ.get("FASTMACE_BASELINE")
    if not root:
        pytest.skip("FASTMACE_BASELINE is not set (export of research-engine-baseline-20261017)")
    directory = Path(root).expanduser() / "Experiments" / "Langevin_Thermostat" / "langevin"
    assert directory.is_dir(), f"FASTMACE_BASELINE={root} has no Experiments/Langevin_Thermostat/langevin"
    if "langevin" not in sys.modules:
        package = types.ModuleType("langevin")
        package.__path__ = [str(directory)]
        sys.modules["langevin"] = package
    return importlib.import_module(f"langevin.{name}")


def assert_same_buffers(new, old):
    old_buffers = dict(old.named_buffers())
    assert set(dict(new.named_buffers())) == set(old_buffers)
    for name, tensor in new.named_buffers():
        assert tensor.dtype == old_buffers[name].dtype and torch.equal(tensor, old_buffers[name]), name


@pytest.mark.parametrize("grouped", [False, True])
def test_block_linear_equals_the_baseline_bitwise(grouped):
    gl = _baseline("gpu_langevin")
    cases = (*LAYOUT_CASES, ("2x1o+3x1o+4x0e+1x1o", "5x1o+2x0e"))
    for dtype in (torch.float64, torch.float32):
        for layout in (cue.ir_mul, cue.mul_ir):
            for irreps_in, irreps_out in cases:
                original = cueq_linear(irreps_in, irreps_out, layout, dtype)
                with mock.patch.object(gl, "GROUPED_BLOCK_LINEAR", grouped):
                    old = gl._block_linear(original)
                new = lm.block_linear(original, grouped=grouped)
                assert new.block_ends == old.block_ends and new.segments_in == old.segments_in
                assert_same_buffers(new, old)
                x = torch.randn((13, original.irreps_in.dim), dtype=dtype, requires_grad=True)
                grad = torch.randn((13, original.irreps_out.dim), dtype=dtype)
                for a, b in zip(values_and_input_gradients(new, x, grad=grad),
                                values_and_input_gradients(old, x, grad=grad)):
                    assert torch.equal(a, b)


@pytest.mark.parametrize("grouped", [False, True])
def test_merged_residual_equals_the_baseline_bitwise(grouped):
    gl = _baseline("gpu_langevin")
    first_cueq = cueq_linear("4x0e+2x1o+3x0e+2x1o", "5x0e+3x1o", cue.ir_mul)
    residual_cueq = cueq_linear("2x0e+1x1o", "5x0e+3x1o", cue.ir_mul)
    with mock.patch.object(gl, "GROUPED_BLOCK_LINEAR", grouped):
        old = gl._ResidualBlockLinear(gl._block_linear(first_cueq), gl._block_linear(residual_cueq))
    new = lm.ResidualBlockLinear(lm.block_linear(first_cueq, grouped=grouped),
                                 lm.block_linear(residual_cueq, grouped=grouped))
    assert new.block_ends == old.block_ends
    assert_same_buffers(new, old)
    x = torch.randn((9, 19), dtype=torch.float64, requires_grad=True)
    y = torch.randn((9, 5), dtype=torch.float64, requires_grad=True)
    grad = torch.randn((9, 14), dtype=torch.float64)
    for a, b in zip(values_and_input_gradients(new, x, y, grad=grad),
                    values_and_input_gradients(old, x, y, grad=grad)):
        assert torch.equal(a, b)


@pytest.mark.parametrize("family", ["mh1", "density"])
def test_tiny_model_rewrite_equals_the_baseline_bitwise(family):
    gl = _baseline("gpu_langevin")
    model, batch = tiny_cueq(family)
    ours, theirs = copy.deepcopy(model), copy.deepcopy(model)
    registry = reg.default_registry()
    registry.apply(ours, registry.resolve(ours, {"block_linear": "on", "grouped_linear": "on"}, strict=False),
                   original=model)
    with mock.patch.object(gl, "GROUPED_BLOCK_LINEAR", True):
        assert gl.block_linears(theirs) == sum(isinstance(m, lm.BlockLinear) for m in ours.modules())
    energy_and_forces(ours, batch)  # throwaway evaluation (docs/development.md)
    for a, b in zip(energy_and_forces(ours, batch), energy_and_forces(theirs, batch)):
        assert torch.equal(a, b)
