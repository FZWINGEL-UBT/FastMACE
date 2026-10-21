"""``split_k`` (plans/opt-2.md OPT2-T05; docs/design/opt2-T05-split-k.md): small-cell atom products split
along K by a shape rule from M, N, K and the SM count.

- The rule: the products of the design note's table, the one-wave threshold, the K limits, no SMs.
- ``split_k_mm`` equals ``a @ b`` to rounding in float64 (strided row views included), with its input
  gradient through the split.
- The record: opt-in, needs ``block_linear``, conflicts with ``grouped_launch``, applies to every block linear,
  self-checks; with a fake SM count that splits every product, the tiny MH-1-type model's energy and forces
  agree with the unsplit model to rounding in float64. Off, nothing is set.
"""

import copy
import importlib.util

import pytest
import torch

from fastmace.kernels.triton import tile_rule
from fastmace.rewrites import fusion

pytest.importorskip("mace")
pytestmark = pytest.mark.skipif(importlib.util.find_spec("cuequivariance_torch") is None,
                                reason="cuequivariance_torch is not installed")

from test_rewrites_fusion import _requests_segment, registry  # noqa: E402
from test_rewrites_mh1 import context, energy_and_forces, tiny  # noqa: E402

RTX_5060_TI, RTX_4000_ADA = 36, 48


# --------------------------------------------------------------------------- the rule


@pytest.mark.parametrize("sms", [RTX_5060_TI, RTX_4000_ADA])
def test_the_rule_splits_only_products_below_one_wave(sms):
    parts = tile_rule.split_k_parts
    assert parts(216, 512, 512, sms) == 4  # NaCl l = 0, forward and input gradient
    assert parts(216, 512, 2048, sms) == 4  # NaCl grouped l = 0 input, forward
    assert parts(216, 2048, 512, sms) == 1  # its input gradient: 128 tiles
    for m in (648, 1080, 1512):  # NaCl l = 1-3
        assert parts(m, 512, 512, sms) == 1
    for m in (1944, 5832, 9720):  # the large cells
        assert parts(m, 512, 512, sms) == 1 and parts(m, 512, 2048, sms) == 1


def test_the_rule_respects_k():
    parts = tile_rule.split_k_parts
    assert parts(64, 64, 512, 36) == 4  # one tile: 4 x 128 is the smallest part allowed
    assert parts(64, 64, 256, 36) == 2
    assert parts(64, 64, 128, 36) == 1  # no part below SPLIT_K_MIN_PART
    assert parts(64, 64, 384, 36) == 2  # 2 parts of 192; 4 parts of 96 are too small
    assert parts(64, 64, 768, 36) == 4  # 4 parts of 192
    assert parts(64, 64, 513, 36) == 1  # K must divide into equal parts
    assert parts(64, 64, 300, 36) == 2  # 2 parts of 150; 4 parts of 75 are too small
    assert parts(216, 512, 512, None) == 1 and parts(216, 512, 512, 0) == 1  # no SMs: not CUDA


def test_the_rule_is_integer_arithmetic():
    assert isinstance(tile_rule.split_k_parts(216, 512, 512, 36), int)


# --------------------------------------------------------------------------- the product


@pytest.mark.parametrize("parts", [1, 2, 4])
def test_split_k_mm_equals_the_product_to_rounding(parts):
    from fastmace.rewrites.linear.modules import split_k_mm

    generator = torch.Generator().manual_seed(5)
    wide = torch.randn((30, 3 * 512), generator=generator, dtype=torch.float64)
    a = wide[:, 512:1024].requires_grad_()  # a strided row view, as a block linear's l = 0 rows
    b = torch.randn((512, 96), generator=generator, dtype=torch.float64)
    got, want = split_k_mm(a, b, parts), a @ b
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    grad = torch.randn((30, 96), generator=generator, dtype=torch.float64)
    (got_grad,) = torch.autograd.grad((got * grad).sum(), a)
    (want_grad,) = torch.autograd.grad((want * grad).sum(), a)
    torch.testing.assert_close(got_grad, want_grad, rtol=1e-12, atol=1e-12)


def test_split_k_matmul_splits_forward_and_input_gradient_by_their_own_parts():
    from fastmace.rewrites.linear.modules import SplitKMatmul

    generator = torch.Generator().manual_seed(6)
    x = torch.randn((8, 256), generator=generator, dtype=torch.float64).requires_grad_()
    w = torch.randn((256, 512), generator=generator, dtype=torch.float64)
    out = SplitKMatmul.apply(x, w, w.t().contiguous(), 2, 4)
    torch.testing.assert_close(out, x @ w, rtol=1e-12, atol=1e-12)
    grad = torch.randn_like(out)
    (dx,) = torch.autograd.grad((out * grad).sum(), x)
    torch.testing.assert_close(dx, grad @ w.t(), rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- the record


def _rewritten(split, grouped="off"):
    model, batch = tiny("mh1", backend="cueq")
    regs = registry()
    out = copy.deepcopy(model)
    asked = _requests_segment("on") | {"split_k": split, "grouped_launch": grouped}
    resolution = regs.resolve(out, asked, strict=False)
    applied = regs.apply(out, resolution, context(batch), original=model)
    return model, out, batch, resolution, applied


def _blocks(model):
    from fastmace.rewrites.linear.modules import BlockLinear, ResidualBlockLinear

    return [m for m in model.modules() if isinstance(m, (BlockLinear, ResidualBlockLinear))]


def test_the_record_is_opt_in_rounding_and_deterministic():
    record = fusion.RECORDS["split_k"]
    assert record.default == "off" and record.exactness == "rounding" and record.deterministic is True
    assert record.requires == ("block_linear",) and record.conflicts == ("grouped_launch",)
    from fastmace.planner import fixed

    assert fixed.AUTO_RULES["RewriteConfig.split_k"] == "never_automatic"


def test_the_record_applies_to_every_block_linear_and_selfchecks():
    _, model, _, resolution, applied = _rewritten("on")
    assert resolution.decision("split_k").enabled
    entry = next(entry for entry in applied if entry.name == "split_k")
    assert entry.result.passed
    blocks = _blocks(model)
    assert blocks and all(m.split_k and m.sms is None for m in blocks)  # CPU: no SMs, so cuBLAS everywhere


def test_off_sets_nothing():
    _, model, _, resolution, _ = _rewritten("off")
    assert not resolution.decision("split_k").enabled
    assert all(not getattr(m, "split_k", False) for m in _blocks(model))


def test_the_record_needs_block_linear_and_excludes_grouped_launch():
    model, _ = tiny("mh1", backend="cueq")
    asked = {name: "off" for name in registry().names()} | {"split_k": "on"}
    assert not registry().resolve(copy.deepcopy(model), asked, strict=False).decision("split_k").enabled
    _, _, _, resolution, _ = _rewritten("on", grouped="on")
    assert not (resolution.decision("split_k").enabled and resolution.decision("grouped_launch").enabled)


def test_split_products_keep_energy_and_forces_to_rounding(monkeypatch):
    """With a fake device of a million SMs (so no product fills a wave) and parts of any size, every product
    whose K divides splits (the CPU's bmm): energy and forces agree with the unsplit model to rounding in
    float64."""
    from fastmace.rewrites.linear import modules

    _, model, batch, _, _ = _rewritten("on")
    _, without, _, _, _ = _rewritten("off")
    energy, forces = energy_and_forces(without, batch)
    calls = []
    real = modules.split_k_mm
    monkeypatch.setattr(modules, "split_k_mm", lambda a, b, parts: calls.append(parts) or real(a, b, parts))
    monkeypatch.setattr(tile_rule, "SPLIT_K_MIN_PART", 1)
    for m in _blocks(model):
        m.sms = 10 ** 6
    new_energy, new_forces = energy_and_forces(model, batch)
    assert any(parts > 1 for parts in calls), "no product was split"
    torch.testing.assert_close(new_energy, energy, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(new_forces, forces, rtol=1e-10, atol=1e-12)
