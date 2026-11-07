"""The waves tile rule of the folded input gradient on CUDA: bitwise the fixed tiles (opt-2 OPT2-T03).

``folded_linear_input_grad(grad, weight, "waves")`` against ``"fixed"`` for the shapes of MH-1's folded last
layer (Q = 1280, P + 1 = 65) at edge counts below, inside and above the rule's 2-8 waves: equal bit for bit,
since BLOCK_K and so each row's reduction order is the same; and both within FP32 rounding of
``grad @ weight`` in float64. Fast tier (a few kernels).
"""

import pytest
import torch

from fastmace.kernels.triton import skinny_linear, tile_rule

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("rows", [2048, 4096, 8203, 16384, 56250])
def test_waves_tiles_are_bitwise_the_fixed_tiles(rows):
    generator = torch.Generator(device="cuda").manual_seed(rows)
    grad = torch.randn((rows, 1280), device="cuda", generator=generator)
    weight = torch.randn((1280, 65), device="cuda", generator=generator)
    fixed = skinny_linear.folded_linear_input_grad(grad, weight, "fixed")
    waves = skinny_linear.folded_linear_input_grad(grad, weight, "waves")
    sms = tile_rule.device_sms(grad.device)
    choice = skinny_linear.tile_choice(rows, sms, "waves")
    assert torch.equal(fixed, waves), (rows, sms, choice)
    reference = (grad.double() @ weight.double()).float()
    scale = reference.abs().max()
    assert float((waves - reference).abs().max() / scale) < 1e-5


def test_the_gradient_through_folded_linear_is_bitwise_the_same():
    generator = torch.Generator(device="cuda").manual_seed(1)
    a = torch.randn((8203, 65), device="cuda", generator=generator, requires_grad=True)
    weight = torch.randn((1280, 65), device="cuda", generator=generator)
    upstream = torch.randn((8203, 1280), device="cuda", generator=generator)
    grads = []
    for rule in ("fixed", "waves"):
        (grad,) = torch.autograd.grad((skinny_linear.folded_linear(a, weight, tile_rule=rule) * upstream).sum(), a)
        grads.append(grad)
    assert torch.equal(*grads)


@pytest.mark.parametrize(("rows", "q", "p"), [(512, 3808, 64), (300, 2176, 64), (100, 1280, 48), (577, 1280, 64)])
def test_split_columns_are_bitwise_the_one_axis_grid(monkeypatch, rows, q, p):
    """``SPLIT_COLUMNS`` (Codex R1): the column-split grid gives the one-axis grid's values bit for bit, here
    forced on whatever ``split_columns`` decides (577 rows: the rule keeps the one-axis grid)."""
    generator = torch.Generator(device="cuda").manual_seed(rows + q)
    grad = torch.randn((rows, q), device="cuda", generator=generator)
    weight = torch.randn((q, p + 1), device="cuda", generator=generator)
    monkeypatch.setattr(skinny_linear, "SPLIT_COLUMNS", False)
    plain = skinny_linear.folded_linear_input_grad(grad, weight, "fixed")
    monkeypatch.setattr(skinny_linear, "SPLIT_COLUMNS", True)
    monkeypatch.setattr(skinny_linear, "split_columns", lambda rows, p, sms, block_m: skinny_linear.SPLIT_BLOCK_P)
    split = skinny_linear.folded_linear_input_grad(grad, weight, "fixed")
    assert torch.equal(plain, split)
    out = torch.empty_like(split)
    skinny_linear.input_grad_into(grad, weight, weight[:, :p].contiguous(), weight[:, p].contiguous(), out)
    assert torch.equal(plain, out)
