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
