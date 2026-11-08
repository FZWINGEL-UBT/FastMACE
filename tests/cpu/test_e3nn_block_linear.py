"""``E3NN_BLOCK_LINEAR``: e3nn ``o3.Linear`` as a ``BlockLinear`` (off-polar-alex, 6 November 2026).

MACE-POLAR's field blocks keep e3nn Linears (512x0e+512x1o -> 512x0e+512x1o) after the cuEquivariance conversion;
``block_linear`` takes them too with the A/B constant, so that ``tf32x3_gemms`` reaches their products. The
conversion reads each instruction's weights off the module and puts mul_ir layout transposes around the ir_mul
core: values and input gradients as e3nn's up to rounding.
"""

from __future__ import annotations

import copy

import pytest
import torch

pytest.importorskip("mace")  # mace first: it sets up e3nn's constants load (as test_fx_e3nn_linear does)
import mace.modules  # noqa: E402,F401
from e3nn import o3  # noqa: E402

from fastmace.rewrites import linear as lin  # noqa: E402
from fastmace.rewrites import precision  # noqa: E402
from fastmace.rewrites import registry as reg  # noqa: E402
from fastmace.rewrites.linear import modules as lm  # noqa: E402

IRREPS_IN = "24x0e+16x1o+8x2e+4x0e"
IRREPS_OUT = "32x0e+16x1o+8x2e"


def _linear(irreps_in=IRREPS_IN, irreps_out=IRREPS_OUT, **options):
    torch.manual_seed(7)
    return o3.Linear(o3.Irreps(irreps_in), o3.Irreps(irreps_out), **options).double()


def _values_and_gradients(module, x, grad):
    x = x.clone().requires_grad_()
    out = module(x)
    return out.detach(), torch.autograd.grad(out, x, grad)[0]


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(lm, "E3NN_MIN_WEIGHTS", 1)


def test_the_block_linear_matches_e3nn_values_and_input_gradients(small):
    linear = _linear()
    block = lm.e3nn_block_linear(linear)
    assert type(block).__name__ == "BlockLinear"  # precision.route finds it by its type name
    generator = torch.Generator().manual_seed(3)
    x = torch.randn((11, linear.irreps_in.dim), generator=generator, dtype=torch.float64)
    grad = torch.randn((11, linear.irreps_out.dim), generator=generator, dtype=torch.float64)
    expected, expected_grad = _values_and_gradients(linear, x, grad)
    got, got_grad = _values_and_gradients(block, x, grad)
    torch.testing.assert_close(got, expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(got_grad, expected_grad, rtol=0, atol=1e-12)
    # 24x0e and 4x0e both feed 32x0e: one block each, and 1o -> 1o, 2e -> 2e.
    assert sorted(block.block_ends) == [(0, 0), (1, 1), (2, 2), (3, 0)]


def test_layout_transposes_are_inverse(small):
    segments = [(0, 3, 1), (3, 4, 3), (15, 2, 5)]
    x = torch.arange(2 * 25, dtype=torch.float64).reshape(2, 25)
    there = lm.LayoutTranspose(segments, True)(x)
    assert torch.equal(lm.LayoutTranspose(segments, False)(there), x)
    # mul_ir (channel u, component m at 3 + u * 3 + m) -> ir_mul (3 + m * 4 + u)
    assert there[0, 3 + 1 * 4 + 2] == x[0, 3 + 2 * 3 + 1]


def test_what_it_does_not_take(small, monkeypatch):
    assert lm.e3nn_block_linear(_linear("8x0e", "8x0e", biases=True)) is None  # a bias
    assert lm.e3nn_block_linear(torch.nn.Linear(4, 4)) is None
    monkeypatch.setattr(lm, "E3NN_MIN_WEIGHTS", 1 << 14)
    assert lm.e3nn_block_linear(_linear()) is None  # 1408 weights: too small to gain


class _Holder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mix = _linear()
        self.other = torch.nn.Linear(3, 3).double()

    def forward(self, x):
        return self.mix(x)


@pytest.mark.parametrize("constant", [None, False, True])
def test_block_linears_takes_e3nn_linears_with_the_constant(small, monkeypatch, constant):
    monkeypatch.setattr(lin, "E3NN_BLOCK_LINEAR", constant)
    model = _Holder()
    original = copy.deepcopy(model)
    replaced = lin.block_linears(model)
    if not constant:  # None: E3NN_BLOCK_LINEAR_GPUS, and the CPU is never in it
        assert replaced == [] and type(model.mix).__name__ == "Linear"
        return
    assert replaced == ["mix"] and type(model.mix).__name__ == "BlockLinear"
    result = lin._block_linear_selfcheck(original, model, reg.Context(seed=1))
    assert result.passed, result.detail
    # tf32x3_gemms routes it like any BlockLinear; on the CPU its products stay in PyTorch.
    assert precision.route(model) == ["mix"] and model.mix.tf32x3
    x = torch.randn((5, original.mix.irreps_in.dim), dtype=torch.float64)
    torch.testing.assert_close(model(x), original(x), rtol=0, atol=1e-12)


def test_the_default_by_device(monkeypatch):
    monkeypatch.setattr(lin, "E3NN_BLOCK_LINEAR", None)
    assert lin.E3NN_BLOCK_LINEAR_GPUS == ()  # none measured yet (decision 66)
    assert not lin.e3nn_block_linear_default(torch.device("cpu"))
    assert not lin.e3nn_block_linear_default("NVIDIA A100-SXM4-80GB")
    monkeypatch.setattr(lin, "E3NN_BLOCK_LINEAR_GPUS", lin.GPUModels(("NVIDIA A100-SXM4-80GB",)))
    assert lin.e3nn_block_linear_default("NVIDIA A100 80GB PCIe")
    monkeypatch.setattr(lin, "E3NN_BLOCK_LINEAR", False)
    assert not lin.e3nn_block_linear_default("NVIDIA A100-SXM4-80GB")
