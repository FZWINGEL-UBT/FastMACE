"""The fused radial hidden stack (opt-2 OPT2-T02): which networks qualify, and the CPU path.

On the CPU (and for float64) ``RadialStack`` runs the original modules, so its output and gradients are
bitwise the original's. The Triton kernels are tested in ``tests/gpu/test_radial_mlp_cuda.py``.
"""

import copy

import pytest
import torch

from fastmace.kernels.triton import radial_mlp


def stack(d=64, k=2, *, affine=True, bias=True, activation=torch.nn.SiLU, seed=0):
    torch.manual_seed(seed)
    modules = [torch.nn.LayerNorm(d, elementwise_affine=affine), activation()]
    for _ in range(k):
        modules += [torch.nn.Linear(d, d, bias=bias), torch.nn.LayerNorm(d, elementwise_affine=affine), activation()]
    for module in modules:
        if isinstance(module, torch.nn.LayerNorm) and affine:
            with torch.no_grad():
                module.weight.uniform_(0.5, 1.5)
                module.bias.uniform_(-0.2, 0.2)
    return modules


@pytest.mark.parametrize(("d", "k"), [(64, 2), (64, 1), (32, 4), (128, 3), (16, 1)])
def test_mace_hidden_stacks_qualify(d, k):
    layers = radial_mlp.stack_layers(stack(d, k))
    assert not isinstance(layers, str)
    norms, linears = layers
    assert len(norms) == k + 1 and len(linears) == k


@pytest.mark.parametrize(("modules", "reason"), [
    (stack(64, 2)[:-1], "modules"),  # ends on a LayerNorm
    (stack(64, 0), "modules"),  # no hidden Linear (MH-1's density net is longer)
    (stack(64, 5), "at most"),
    (stack(48, 2), "width"),
    (stack(64, 2, affine=False), "affine"),
    (stack(64, 2, bias=False), "bias"),
    (stack(64, 2, activation=torch.nn.Tanh), "SiLU"),
])
def test_other_layouts_are_refused_with_a_reason(modules, reason):
    why = radial_mlp.stack_layers(modules)
    assert isinstance(why, str) and reason in why
    with pytest.raises(ValueError, match="not fusable"):
        radial_mlp.RadialStack(modules)


def test_a_wider_linear_is_refused():
    modules = stack(64, 2)
    modules[2] = torch.nn.Linear(64, 32)
    assert "64 x 64" in radial_mlp.stack_layers(modules)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_cpu_path_is_the_original_modules_bitwise(dtype):
    modules = [m.to(dtype) for m in stack(64, 2)]
    reference = torch.nn.Sequential(*copy.deepcopy(modules))
    fused = radial_mlp.RadialStack(modules)
    x = torch.randn(37, 64, dtype=dtype, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    out, expected = fused(x), reference(y)
    assert torch.equal(out, expected)
    grad = torch.randn_like(out)
    (gx,) = torch.autograd.grad(out, x, grad)
    (gy,) = torch.autograd.grad(expected, y, grad)
    assert torch.equal(gx, gy)


def test_the_stack_keeps_the_original_parameters():
    modules = stack(64, 2)
    fused = radial_mlp.RadialStack(modules)
    assert fused.norms[0] is modules[0] and fused.linears[1] is modules[5]
    keys = set(fused.state_dict())
    assert keys == {f"modules_.{key}" for key in torch.nn.Sequential(*modules).state_dict()}
    copied = copy.deepcopy(fused)
    assert copied.norms[0] is copied.modules_[0] and copied.linears[0] is copied.modules_[2]
    gamma, beta, weight, bias, eps = fused.operands()
    assert gamma.shape == (3, 64) and weight.shape == (2, 64, 64) and bias.shape == (2, 64)
    assert eps.tolist() == [pytest.approx(1e-5)] * 3


@pytest.mark.parametrize(("d", "expected"), [(16, (32, 2, True)), (32, (32, 4, True)), (64, (32, 8, True)),
                                             (128, (16, 8, True))])
def test_kernel_config_is_a_rule_from_the_width(d, expected):
    # 8 elements of a [rows, D] tile per thread: the budget at which the OPT2-T02 screen found no spills.
    config = radial_mlp.kernel_config(d)
    assert config == expected
    rows, warps, _ = config
    assert rows * d == radial_mlp.ELEMENTS_PER_THREAD * 32 * warps


def test_the_default_config_is_the_rule():
    fused = radial_mlp.RadialStack(stack(64, 2))
    assert fused.config is None  # resolved per call from the input width
    assert radial_mlp.RadialStack(stack(64, 2), config=(16, 4, False)).config == (16, 4, False)
