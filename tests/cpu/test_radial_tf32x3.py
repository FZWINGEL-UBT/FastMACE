"""``RADIAL_TF32X3``: the folded radial last layer and its input gradient as 3xTF32 products (off-polar-alex,
6 November 2026; an A/B constant, off everywhere until measured).

On the CPU ``tf32x3_mm`` is PyTorch's product, so these tests check the padding and the routing: the padded
products equal the unpadded ones, and ``folded_linear`` takes the 3xTF32 path only where the default says so.
"""

from __future__ import annotations

import pytest
import torch

from fastmace.kernels.triton import skinny_linear as sl


def _inputs(rows=37, p=65, q=44, seed=0):
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn((rows, p), generator=generator, dtype=torch.float64)
    weight = torch.randn((q, p), generator=generator, dtype=torch.float64)
    grad = torch.randn((rows, q), generator=generator, dtype=torch.float64)
    return a, weight, grad


@pytest.mark.parametrize("p", [64, 65, 66])
def test_padded_products_equal_the_plain_ones(p):
    a, weight, grad = _inputs(p=p)
    torch.testing.assert_close(sl.tf32x3_forward(a, weight), a @ weight.t(), rtol=0, atol=1e-12)
    got = sl.folded_linear_input_grad_tf32x3(grad, weight)
    assert got.shape == (a.shape[0], p) and got.is_contiguous()
    torch.testing.assert_close(got, grad @ weight, rtol=0, atol=1e-12)


def test_the_default_is_off_and_never_on_the_cpu(monkeypatch):
    assert sl.RADIAL_TF32X3 is None and sl.RADIAL_TF32X3_GPUS == ()
    assert not sl.radial_tf32x3_default(torch.device("cpu"))
    monkeypatch.setattr(sl, "RADIAL_TF32X3", True)
    assert not sl.radial_tf32x3_default("cpu")  # the constant does not reach the CPU


def test_folded_linear_routes_through_the_tf32x3_products(monkeypatch):
    a, weight, grad = _inputs()
    a, weight, grad = a.float(), weight.float(), grad.float()
    calls = []
    monkeypatch.setattr(sl, "radial_tf32x3_default", lambda device: True)
    real_forward = sl.tf32x3_forward
    monkeypatch.setattr(sl, "tf32x3_forward", lambda *args: calls.append("forward") or real_forward(*args))
    x = a.clone().requires_grad_()
    out = sl.folded_linear(x, weight)
    (got,) = torch.autograd.grad(out, x, grad)
    assert calls == ["forward"]
    torch.testing.assert_close(out, a @ weight.t())
    torch.testing.assert_close(got, grad @ weight)
    monkeypatch.setattr(sl, "radial_tf32x3_default", lambda device: False)
    calls.clear()
    sl.folded_linear(a, weight)
    assert calls == []
