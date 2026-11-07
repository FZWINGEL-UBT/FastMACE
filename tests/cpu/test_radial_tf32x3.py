"""``RADIAL_TF32X3``: the folded radial last layer and its input gradient as 3xTF32 products (off-polar-alex,
6 November 2026; on the A100 for MACE-OFF's radial networks since 7 November 2026, decision 66).

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


def test_the_default_is_the_a100_and_never_on_the_cpu(monkeypatch):
    assert sl.RADIAL_TF32X3 is None and sl.RADIAL_TF32X3_GPUS == ("NVIDIA A100-SXM4-80GB",)
    assert "NVIDIA A100 80GB PCIe" in sl.RADIAL_TF32X3_GPUS  # the family
    assert not sl.radial_tf32x3_default(torch.device("cpu"), True)
    monkeypatch.setattr(sl, "RADIAL_TF32X3", True)
    assert not sl.radial_tf32x3_default("cpu", True)  # the constant does not reach the CPU


@pytest.mark.parametrize("name, eligible, expected", [
    ("NVIDIA A100-SXM4-80GB", True, True),
    ("NVIDIA A100-SXM4-80GB", False, False),  # not MACE-OFF: the GPU list does not reach it
    ("NVIDIA GeForce RTX 5060 Ti", True, False),
])
def test_the_gpu_list_applies_to_eligible_products_only(monkeypatch, name, eligible, expected):
    monkeypatch.setattr(sl, "_tf32x3_listed", lambda index: name in sl.RADIAL_TF32X3_GPUS)
    assert sl.radial_tf32x3_default(torch.device("cuda", 0), eligible) is expected
    monkeypatch.setattr(sl, "RADIAL_TF32X3", True)  # an A/B side's choice: every product
    assert sl.radial_tf32x3_default(torch.device("cuda", 0), False)
    monkeypatch.setattr(sl, "RADIAL_TF32X3", False)
    assert not sl.radial_tf32x3_default(torch.device("cuda", 0), True)


def test_folded_linear_passes_its_eligibility(monkeypatch):
    a, weight, _ = _inputs()
    seen = []
    monkeypatch.setattr(sl, "radial_tf32x3_default", lambda device, eligible=False: seen.append(eligible) or False)
    sl.folded_linear(a.float(), weight.float())
    sl.folded_linear(a.float(), weight.float(), tf32x3=True)
    assert seen == [False, True]


def test_folded_linear_routes_through_the_tf32x3_products(monkeypatch):
    a, weight, grad = _inputs()
    a, weight, grad = a.float(), weight.float(), grad.float()
    calls = []
    monkeypatch.setattr(sl, "radial_tf32x3_default", lambda device, eligible=False: True)
    real_forward = sl.tf32x3_forward
    monkeypatch.setattr(sl, "tf32x3_forward", lambda *args: calls.append("forward") or real_forward(*args))
    x = a.clone().requires_grad_()
    out = sl.folded_linear(x, weight)
    (got,) = torch.autograd.grad(out, x, grad)
    assert calls == ["forward"]
    torch.testing.assert_close(out, a @ weight.t())
    torch.testing.assert_close(got, grad @ weight)
    monkeypatch.setattr(sl, "radial_tf32x3_default", lambda device, eligible=False: False)
    calls.clear()
    sl.folded_linear(a, weight)
    assert calls == []
