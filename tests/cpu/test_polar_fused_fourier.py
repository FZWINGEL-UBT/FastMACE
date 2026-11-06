"""``polar.FUSED_FOURIER`` (A/B constant, off): POLAR's k-space sums without dense cosines and sines.

``fastmace.kernels.triton.fourier`` gives the structure factor ``(mask * cos(k r^T) @ x, ... sin ...)`` and the
projection ``cos(k r^T)^T @ (mask a) + sin(k r^T)^T @ (mask b)`` with their own backward (positions, k-vectors
and the thin factors). On the CPU its custom ops are dense PyTorch, so these tests check the backward formulas
against autograd through the dense matrices, and the enabled PolarMACE path (energy, forces, stress; a variable
cell whose fixed k-set masks k-vectors) against the unfused one. The Triton kernels:
tests/gpu/test_fourier_cuda.py.
"""

import copy

import pytest
import torch

pytest.importorskip("mace")

from _support.structures import nacl  # noqa: E402
from _support.tiny_models import polar_available, tiny_family_calculator  # noqa: E402
from fastmace.config import PolarConfig  # noqa: E402
from fastmace.kernels.triton import fourier  # noqa: E402
from fastmace.longrange import polar  # noqa: E402

torch.serialization.add_safe_globals([slice])

needs_polar = pytest.mark.skipif(not polar_available(), reason="PolarMACE needs graph_longrange")


def _inputs(K=23, N=17, M=5, seed=0):
    generator = torch.Generator().manual_seed(seed)

    def leaf(*shape, scale=1.0):
        return (scale * torch.randn(shape, generator=generator, dtype=torch.float64)).requires_grad_()

    mask = (torch.rand(K, generator=generator) > 0.25).to(torch.float64)
    assert 0 < mask.sum() < K
    return leaf(K, 3, scale=2.0), leaf(N, 3, scale=3.0), mask, leaf(N, M), leaf(K, M), leaf(K, M)


def _gradients(outputs, inputs, seed=1):
    generator = torch.Generator().manual_seed(seed)
    weights = [torch.randn(o.shape, generator=generator, dtype=o.dtype) for o in outputs]
    loss = sum((w * o).sum() for w, o in zip(weights, outputs))
    return torch.autograd.grad(loss, inputs)


def test_structure_factor_matches_the_dense_sums_and_their_gradients():
    k, r, mask, x, _, _ = _inputs()
    theta = k @ r.t()
    dense = (mask[:, None] * torch.cos(theta) @ x, mask[:, None] * torch.sin(theta) @ x)
    fused = fourier.structure_factor(k, r, mask, x)
    for a, b in zip(fused, dense):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    for a, b in zip(_gradients(fused, (k, r, x)), _gradients(dense, (k, r, x))):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)


def test_projection_matches_the_dense_sums_and_their_gradients():
    k, r, mask, _, a, b = _inputs()
    theta = k @ r.t()
    dense = torch.cos(theta).t() @ (mask[:, None] * a) + torch.sin(theta).t() @ (mask[:, None] * b)
    fused = fourier.fourier_project(k, r, mask, a, b)
    torch.testing.assert_close(fused, dense, rtol=1e-12, atol=1e-12)
    for got, expected in zip(_gradients((fused,), (k, r, a, b)), _gradients((dense,), (k, r, a, b))):
        torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-11)


def test_one_unused_output_of_the_structure_factor():
    k, r, mask, x, _, _ = _inputs(seed=4)
    yc, _ = fourier.structure_factor(k, r, mask, x)
    (got,) = torch.autograd.grad(yc.sum(), (r,))
    (expected,) = torch.autograd.grad((mask[:, None] * torch.cos(k @ r.t()) @ x).sum(), (r,))
    torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-11)


def test_the_mask_takes_one_graph_only(monkeypatch):
    k_batch = torch.tensor([0, 1, 0])
    positions = torch.zeros(4, 3)
    one = polar._Call(None, {"ptr": torch.tensor([0, 4])})
    two = polar._Call(None, {"ptr": torch.tensor([0, 2, 4])})
    assert polar._fourier_mask(k_batch, positions, one) is None  # the constant is off
    monkeypatch.setattr(polar, "FUSED_FOURIER", True)
    assert torch.equal(polar._fourier_mask(k_batch, positions, one), torch.tensor([1.0, 0.0, 1.0]))
    assert polar._fourier_mask(k_batch, positions, two) is None
    assert polar._fourier_mask(k_batch, positions, None) is None


def test_the_replaced_dynamic_forward_is_pinned():
    assert "graph_longrange.features:GTOElectrostaticFeatures._pbc_forward_dynamic" in polar.SOURCE_HASHES
    if polar_available():
        assert polar.source_problems() == []


def _evaluate(calc, model, atoms, stress):
    out = model(calc._atoms_to_batch(atoms).to_dict(), compute_stress=stress, training=False)
    return {key: out[key].detach() for key in ("energy", "forces", "stress") if out.get(key) is not None}


@needs_polar
@pytest.mark.parametrize("stress,variable_cell", [(False, False), (True, False), (True, True)])
def test_enabled_path_with_fused_fourier(monkeypatch, stress, variable_cell):
    calc = tiny_family_calculator("polar", [11, 17])
    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    model = copy.deepcopy(calc.models[0])
    polar.enable(model, atoms.cell.array, PolarConfig(), variable_cell=variable_cell)
    expected = _evaluate(calc, model, atoms, stress)
    masks, sums = [], []
    real_mask, real_sf = polar._fourier_mask, fourier.structure_factor

    def recording_mask(*args):
        mask = real_mask(*args)
        masks.append(mask)
        return mask

    monkeypatch.setattr(polar, "_fourier_mask", recording_mask)
    monkeypatch.setattr(fourier, "structure_factor", lambda *args: sums.append(1) or real_sf(*args))
    monkeypatch.setattr(polar, "FUSED_FOURIER", True)
    got = _evaluate(calc, model, atoms, stress)
    assert masks and all(m is not None for m in masks) and sums  # the fused sums ran
    if variable_cell:  # the fixed k-set's headroom: some k-vectors lie outside the cutoff and are masked
        assert any(bool((m == 0).any()) for m in masks)
    assert set(got) == set(expected)
    for key in got:
        torch.testing.assert_close(got[key], expected[key], rtol=1e-9, atol=1e-11)
