"""``polar.MEGAKERNEL`` (opt-in, ``MEGAKERNEL_GPUS`` empty): one POLAR field-feature call as one k-space kernel.

On the CPU ``fastmace.kernels.triton.kspace_megakernel``'s custom ops are its dense formulas, so these tests check
the hand-written backward and the per-k glue (density scale, Coulomb kernel, projection factor) against autograd
through graph_longrange's own functions in float64, for one and two channels (the spins). The Triton kernel:
tests/gpu/test_kspace_megakernel_cuda.py.
"""

import math

import pytest
import torch

from _support.tiny_models import polar_available
from fastmace.kernels.triton import kspace_megakernel as mk
from fastmace.longrange import polar

torch.serialization.add_safe_globals([slice])  # e3nn's constants, as in test_polar_fused_fourier.py

needs_polar = pytest.mark.skipif(not polar_available(), reason="PolarMACE needs graph_longrange")


def _cache(K=29, N=13, J=4, sigma=2, m=4, seed=0):
    generator = torch.Generator().manual_seed(seed)

    def leaf(*shape, scale=1.0, shift=0.0):
        return (shift + scale * torch.rand(shape, generator=generator, dtype=torch.float64)).requires_grad_()

    mask = (torch.rand(K, generator=generator) > 0.25).to(torch.float64)
    assert 0 < mask.sum() < K
    return {"k_vectors": (2.0 * torch.randn((K, 3), generator=generator, dtype=torch.float64)).requires_grad_(),
            "node_positions": leaf(N, 3, scale=8.0), "fourier_mask": mask,
            "density_basis_fs": leaf(K, 1, J, 2, shift=-0.5), "feature_basis_fs": leaf(K, sigma, m, 2, shift=-0.5),
            "volume_per_k": leaf(K, scale=0.2, shift=100.0), "k_factor_coulomb": leaf(K, scale=0.5, shift=0.1),
            "k_factor_proj": leaf(K, scale=0.5, shift=0.5)}


def _reference(cache, source):
    """graph_longrange's multichannel density, Coulomb kernel and projection on masked dense cosines and sines."""
    from graph_longrange.features import (
        apply_coulomb_kernel_batch,
        assemble_fourier_series_batch_channels,
        project_to_features_batch_channels,
    )

    theta = cache["k_vectors"] @ cache["node_positions"].t()
    mask = cache["fourier_mask"][:, None]
    cos, sin = mask * torch.cos(theta), mask * torch.sin(theta)
    density = assemble_fourier_series_batch_channels(source_feats=source, cosines=cos, sines=sin,
                                                     density_basis_fs=cache["density_basis_fs"],
                                                     volume_per_k=cache["volume_per_k"])
    potential = apply_coulomb_kernel_batch(k_norm2=None, density=density, k_factor_coulomb=cache["k_factor_coulomb"])
    features = project_to_features_batch_channels(potential=potential, feature_basis_fs=cache["feature_basis_fs"],
                                                  cosines=cos, sines=sin, k_factor_proj=cache["k_factor_proj"])
    return features.reshape(source.shape[0], source.shape[1], -1)


def _leaves(cache, source):
    return [cache[name] for name in ("k_vectors", "node_positions", "density_basis_fs", "feature_basis_fs",
                                     "volume_per_k", "k_factor_coulomb", "k_factor_proj")] + [source]


@needs_polar
@pytest.mark.parametrize("channels", [1, 2])
def test_megakernel_features_match_graph_longrange_and_their_gradients(channels):
    cache = _cache(seed=channels)
    generator = torch.Generator().manual_seed(7)
    source = torch.randn((13, channels, 4), generator=generator, dtype=torch.float64).requires_grad_()
    got = polar._megakernel_features(cache, source)
    expected = _reference(cache, source)
    assert got.shape == expected.shape
    torch.testing.assert_close(got, expected, rtol=1e-11, atol=1e-13)
    weight = torch.randn(got.shape, generator=generator, dtype=torch.float64)
    grads = torch.autograd.grad((weight * got).sum(), _leaves(cache, source))
    expected_grads = torch.autograd.grad((weight * expected).sum(), _leaves(cache, source))
    for g, e in zip(grads, expected_grads):
        torch.testing.assert_close(g, e, rtol=1e-10, atol=1e-12)


def test_only_the_asked_gradients_and_no_k_gradient_in_nvt():
    """Positions and sources only (NVT): the k-vector factors' gradients are not formed."""
    cache = {name: (t.detach() if torch.is_tensor(t) else t) for name, t in _cache(seed=3).items()}
    k, r, mask = cache["k_vectors"], cache["node_positions"].requires_grad_(), cache["fourier_mask"]
    generator = torch.Generator().manual_seed(1)
    K, N = k.shape[0], r.shape[0]
    x = torch.randn((N, 2, 3), generator=generator, dtype=torch.float64).requires_grad_()
    dr, di = torch.randn((K, 3), generator=generator, dtype=torch.float64), torch.randn((K, 3), dtype=torch.float64)
    w = torch.rand(K, dtype=torch.float64)
    pr, pi = torch.randn((K, 5), dtype=torch.float64), torch.randn((K, 5), dtype=torch.float64)
    out = mk.kspace_field(k, r, mask, x, dr, di, w, pr, pi)
    assert out.shape == (N, 2, 5)
    g = torch.randn(out.shape, dtype=torch.float64)
    yc, ys, a, b = mk.dense_forward(k, r, mask, x, dr, di, w, pr, pi)[1:]
    res = mk.kspace_field_backward(k, r, mask, x, g, yc, ys, a, b, dr, di, w, pr, pi, False, False)
    assert res[0].numel() == 0 and all(t.numel() == 0 for t in res[3:])
    grad_r, grad_x = torch.autograd.grad((g * out).sum(), (r, x))
    torch.testing.assert_close(res[1], grad_r)
    torch.testing.assert_close(res[2], grad_x)


def test_the_switch_is_off_without_a_listed_gpu(monkeypatch):
    assert polar.MEGAKERNEL is None and tuple(polar.MEGAKERNEL_GPUS) == ()
    cpu = torch.zeros(2, 3)
    assert not polar.megakernel_on(cpu)
    monkeypatch.setattr(polar, "MEGAKERNEL", True)
    assert not polar.megakernel_on(cpu)  # CUDA float32 only
    assert not polar.megakernel_on(cpu.double())


def test_the_mask_follows_the_megakernel(monkeypatch):
    k_batch = torch.tensor([0, 1, 0])
    positions = torch.zeros(4, 3)
    one = polar._Call(None, {"ptr": torch.tensor([0, 4])})
    monkeypatch.setattr(polar, "megakernel_on", lambda tensor: True)
    assert torch.equal(polar._fourier_mask(k_batch, positions, one), torch.tensor([1.0, 0.0, 1.0]))
    assert math.isclose(mk.S0, 2.0 / (2.0 * math.pi) ** 3)
