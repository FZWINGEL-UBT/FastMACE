"""``fastmace.kernels.triton.fourier`` on CUDA: the Triton tile kernels (float32) against the dense sums in
float64, values and the backward (positions, k-vectors, thin factors), on POLAR-like sizes and values (phases up
to about 100 rad, a masked k-set)."""

import pytest
import torch

from fastmace.kernels.triton import fourier

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _inputs(K, N, M, seed=0):
    generator = torch.Generator().manual_seed(seed)
    k = 1.5 * torch.randn((K, 3), generator=generator, dtype=torch.float64)
    r = 25.0 * torch.rand((N, 3), generator=generator, dtype=torch.float64)
    mask = (torch.rand(K, generator=generator) > 0.2).to(torch.float64)
    x = torch.randn((N, M), generator=generator, dtype=torch.float64)
    a = torch.randn((K, M), generator=generator, dtype=torch.float64)
    b = torch.randn((K, M), generator=generator, dtype=torch.float64)
    w = torch.randn((K, M), generator=generator, dtype=torch.float64)
    v = torch.randn((N, M), generator=generator, dtype=torch.float64)
    return [t.cuda() for t in (k, r, mask, x, a, b, w, v)]


def _run(function, args, weights, grads_of):
    leaves = [t.clone().requires_grad_(i in grads_of) for i, t in enumerate(args)]
    out = function(*leaves)
    out = out if isinstance(out, tuple) else (out,)
    loss = sum((w * o).sum() for w, o in zip(weights, out))
    grads = torch.autograd.grad(loss, [leaves[i] for i in grads_of])
    return [o.detach() for o in out], list(grads)


def _close(got, expected, rtol):
    scale = expected.abs().max().clamp_min(1e-30)
    assert float((got.double() - expected).abs().max() / scale) < rtol


@pytest.mark.parametrize("K,N,M", [(2989, 1944, 4), (243, 216, 8), (70, 33, 1)])
def test_kernels_match_float64(K, N, M):
    k, r, mask, x, a, b, w, v = _inputs(K, N, M)
    for function, args, weights in ((fourier.structure_factor, (k, r, mask, x), (w, w.flip(0))),
                                    (fourier.fourier_project, (k, r, mask, a, b), (v,))):
        grads_of = [0, 1] + list(range(3, len(args)))
        out64, grads64 = _run(function, args, weights, grads_of)  # float64 on CUDA: the dense sums
        out32, grads32 = _run(function, [t.float() for t in args], [t.float() for t in weights], grads_of)
        for got, expected in zip(out32 + grads32, out64 + grads64):
            assert got.dtype == torch.float32
            _close(got, expected, 2e-4)


def test_kernels_are_deterministic():
    k, r, mask, x, *_ = [t.float() for t in _inputs(2989, 1944, 4)]
    first = fourier.fourier_ksum(k, r, mask, x)
    second = fourier.fourier_ksum(k, r, mask, x)
    assert all(torch.equal(p, q) for p, q in zip(first, second))
