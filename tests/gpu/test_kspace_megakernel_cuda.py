"""``fastmace.kernels.triton.kspace_megakernel`` on CUDA: the persistent Triton kernels (float32) against the dense
formulas in float64, values and every gradient, on POLAR-1 sizes (NaCl 243 k-vectors x 216 atoms, NbCl5 2989 x
1944; two spins, J = 4, F = 8), plus determinism and CUDA-graph replays (the grid barrier's counter)."""

import pytest
import torch

from fastmace.kernels.triton import kspace_megakernel as mk

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _inputs(K, N, C, J, F, seed=0):
    generator = torch.Generator().manual_seed(seed)

    def rnd(*shape, scale=1.0, shift=0.0):
        return shift + scale * torch.randn(shape, generator=generator, dtype=torch.float64)

    k = rnd(K, 3, scale=1.5)
    r = 25.0 * torch.rand((N, 3), generator=generator, dtype=torch.float64)
    mask = (torch.rand(K, generator=generator) > 0.2).to(torch.float64)
    x = rnd(N, C, J)
    dr, di = rnd(K, J), rnd(K, J)
    w = torch.rand(K, generator=generator, dtype=torch.float64) + 0.1
    pr, pi = rnd(K, F), rnd(K, F)
    return [t.cuda() for t in (k, r, mask, x, dr, di, w, pr, pi)]


def _run(args, weight):
    leaves = [t.clone().requires_grad_(i != 2) for i, t in enumerate(args)]
    out = mk.kspace_field(*leaves)
    grads = torch.autograd.grad((weight * out).sum(), [t for i, t in enumerate(leaves) if i != 2])
    return [out.detach()] + list(grads)


def _close(got, expected, rtol):
    scale = expected.abs().max().clamp_min(1e-30)
    err = float((got.double() - expected).abs().max() / scale)
    assert err < rtol, err


@pytest.mark.parametrize("K,N,C", [(2989, 1944, 2), (243, 216, 2), (243, 216, 1), (70, 33, 1)])
def test_kernels_match_float64(K, N, C):
    args = _inputs(K, N, C, 4, 8)
    weight = torch.randn((N, C, 8), dtype=torch.float64, device="cuda")
    expected = _run(args, weight)  # float64 on CUDA: the dense formulas
    got = _run([t.float() for t in args], weight.float())
    for g, e in zip(got, expected):
        assert g.dtype == torch.float32
        _close(g, e, 3e-4)


def test_kernels_are_deterministic_and_replay_in_a_graph():
    k, r, mask, x, dr, di, w, pr, pi = [t.float() for t in _inputs(2989, 1944, 2, 4, 8, seed=3)]
    g = torch.randn((1944, 2, 8), device="cuda")
    first = mk.kspace_field_forward(k, r, mask, x, dr, di, w, pr, pi)
    second = mk.kspace_field_forward(k, r, mask, x, dr, di, w, pr, pi)
    assert all(torch.equal(p, q) for p, q in zip(first, second))
    back = mk.kspace_field_backward(k, r, mask, x, g, *first[1:], dr, di, w, pr, pi, True, True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            mk.kspace_field_forward(k, r, mask, x, dr, di, w, pr, pi)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = mk.kspace_field_forward(k, r, mask, x, dr, di, w, pr, pi)
        res = mk.kspace_field_backward(k, r, mask, x, g, *out[1:], dr, di, w, pr, pi, True, True)
    for _ in range(5):  # every replay passes its barriers (the counter keeps counting)
        graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(p, q) for p, q in zip(out, first))
    assert all(torch.equal(p, q) for p, q in zip(res, back))
