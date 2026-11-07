"""``kspace_megakernel.CUDA = True``: the CUDA C++ kernels (``fastmace.kernels.cuda.kspace``) against the dense
formulas in float64, values and every gradient, on POLAR-1 sizes (NaCl 243 k-vectors x 216 atoms, NbCl5 2989 x
1944; J = 4, F = 8, one and two spins), the Coulomb energy's structure factor, determinism and CUDA-graph replays
(the grid barriers' counters)."""

import pytest
import torch

from fastmace.kernels.cuda import kspace as cuda_kspace
from fastmace.kernels.triton import kspace_megakernel as mk

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and cuda_kspace.available("cuda")),
                                reason="needs CUDA and a k-space cubin for this GPU")

SIZES = [(2989, 1944, 2), (243, 216, 2), (243, 216, 1), (70, 33, 1), (5, 300, 2)]


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


def _close(got, expected, rtol):
    scale = expected.abs().max().clamp_min(1e-30)
    err = float((got.double() - expected).abs().max() / scale)
    assert err < rtol, err


def _field(args, weight, wanted):
    leaves = [t.clone().requires_grad_(i in wanted) for i, t in enumerate(args)]
    out = mk.kspace_field(*leaves)
    grads = torch.autograd.grad((weight * out).sum(), [leaves[i] for i in wanted])
    return [out.detach()] + list(grads)


@pytest.mark.parametrize("K,N,C", SIZES)
@pytest.mark.parametrize("wanted", [(0, 1, 3, 4, 5, 6, 7, 8), (1, 3), (0, 1, 3)])
def test_field_matches_float64(monkeypatch, K, N, C, wanted):
    monkeypatch.setattr(mk, "CUDA", True)
    args = _inputs(K, N, C, 4, 8)
    weight = torch.randn((N, C, 8), dtype=torch.float64, device="cuda")
    expected = _field(args, weight, wanted)  # float64 on CUDA: the dense formulas
    called = []
    original = cuda_kspace.field_backward
    monkeypatch.setattr(cuda_kspace, "field_backward", lambda *a: called.append(1) or original(*a))
    got = _field([t.float() for t in args], weight.float(), wanted)
    assert called, "the CUDA backward did not run"
    for g, e in zip(got, expected):
        assert g.dtype == torch.float32
        _close(g, e, 3e-4)


@pytest.mark.parametrize("K,N", [(2989, 1944), (243, 216), (70, 33)])
@pytest.mark.parametrize("W", [4, 8])
@pytest.mark.parametrize("grad_k", [False, True])
def test_structure_factor_matches_float64(K, N, W, grad_k):
    k, r, mask = _inputs(K, N, 1, W, 8)[:3]
    x = torch.randn((N, W), dtype=torch.float64, device="cuda")
    wc, ws = torch.randn((K, W), dtype=torch.float64, device="cuda"), torch.randn((K, W), dtype=torch.float64,
                                                                                    device="cuda")

    def run(dtype, structure_factor):
        leaves = [t.to(dtype).clone().requires_grad_(i != 2 and (i != 0 or grad_k))
                  for i, t in enumerate((k, r, mask, x))]
        yc, ys = structure_factor(*leaves)
        grads = torch.autograd.grad((wc.to(dtype) * yc + ws.to(dtype) * ys).sum(),
                                    [t for t in leaves if t.requires_grad])
        return [yc.detach(), ys.detach()] + list(grads)

    expected = run(torch.float64, cuda_kspace.structure_factor)  # float64: the dense formulas
    for g, e in zip(run(torch.float32, cuda_kspace.structure_factor), expected):
        assert g.dtype == torch.float32
        _close(g, e, 3e-4)


def test_kernels_are_deterministic_and_replay_in_a_graph(monkeypatch):
    monkeypatch.setattr(mk, "CUDA", True)
    k, r, mask, x, dr, di, w, pr, pi = [t.float() for t in _inputs(2989, 1944, 2, 4, 8, seed=3)]
    g = torch.randn((1944, 2, 8), device="cuda")
    first = mk.kspace_field_forward(k, r, mask, x, dr, di, w, pr, pi)
    second = mk.kspace_field_forward(k, r, mask, x, dr, di, w, pr, pi)
    assert all(torch.equal(p, q) for p, q in zip(first, second))
    back = mk.kspace_field_backward(k, r, mask, x, g, *first[1:], dr, di, w, pr, pi, True, True)
    sf = cuda_kspace.sf_forward(k, r, mask, x.reshape(1944, 8))
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
        sf_out = cuda_kspace.sf_forward(k, r, mask, x.reshape(1944, 8))
    for _ in range(5):  # every replay passes its barriers (the counters keep counting)
        graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(p, q) for p, q in zip(out, first))
    assert all(torch.equal(p, q) for p, q in zip(res, back))
    assert all(torch.equal(p, q) for p, q in zip(sf_out, sf))
