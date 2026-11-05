"""fastmace.kernels.triton.edge_sh.edge_sh3 on CUDA (``EDGE_SH``, round 5): values and the vectors' gradient against
e3nn's ``SphericalHarmonics([0, 1, 2, 3], normalize=True, normalization="component")`` in float64 within float32
rounding, zero-length padding rows (normalize's clamp), non-contiguous input, torch.compile, a
captured CUDA graph, and ``EdgeSH``'s fall-through for other inputs."""

import pytest
import torch

from fastmace.kernels.triton.edge_sh import edge_sh3
from fastmace.rewrites.generic import modules as gm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _e3nn():
    torch.serialization.add_safe_globals([slice])  # e3nn 0.4.4 loads its Wigner constants with torch.load
    from e3nn import o3
    return o3.SphericalHarmonics([0, 1, 2, 3], normalize=True, normalization="component")


def _vectors(e=5000, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    v = torch.randn(e, 3, dtype=torch.float64, device="cuda", generator=g)
    v = v / v.norm(dim=1, keepdim=True) * (0.5 + 6.0 * torch.rand(e, 1, dtype=torch.float64, device="cuda",
                                                                   generator=g))
    v[:7] = 0.0
    v[7] = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)  # the poles
    v[8] = torch.tensor([0.0, -2.5, 0.0], dtype=torch.float64)
    return v, torch.randn(e, 16, dtype=torch.float64, device="cuda", generator=g)


def _grad(fn, v, g):
    v = v.detach().clone().requires_grad_(True)
    out = fn(v)
    (gv,) = torch.autograd.grad(out, v, g)
    return out.detach(), gv


def test_values_and_gradient_match_e3nn_float64():
    sh = _e3nn().cuda()
    v, g = _vectors()
    ref, ref_grad = _grad(sh, v, g)
    out, grad = _grad(edge_sh3, v.float(), g.float())
    assert out.dtype == torch.float32 and out.shape == (v.shape[0], 16)
    assert float((out.double() - ref).abs().max()) <= 2e-6 * float(ref.abs().max())
    rows = slice(7, None)  # zero vectors apart: normalize's clamp gives them g / eps, e3nn's too (checked below)
    assert float((grad[rows].double() - ref_grad[rows]).abs().max()) <= 2e-6 * float(ref_grad[rows].abs().max())
    assert torch.allclose(grad[:7].double(), ref_grad[:7], rtol=1e-5, atol=0)
    assert torch.equal(out[:7, 1:], torch.zeros_like(out[:7, 1:])) and bool((out[:, 0] == 1).all())


def test_strided_input_compile_and_graph():
    v, g = _vectors(4096, seed=1)
    v32, g32 = v.float(), g.float()
    wide = torch.zeros(v32.shape[0], 5, device="cuda")
    wide[:, 1:4] = v32
    strided = wide[:, 1:4]
    assert not strided.is_contiguous()
    out, grad = _grad(edge_sh3, v32, g32)
    out_s, grad_s = _grad(edge_sh3, strided, g32)
    assert torch.equal(out, out_s) and torch.equal(grad, grad_s)
    compiled = torch.compile(lambda x: edge_sh3(x) * 2.0, fullgraph=True)
    out_c, grad_c = _grad(compiled, v32, g32)
    assert torch.equal(out_c, 2.0 * out) and torch.allclose(grad_c, 2.0 * grad, rtol=0, atol=1e-6)
    static = v32.clone()
    edge_sh3(static)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = edge_sh3(static)
    static.copy_(v32.flip(0))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, edge_sh3(v32.flip(0)))


def test_edge_sh_module_falls_through():
    sh = _e3nn().cuda()
    assert gm.edge_sh_replaceable(sh)
    assert not gm.edge_sh_replaceable(_e3nn().__class__([0, 1, 2], normalize=True, normalization="component"))
    module = gm.EdgeSH(sh)
    v, _ = _vectors(64, seed=2)
    assert torch.allclose(module(v), sh(v), rtol=0, atol=1e-12)  # float64: the original module
    assert torch.allclose(module(v.float()).double(), sh(v), rtol=0, atol=2e-6)
