"""fastmace.kernels.triton.radial_table on CUDA: the Triton kernels against ``evaluate``, their PyTorch twin.

A random table (3 pairs, 5 pieces of uneven width, degree 10, J = 66 outputs) at lengths across the domain,
below its lower end (evaluated there, zero derivative) and beyond the cutoff (exact zeros), forward and
backward, eager and in a CUDA graph.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _table(generator):
    pairs, terms, width = 3, 11, 66
    starts = torch.tensor([0.3, 1.0, 1.5, 3.0, 4.5])
    ends = torch.tensor([1.0, 1.5, 3.0, 4.5, 6.0])
    coef = torch.randn((pairs, 5, terms, width), generator=generator) / torch.arange(1, terms + 1)[None, None, :, None]
    dcoef = torch.randn((pairs, 5, terms - 1, width), generator=generator)
    n_bins = 256
    centres = 0.3 + (torch.arange(n_bins) + 0.5) * (6.0 - 0.3) / n_bins
    bins = (torch.searchsorted(starts, centres, right=True) - 1).to(torch.int32)
    return dict(coef=coef, dcoef=dcoef, starts=starts, scales=2.0 / (ends - starts), bins=bins,
                r_lo=0.3, bin_scale=n_bins / 5.7, r_max=6.0)


def test_kernels_equal_their_pytorch_twin():
    from fastmace.kernels.triton.radial_table import evaluate, radial_table_backward, radial_table_forward

    generator = torch.Generator().manual_seed(7)
    table = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in _table(generator).items()}
    edges = 4099
    r = torch.rand(edges, generator=generator) * 6.4
    r[:5] = torch.tensor([0.1, 0.3, 5.999, 6.0, 7.5])
    r = r.cuda()[:, None]
    pair = torch.randint(0, 3, (edges,), generator=generator, dtype=torch.int32).cuda()
    tensors = (table["starts"], table["scales"], table["bins"])
    constants = (table["r_lo"], table["bin_scale"], table["r_max"])
    g, density = radial_table_forward(r, pair, table["coef"], *tensors, *constants)
    expected = evaluate(r, pair, table["coef"], *tensors, *constants)
    torch.testing.assert_close(torch.cat([g, density], 1), expected, rtol=2e-6, atol=2e-6)
    assert torch.equal(g[3:5], torch.zeros_like(g[3:5])) and torch.equal(density[3:5], torch.zeros(2, 1).cuda())
    grad_g = torch.randn((edges, 65), generator=generator).cuda()
    grad_d = torch.randn((edges, 1), generator=generator).cuda()
    grad_r = radial_table_backward(r, pair, table["dcoef"], *tensors, grad_g, grad_d, *constants)
    slopes = evaluate(r, pair, table["dcoef"], *tensors, *constants, derivative=True)
    expected_r = (torch.cat([grad_g, grad_d], 1) * slopes).sum(1, keepdim=True)
    torch.testing.assert_close(grad_r, expected_r, rtol=1e-5, atol=1e-5)
    assert torch.equal(grad_r[:1], torch.zeros(1, 1).cuda())  # below r_lo: evaluated at r_lo, zero slope
    assert torch.equal(grad_r[3:5], torch.zeros(2, 1).cuda())

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = radial_table_forward(r, pair, table["coef"], *tensors, *constants)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured[0], g) and torch.equal(captured[1], density)
