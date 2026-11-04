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
    # Coefficients of order 1 summed over 11 terms: FMA against separate multiply-adds differ by a few float32 ulps.
    torch.testing.assert_close(torch.cat([g, density], 1), expected, rtol=1e-5, atol=2e-5)
    assert torch.equal(g[3:5], torch.zeros_like(g[3:5])) and torch.equal(density[3:5], torch.zeros(2, 1).cuda())
    grad_g = torch.randn((edges, 65), generator=generator).cuda()
    grad_d = torch.randn((edges, 1), generator=generator).cuda()
    grad_r = radial_table_backward(r, pair, table["dcoef"], *tensors, grad_g, grad_d, *constants)
    slopes = evaluate(r, pair, table["dcoef"], *tensors, *constants, derivative=True)
    expected_r = (torch.cat([grad_g, grad_d], 1) * slopes).sum(1, keepdim=True)
    torch.testing.assert_close(grad_r, expected_r, rtol=1e-5, atol=1e-4)
    assert torch.equal(grad_r[:1], torch.zeros(1, 1).cuda())  # below r_lo: evaluated at r_lo, zero slope
    assert torch.equal(grad_r[3:5], torch.zeros(2, 1).cuda())

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = radial_table_forward(r, pair, table["coef"], *tensors, *constants)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured[0], g) and torch.equal(captured[1], density)


def test_without_density_the_folded_input_and_its_gradient_are_unchanged():
    """``density=False`` (residual blocks): the same folded input bit for bit, an ``[E, 0]`` density, and the
    lengths' gradient of a zero density gradient."""
    from fastmace.kernels.triton.radial_table import radial_table

    generator = torch.Generator().manual_seed(11)
    table = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in _table(generator).items()}
    edges = 3001
    r = (torch.rand(edges, generator=generator) * 6.4).cuda()[:, None]
    pair = torch.randint(0, 3, (edges,), generator=generator, dtype=torch.int32).cuda()
    args = (pair, table["coef"], table["dcoef"], table["starts"], table["scales"], table["bins"], table["r_lo"],
            table["bin_scale"], table["r_max"])
    grad_g = torch.randn((edges, 65), generator=generator).cuda()
    results = []
    for density in (True, False):
        leaf = r.clone().requires_grad_()
        g, d = radial_table(leaf, *args, density=density)
        assert d.shape == (edges, int(density))
        (g * grad_g).sum().backward()
        results.append((g.detach(), leaf.grad))
    assert torch.equal(results[0][0], results[1][0]) and torch.equal(results[0][1], results[1][1])


@pytest.mark.parametrize("density", [True, False])
def test_column_blocks_give_the_same_values_bit_for_bit(monkeypatch, density):
    """COLUMN_BLOCK = 64 (J = 66 in two column programs): every column, and the density, bit for bit."""
    from fastmace.kernels.triton import radial_table as rt

    generator = torch.Generator().manual_seed(13)
    table = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in _table(generator).items()}
    edges = 4099
    r = (torch.rand(edges, generator=generator) * 6.4).cuda()[:, None]
    pair = torch.randint(0, 3, (edges,), generator=generator, dtype=torch.int32).cuda()
    args = (r, pair, table["coef"], table["starts"], table["scales"], table["bins"], table["r_lo"],
            table["bin_scale"], table["r_max"], density)
    whole = rt.radial_table_forward(*args)
    monkeypatch.setattr(rt, "COLUMN_BLOCK", 64)
    blocked = rt.radial_table_forward(*args)
    assert torch.equal(whole[0], blocked[0]) and torch.equal(whole[1], blocked[1])


@pytest.mark.parametrize("density", [True, False])
def test_leading_columns_bit_for_bit(density):
    """``columns=64`` (the reassociated first interaction, ``FIRST_TABLE_COLUMNS``): the full folded input's leading
    64 columns, the same density, and bit for bit the lengths' gradient of the full input with a zero last column."""
    from fastmace.kernels.triton.radial_table import radial_table

    generator = torch.Generator().manual_seed(17)
    table = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in _table(generator).items()}
    edges = 4099
    r = (torch.rand(edges, generator=generator) * 6.4).cuda()[:, None]
    pair = torch.randint(0, 3, (edges,), generator=generator, dtype=torch.int32).cuda()
    args = (pair, table["coef"], table["dcoef"], table["starts"], table["scales"], table["bins"], table["r_lo"],
            table["bin_scale"], table["r_max"], density)
    grad_g = torch.randn((edges, 64), generator=generator).cuda()
    grad_d = torch.randn((edges, int(density)), generator=generator).cuda()
    results = []
    for columns in (-1, 64):
        leaf = r.clone().requires_grad_()
        g, d = radial_table(leaf, *args, columns=columns)
        assert g.shape == (edges, 65 if columns < 0 else 64)
        ((g[:, :64] * grad_g).sum() + (d * grad_d).sum()).backward()
        results.append((g[:, :64].detach(), d.detach(), leaf.grad))
    for a, b in zip(*results):
        assert torch.equal(a, b)
