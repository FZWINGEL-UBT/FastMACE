"""fastmace.kernels.triton.w_table on CUDA (ideas-20261027 D2): the w-table kernels against the chain they replace.

A random float64 table (3 pairs, 5 pieces of uneven width, degree 16, 66 outputs) and a folded weight [80, 65]. At
lengths across the domain, below its lower end and beyond the cutoff, the per-edge weights must equal
``table -> a -> a W^T`` and the r-gradient must equal ``(dw W) . da/dr`` from the table's derivative series, to
float64 rounding; the same eager and replayed in a CUDA graph.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _table(generator):
    pairs, terms, width = 3, 17, 66
    starts = torch.tensor([0.3, 1.0, 1.5, 3.0, 4.5], dtype=torch.float64)
    ends = torch.tensor([1.0, 1.5, 3.0, 4.5, 6.0], dtype=torch.float64)
    scale = torch.arange(1, terms + 1, dtype=torch.float64)[None, None, :, None]
    coef = torch.randn((pairs, 5, terms, width), generator=generator, dtype=torch.float64) / scale
    dcoef = torch.randn((pairs, 5, terms, width), generator=generator, dtype=torch.float64) / scale
    n_bins = 256
    centres = 0.3 + (torch.arange(n_bins, dtype=torch.float64) + 0.5) * (6.0 - 0.3) / n_bins
    bins = (torch.searchsorted(starts, centres, right=True) - 1).to(torch.int32)
    return dict(coef=coef, dcoef=dcoef, starts=starts, scales=2.0 / (ends - starts), bins=bins,
                r_lo=0.3, bin_scale=n_bins / 5.7, r_max=6.0)


def _chain(table, weight, r, pair, dw):
    from fastmace.kernels.triton.radial_table import evaluate

    tensors = (table["starts"], table["scales"], table["bins"])
    constants = (table["r_lo"], table["bin_scale"], table["r_max"])
    a = evaluate(r, pair, table["coef"], *tensors, *constants)[:, :-1]
    slope = evaluate(r, pair, table["dcoef"], *tensors, *constants, derivative=True)[:, :-1]
    return a @ weight.T, ((dw @ weight) * slope).sum(dim=1)


def test_kernels_equal_the_chain_eager_and_in_a_graph():
    from fastmace.kernels.triton.w_table import fold_tables, w_table_backward, w_table_forward

    generator = torch.Generator().manual_seed(1028)
    table = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in _table(generator).items()}
    weight = torch.randn((80, 65), generator=generator, dtype=torch.float64).cuda()
    g, dg = fold_tables(table["coef"], table["dcoef"], weight)
    edges = 4099
    r = torch.rand(edges, generator=generator, dtype=torch.float64) * 6.4
    r[:5] = torch.tensor([0.1, 0.3, 5.999, 6.0, 7.5], dtype=torch.float64)
    r = r.cuda()[:, None]
    pair = torch.randint(0, 3, (edges,), generator=generator, dtype=torch.int32).cuda()
    dw = torch.randn((edges, 80), generator=generator, dtype=torch.float64).cuda()
    args = (g, table["starts"], table["scales"], table["bins"], table["r_lo"], table["bin_scale"], table["r_max"])
    want_w, want_dr = _chain(table, weight, r, pair, dw)

    w, xs, gs, perm, slope = w_table_forward(r, pair, *args)
    dr = w_table_backward(dw, dg, xs, gs, perm, slope)
    scale = want_w.abs().max()
    assert float((w - want_w).abs().max() / scale) < 1e-13
    assert float((dr - want_dr).abs().max() / want_dr.abs().max()) < 1e-13
    assert bool((w[3:5] == 0).all()) and bool((dr[3:5] == 0).all())  # at and beyond the cutoff
    assert float(dr[0]) == 0.0  # below the domain: evaluated at r_lo with a zero slope

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):  # warm up outside the capture
        w_table_forward(r, pair, *args)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        gw, gxs, ggs, gperm, gslope = w_table_forward(r, pair, *args)
        gdr = w_table_backward(dw, dg, gxs, ggs, gperm, gslope)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(gw, w) and torch.equal(gdr, dr)
