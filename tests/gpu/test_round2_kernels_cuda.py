"""Round-2 kernels (5 November 2026) on CUDA: the outer-sum backward's run walk (``edge_outer_sum.BACKWARD_WALK``)
and the tabled outer sum's fused backward (``tabled_outer_sum``), against float64 PyTorch references."""

import pytest
import torch

from fastmace.kernels import triton_available
from fastmace.kernels.triton import edge_outer_sum as eos
from fastmace.kernels.triton import tabled_outer_sum as tos
from fastmace.kernels.triton.radial_table import evaluate

pytestmark = pytest.mark.skipif(not triton_available(), reason=triton_available().reason)


def _inputs(edges=400, k=64, d=16, slots=41, seed=5):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn((edges, k + 1), generator=g, dtype=torch.float64)[:, :k]  # a strided slice, as in the plan
    y = torch.randn((edges, d), generator=g, dtype=torch.float64)
    key = torch.sort(torch.randint(0, slots, (edges,), generator=g)).values
    grad = torch.randn((slots, d * k), generator=g, dtype=torch.float64)
    return a, y, key, grad, slots


@pytest.mark.parametrize("sorted_keys", [True, False])
def test_backward_walk_matches_float64(monkeypatch, sorted_keys):
    a, y, key, grad, slots = _inputs()
    if not sorted_keys:
        key = key[torch.randperm(key.numel(), generator=torch.Generator().manual_seed(1))]
    rows = grad.view(slots, y.shape[1], a.shape[1])[key]
    ref_a, ref_y = torch.einsum("edk,ed->ek", rows, y), torch.einsum("edk,ek->ed", rows, a)
    monkeypatch.setattr(eos, "BACKWARD_WALK", True)
    grad_a, grad_y = eos.edge_outer_sum_backward(grad.float().cuda(), a.float().cuda(), y.float().cuda(), key.cuda())
    torch.testing.assert_close(grad_a.double().cpu(), ref_a, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grad_y.double().cpu(), ref_y, rtol=1e-5, atol=1e-5)


def _table(pairs=4, pieces=8, terms=6, columns=66, seed=7):
    """A random piecewise table on [0.5, 6) with eight equal pieces (the layouts of radial_table)."""
    g = torch.Generator().manual_seed(seed)
    r_lo, r_max, n_bins = 0.5, 6.0, 64
    width = (r_max - r_lo) / pieces
    coef = torch.randn((pairs, pieces, terms, columns), generator=g, dtype=torch.float64)
    dcoef = torch.randn((pairs, pieces, terms - 1, columns), generator=g, dtype=torch.float64)
    starts = r_lo + width * torch.arange(pieces, dtype=torch.float64)
    scales = torch.full((pieces,), 2.0 / width, dtype=torch.float64)
    bins = (torch.arange(n_bins) * pieces // n_bins).to(torch.int32)
    return coef, dcoef, starts, scales, bins, r_lo, n_bins / (r_max - r_lo), r_max


@pytest.mark.parametrize("tabled", [False, True])
@pytest.mark.parametrize("density", [True, False])
def test_tabled_outer_sum_matches_float64(monkeypatch, density, tabled):
    monkeypatch.setattr(tos, "TABLED_FORWARD", tabled)
    coef, dcoef, starts, scales, bins, r_lo, bin_scale, r_max = _table()
    edges, d, k, slots = 500, 16, 64, 53
    g = torch.Generator().manual_seed(11)
    r = (torch.rand((edges, 1), generator=g, dtype=torch.float64) * 6.4 + 0.2)  # some below r_lo, some past r_max
    pair = torch.randint(0, coef.shape[0], (edges,), generator=g).to(torch.int32)
    y = torch.randn((edges, d), generator=g, dtype=torch.float64)
    key = torch.sort(torch.randint(0, slots, (edges,), generator=g)).values
    grad = torch.randn((slots, d * k), generator=g, dtype=torch.float64)
    grad_density = torch.randn((edges, 1), generator=g, dtype=torch.float64)
    # float64 reference: a = table values, S = sum y a^T, dr = sum_j dL/dg_j g_j'(r)
    values = evaluate(r, pair, coef, starts, scales, bins, r_lo, bin_scale, r_max)
    slopes = evaluate(r, pair, dcoef, starts, scales, bins, r_lo, bin_scale, r_max, derivative=True)
    a = values[:, :k]
    rows = grad.view(slots, d, k)[key]
    grad_a = torch.einsum("edk,ed->ek", rows, y)
    ref_dr = (grad_a * slopes[:, :k]).sum(1) + (grad_density[:, 0] * slopes[:, -1] if density else 0.0)
    ref_y = torch.einsum("edk,ek->ed", rows, a)
    ref_s = torch.zeros((slots, d * k), dtype=torch.float64).index_add(
        0, key, (y[:, :, None] * a[:, None, :]).reshape(edges, -1))

    cuda = [t.float().cuda() if t.is_floating_point() else t.cuda() for t in
            (coef, dcoef, starts, scales, bins, r, pair, y, key)]
    coef_c, dcoef_c, starts_c, scales_c, bins_c, r_c, pair_c, y_c, key_c = cuda
    r_c.requires_grad_(True)
    y_c.requires_grad_(True)
    summed, edge_density = tos.tabled_outer_sum(r_c, pair_c, coef_c, dcoef_c, starts_c, scales_c, bins_c, r_lo,
                                                bin_scale, r_max, k, y_c, key_c, slots)
    torch.testing.assert_close(summed.double().cpu(), ref_s, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(edge_density.double().cpu()[:, 0], values[:, -1], rtol=1e-5, atol=1e-5)
    outputs, grads = [summed], [grad.float().cuda()]
    if density:
        outputs.append(edge_density)
        grads.append(grad_density.float().cuda())
    dr, dy = torch.autograd.grad(outputs, (r_c, y_c), grads)
    scale = ref_dr.abs().max()
    assert (dr.double().cpu()[:, 0] - ref_dr).abs().max() / scale < 1e-5
    assert (dy.double().cpu() - ref_y).abs().max() / ref_y.abs().max() < 1e-5  # float32 sums of 64 terms


@pytest.mark.parametrize("power", [False, True])
def test_packed_coef_changes_no_bit(monkeypatch, power):
    """PACKED_COEF (idea 3): the walks reading the channel-pair packed series give the unpacked walks' bits (the
    edge density, dL/dr and grad_y; the slot sums up to the atomics' order across programs)."""
    monkeypatch.setattr(tos, "TABLED_FORWARD", True)
    monkeypatch.setattr(tos, "POWER_BACKWARD", power)
    from fastmace.rewrites.radial import power_basis

    coef, dcoef, starts, scales, bins, r_lo, bin_scale, r_max = _table()
    edges, d, k, slots = 500, 16, 64, 53
    g = torch.Generator().manual_seed(13)
    r = (torch.rand((edges, 1), generator=g, dtype=torch.float64) * 6.4 + 0.2).float().cuda()
    pair = torch.randint(0, coef.shape[0], (edges,), generator=g).to(torch.int32).cuda()
    y = torch.randn((edges, d), generator=g).cuda()
    key = torch.sort(torch.randint(0, slots, (edges,), generator=g)).values.cuda()
    grads = [torch.randn((slots, d * k), generator=g).cuda(), torch.randn((edges, 1), generator=g).cuda()]
    coef_c, dcoef_c, starts_c, scales_c, bins_c = (t.float().cuda() if t.is_floating_point() else t.cuda()
                                                   for t in (coef, dcoef, starts, scales, bins))
    pcoef_c = power_basis(coef_c)
    results = []
    for packed in (False, True):
        r_c, y_c = r.clone().requires_grad_(True), y.clone().requires_grad_(True)
        summed, density = tos.tabled_outer_sum(
            r_c, pair, coef_c, dcoef_c, starts_c, scales_c, bins_c, r_lo, bin_scale, r_max, k, y_c, key, slots,
            pcoef=pcoef_c, packed=tos.packed_coef(coef_c) if packed else None,
            ppacked=tos.packed_coef(pcoef_c) if packed else None)
        dr, dy = torch.autograd.grad([summed, density], (r_c, y_c), grads)
        results.append((summed, density, dr, dy))
    (s0, e0, r0, y0), (s1, e1, r1, y1) = results
    assert torch.equal(e0, e1) and torch.equal(r0, r1) and torch.equal(y0, y1)
    torch.testing.assert_close(s1, s0, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("power", [False, True])
def test_recomputing_backward_allocates_no_value_rows(monkeypatch, power):
    """With TABLED_FORWARD the fused backward evaluates the table's values again inside the kernel; it passed a
    shape-only ``[E, K]`` tensor, 4.1 GiB at SrTiO3 233,280 on the RTX 4000 Ada (18.9 kB/atom at the force peak,
    results/memory-profile-20261109). SHAPE_CARRIER: a stride-0 view carries the shape. The backward's peak above its
    inputs stays below one ``[E, K]`` float block with it and reaches one without it;
    the gradients are the same bits."""
    monkeypatch.setattr(tos, "TABLED_FORWARD", True)
    monkeypatch.setattr(tos, "POWER_BACKWARD", power)
    from fastmace.rewrites.radial import power_basis

    coef, dcoef, starts, scales, bins, r_lo, bin_scale, r_max = _table()
    edges, d, k, slots = 200_000, 16, 64, 4001
    g = torch.Generator().manual_seed(17)
    r0 = (torch.rand((edges, 1), generator=g) * 6.4 + 0.2).cuda()
    pair = torch.randint(0, coef.shape[0], (edges,), generator=g).to(torch.int32).cuda()
    y0 = torch.randn((edges, d), generator=g).cuda()
    key = torch.sort(torch.randint(0, slots, (edges,), generator=g)).values.cuda()
    coef_c, dcoef_c, starts_c, scales_c, bins_c = (t.float().cuda() if t.is_floating_point() else t.cuda()
                                                   for t in (coef, dcoef, starts, scales, bins))
    pcoef_c = power_basis(coef_c)
    grads = [torch.randn((slots, d * k), generator=g).cuda(), torch.randn((edges, 1), generator=g).cuda()]
    peaks, results = {}, {}
    for carrier in (True, False):
        monkeypatch.setattr(tos, "SHAPE_CARRIER", carrier)
        r, y = r0.clone().requires_grad_(True), y0.clone().requires_grad_(True)
        summed, density = tos.tabled_outer_sum(r, pair, coef_c, dcoef_c, starts_c, scales_c, bins_c, r_lo, bin_scale,
                                               r_max, k, y, key, slots, pcoef=pcoef_c)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        results[carrier] = torch.autograd.grad([summed, density], (r, y), grads)
        torch.cuda.synchronize()
        peaks[carrier] = torch.cuda.max_memory_allocated() - before
        del summed, density
    block = edges * k * 4
    assert peaks[True] < block <= peaks[False], peaks
    assert all(torch.equal(a, b) for a, b in zip(results[True], results[False]))


@pytest.mark.parametrize("power", [True, False])
@pytest.mark.parametrize("inner", [1, 3])
def test_nested_split_walks_match_the_full_walks(monkeypatch, inner, power):
    """rewrites.radial.nested's split walks (``r_far``): on a table whose channels K/2.. are zero from ``r_far`` (a
    piece start), the sums, the edge density, dL/dr and grad_y of the full walks, to rounding; the backward in the
    power basis or by recompute (NESTED_POWER False with POWER_BACKWARD off)."""
    monkeypatch.setattr(tos, "TABLED_FORWARD", True)
    monkeypatch.setattr(tos, "POWER_BACKWARD", power)
    monkeypatch.setattr(tos, "NESTED_POWER", power)
    from fastmace.rewrites.radial import power_basis

    coef, dcoef, starts, scales, bins, r_lo, bin_scale, r_max = _table()
    first_far = 5
    coef[:, first_far:, :, 32:64] = 0.0
    dcoef[:, first_far:, :, 32:64] = 0.0
    r_far = float(starts[first_far].float())
    edges, d, k, slots = 500, 16, 64, 54
    g = torch.Generator().manual_seed(17)
    r = (torch.rand((edges, 1), generator=g, dtype=torch.float64) * 6.4 + 0.2).float().cuda()
    pair = torch.randint(0, coef.shape[0], (edges,), generator=g).to(torch.int32).cuda()
    y = torch.randn((edges, d), generator=g).cuda()
    key = torch.sort(torch.randint(0, slots, (edges,), generator=g)).values.cuda()
    grads = [torch.randn((slots, d * k), generator=g).cuda(),  # the op's [slots, D K] in either layout
             torch.randn((edges, 1), generator=g).cuda()]
    coef_c, dcoef_c, starts_c, scales_c, bins_c = (t.float().cuda() if t.is_floating_point() else t.cuda()
                                                   for t in (coef, dcoef, starts, scales, bins))
    pcoef_c = power_basis(coef_c)
    results = []
    for far in (0.0, r_far):
        r_c, y_c = r.clone().requires_grad_(True), y.clone().requires_grad_(True)
        summed, density = tos.tabled_outer_sum(r_c, pair, coef_c, dcoef_c, starts_c, scales_c, bins_c, r_lo,
                                               bin_scale, r_max, k, y_c, key, slots, inner, pcoef=pcoef_c, r_far=far)
        dr, dy = torch.autograd.grad([summed, density], (r_c, y_c), grads)
        results.append((summed, density, dr, dy))
    (s0, e0, r0, y0), (s1, e1, r1, y1) = results
    assert torch.equal(e0, e1)
    for new, ref in ((s1, s0), (r1, r0), (y1, y0)):
        assert (new - ref).abs().max() <= 1e-5 * ref.abs().max()
    assert (r >= r_far).float().mean() > 0.3  # both branches ran
