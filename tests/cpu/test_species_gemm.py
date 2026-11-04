"""fastmace.kernels.triton.species_gemm on the CPU: the plan and the kernel's tile rule (fixed count, every row of
every species in exactly one tile of its species, empty tiles past the last), the PyTorch fallback of
``species_mm`` and its gradient, and ``SpeciesSkip`` with ``SPECIES_GEMM`` equal to the concatenated product."""

import pytest
import torch

from fastmace.kernels.triton import species_gemm as sg


@pytest.mark.parametrize("counts, size", [((1000, 1000, 3000), 3), ((5, 0, 7), 1), ((64,), 7), ((1, 2, 3, 4), 16)])
def test_plan_covers_every_row_once(counts, size):
    index = torch.cat([torch.full((n,), s) for s, n in enumerate(counts)])
    index = index[torch.randperm(index.numel(), generator=torch.Generator().manual_seed(0))]
    order, found = sg.species_plan(index, len(counts))
    atoms = index.numel()
    assert found.tolist() == list(counts)
    assert sorted(order.tolist()) == list(range(atoms))
    assert index[order.long()].tolist() == sorted(index.tolist())
    seen = []
    for t in range(sg.tiles(atoms, size, len(counts))):
        species, first, end = sg.tile_rows(t, list(counts), size)
        assert 0 <= end - first <= sg.BLOCK_M
        for r in range(first, end):
            atom, j = int(order[r // size]), r % size
            assert int(index[atom]) == species
            seen.append((atom, j))
    assert sorted(seen) == [(a, j) for a in range(atoms) for j in range(size)]


def test_fallback_and_gradient():
    g = torch.Generator().manual_seed(1)
    index = torch.tensor([2, 0, 1, 2, 2, 0])
    x = torch.randn(6, 3, 5, generator=g, dtype=torch.float64).transpose(1, 2).contiguous().transpose(1, 2)
    x.requires_grad_()
    w = torch.randn(3, 5, 4, generator=g, dtype=torch.float64)
    order, counts = sg.species_plan(index, 3)
    y = sg.species_mm(x, w, w.transpose(1, 2).contiguous(), index, order, counts, "ieee")
    expected = torch.stack([x[a] @ w[index[a]] for a in range(6)])
    assert torch.allclose(y, expected, atol=1e-14)
    grad = torch.randn(6, 3, 4, generator=g, dtype=torch.float64)
    (gx,) = torch.autograd.grad(y, x, grad)
    assert torch.allclose(gx, torch.stack([grad[a] @ w[index[a]].T for a in range(6)]), atol=1e-14)


@pytest.mark.parametrize("layout", ["ir_mul", "mul_ir"])
def test_species_skip_grouped_matches(monkeypatch, layout):
    from fastmace.rewrites.generic import modules as gm

    g = torch.Generator().manual_seed(2)
    mul = 4
    segments = [(0, mul, mul), (mul, mul, 3 * mul)]
    blocks = [(0, 0, torch.randn(3, mul, mul, generator=g, dtype=torch.float64)),
              (1, 1, torch.randn(3, mul, mul, generator=g, dtype=torch.float64))]
    skip = gm.SpeciesSkip(torch.tensor([3, 7, 11]), segments, segments, blocks, layout).double()
    x1 = torch.randn(9, 4 * mul, generator=g, dtype=torch.float64)
    x2 = torch.zeros(9, 89, dtype=torch.float64)
    x2[torch.arange(9), torch.tensor([3, 7, 11, 11, 3, 7, 7, 11, 3])] = 1
    reference = skip(x1, x2)
    monkeypatch.setattr(gm, "SPECIES_GEMM", True)
    assert torch.allclose(skip(x1, x2), reference, atol=1e-13)  # the CPU stays on the concatenated product
