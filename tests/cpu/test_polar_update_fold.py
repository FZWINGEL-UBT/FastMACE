"""``update_fold``: PolarMACE's field-update MLP folded at setup (N1: the last linear through ``tp_out``; N4: the
species half of the first linear as a per-species bias). Off-polar-alex lab, 7 November 2026.

The folded forward must match MACE's ``AgnosticEmbeddedOneBodyVariableUpdate.forward`` in value and in the
gradients to the node features and the potential features, on update blocks built like MACE-POLAR-1-S, -M and -L
(one, two and three ``tp_out`` paths), in both layouts.
"""

from __future__ import annotations

import copy

import pytest
import torch

field_blocks = pytest.importorskip("mace.modules.field_blocks")
from e3nn import o3  # noqa: E402

from fastmace.rewrites.polar import update_fold, uvu  # noqa: E402


def _block(lmax, channels=16, species=5, seed=0):
    torch.manual_seed(seed)
    node_irreps = o3.Irreps("+".join(f"{channels}x{degree}{'eo'[degree % 2]}" for degree in range(lmax + 1)))
    block = field_blocks.AgnosticEmbeddedOneBodyVariableUpdate(
        node_attrs_irreps=o3.Irreps(f"{species}x0e"), node_feats_irreps=node_irreps,
        edge_attrs_irreps=o3.Irreps("1x0e+1x1o"), edge_feats_irreps=o3.Irreps("8x0e"), target_irreps=node_irreps,
        hidden_irreps=node_irreps, avg_num_neighbors=10.0, potential_irreps=o3.Irreps("4x0e+4x1o"),
        charges_irreps=o3.Irreps("1x0e+1x1o"), field_norm_factor=1.0)
    with torch.no_grad():  # random tp_out and MLP weights (the defaults may be zero or ones)
        for parameter in block.parameters():
            parameter.add_(0.3 * torch.randn_like(parameter))
    return block.double()


def _inputs(block, atoms=11, seed=1):
    generator = torch.Generator().manual_seed(seed)
    species = block.node_attrs_irreps.dim
    node_attrs = torch.nn.functional.one_hot(torch.randint(species, (atoms,), generator=generator), species).double()
    node_feats = torch.randn(atoms, block.node_feats_irreps.dim, generator=generator, dtype=torch.float64)
    potential = torch.randn(atoms, block.potential_irreps.dim, generator=generator, dtype=torch.float64)
    charges = torch.randn(atoms, block.charges_irreps.dim, generator=generator, dtype=torch.float64)
    return node_attrs, node_feats, potential, charges


def _run(block, node_attrs, node_feats, potential, charges):
    x = node_feats.clone().requires_grad_()
    p = potential.clone().requires_grad_()
    out = block(node_attrs, x, None, None, None, p, charges)
    cotangent = torch.linspace(-1, 1, out.numel(), dtype=out.dtype).view_as(out)
    gx, gp = torch.autograd.grad(out, (x, p), cotangent)
    return out.detach(), gx, gp


def test_the_constant_is_off():
    assert update_fold.FOLD_UPDATE is False


@pytest.mark.parametrize("lmax", [0, 1, 2])
def test_the_folded_block_matches_mace(lmax):
    block = _block(lmax)
    assert update_fold.support(block) == (True, "ok")
    folded = copy.deepcopy(block)
    assert update_fold.apply(folded) == [""]
    assert len(folded.tp_out._path_meta) == lmax + 1
    inputs = _inputs(block)
    for expected, got in zip(_run(block, *inputs), _run(folded, *inputs)):
        torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)
    assert update_fold.apply(folded) == []  # once


class _Wrapped(torch.nn.Module):
    """A source_embedding another rewrite replaced (as BlockLinear does): no ``irreps_in``."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)


def test_a_replaced_source_embedding_is_folded_as_it_runs():
    block = _block(1)
    block.source_embedding = _Wrapped(block.source_embedding)
    folded = copy.deepcopy(block)
    assert update_fold.apply(folded) == [""]
    inputs = _inputs(block)
    for expected, got in zip(_run(block, *inputs), _run(folded, *inputs)):
        torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)


def test_uvu_apply_folds_only_with_the_constant(monkeypatch):
    holder = torch.nn.Module()
    holder.block = _block(1)
    uvu.apply(holder)
    assert not getattr(holder.block, update_fold.MARKER, False)
    holder = torch.nn.Module()
    holder.block = _block(1)
    monkeypatch.setattr(update_fold, "FOLD_UPDATE", True)
    names = uvu.apply(holder)
    assert "block (update MLP folded)" in names and getattr(holder.block, update_fold.MARKER)


def test_a_changed_forward_is_not_folded(monkeypatch):
    block = _block(1)
    monkeypatch.setitem(update_fold.SOURCE_HASHES, "forward", "0" * 64)
    assert not update_fold.support(block)[0]
    assert update_fold.apply(block) == []


def test_float32_constants_follow_the_model():
    block = _block(2).float()
    update_fold.apply(block)
    assert block._fold_out.dtype == torch.float32 and block._fold_species.dtype == torch.float32
