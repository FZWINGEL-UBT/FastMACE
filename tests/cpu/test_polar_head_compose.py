"""``head_compose`` (P1): PolarMACE's head projections composed into their l x l -> 0 dot products at setup.

The composed field update (with and without FOLD_UPDATE) and the composed electron readout must match MACE's
forward and its gradients (node features, potentials, charges) in float64, on blocks built like MACE-POLAR-1
(hidden degrees 0..lmax, a duplicated potential irreps list as ``field_irreps * 2``). The checkpoints: the gate.
"""

from __future__ import annotations

import copy

import pytest
import torch

field_blocks = pytest.importorskip("mace.modules.field_blocks")
from e3nn import o3  # noqa: E402

from fastmace.rewrites.polar import head_compose, update_fold, uvu  # noqa: E402

POTENTIAL = o3.Irreps("2x0e+2x1o+2x0e+2x1o")  # field_irreps * 2 (two spins), as PolarMACE builds it
CHARGES = o3.Irreps("1x0e+1x1o+1x0e+1x1o")


def _node_irreps(lmax, channels):
    return o3.Irreps("+".join(f"{channels}x{degree}{'eo'[degree % 2]}" for degree in range(lmax + 1)))


def _randomise(block, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.add_(0.3 * torch.randn_like(parameter))
    return block.double()


def _update(lmax, channels=16, species=5, seed=0):
    node = _node_irreps(lmax, channels)
    block = field_blocks.AgnosticEmbeddedOneBodyVariableUpdate(
        node_attrs_irreps=o3.Irreps(f"{species}x0e"), node_feats_irreps=node, edge_attrs_irreps=o3.Irreps("1x0e+1x1o"),
        edge_feats_irreps=o3.Irreps("8x0e"), target_irreps=node, hidden_irreps=node, avg_num_neighbors=10.0,
        potential_irreps=POTENTIAL, charges_irreps=CHARGES, field_norm_factor=1.0)
    return _randomise(block, seed)


def _readout(lmax, channels=16, species=5, seed=0):
    node = _node_irreps(lmax, channels)
    block = field_blocks.OneBodyMLPFieldReadout(
        node_attrs_irreps=o3.Irreps(f"{species}x0e"), node_feats_irreps=node, edge_attrs_irreps=o3.Irreps("1x0e+1x1o"),
        edge_feats_irreps=o3.Irreps("8x0e"), target_irreps=node, hidden_irreps=node, avg_num_neighbors=10.0,
        potential_irreps=POTENTIAL, charges_irreps=CHARGES)
    return _randomise(block, seed)


def _inputs(block, atoms=11, seed=1):
    generator = torch.Generator().manual_seed(seed)
    species = block.node_attrs_irreps.dim

    def rnd(dim):
        return torch.randn(atoms, dim, generator=generator, dtype=torch.float64)

    node_attrs = torch.nn.functional.one_hot(torch.randint(species, (atoms,), generator=generator), species).double()
    return (node_attrs, rnd(block.node_feats_irreps.dim), rnd(block.potential_irreps.dim),
            rnd(block.charges_irreps.dim), rnd(block.charges_irreps.dim))


def _run(block, node_attrs, node_feats, potential, charges, charges_0):
    leaves = [t.clone().requires_grad_() for t in (node_feats, potential, charges)]
    if isinstance(block, field_blocks.OneBodyMLPFieldReadout):
        out = block(node_attrs, leaves[0], None, None, None, leaves[1], charges_0, leaves[2])
    else:
        out = block(node_attrs, leaves[0], None, None, None, leaves[1], leaves[2])
    cotangent = torch.linspace(-1, 1, out.numel(), dtype=out.dtype).view_as(out)
    return [out.detach(), *torch.autograd.grad(out, leaves, cotangent)]


def _check(block, composed):
    inputs = _inputs(block)
    for expected, got in zip(_run(block, *inputs), _run(composed, *inputs)):
        torch.testing.assert_close(got, expected, rtol=1e-11, atol=1e-12)


def test_the_constant_is_off():
    assert head_compose.COMPOSE_HEADS is False


@pytest.mark.parametrize("fold", [False, True])
@pytest.mark.parametrize("lmax", [0, 1, 3])
def test_the_composed_update_matches_mace(lmax, fold):
    block = _update(lmax)
    assert head_compose.support(block) == (True, "ok")
    composed = copy.deepcopy(block)
    uvu.apply(composed)
    if fold:
        update_fold.apply(composed)
    assert head_compose.apply(composed) == [""]
    assert head_compose.apply(composed) == []  # once
    _check(block, composed)


@pytest.mark.parametrize("lmax", [0, 1, 3])
def test_the_composed_readout_matches_mace(lmax):
    block = _readout(lmax)
    assert head_compose.support(block) == (True, "ok")
    composed = copy.deepcopy(block)
    uvu.apply(composed)
    assert head_compose.apply(composed) == [""]
    _check(block, composed)


def test_a_composed_block_survives_deep_copy_and_dtype_moves():
    block = _update(1)
    composed = copy.deepcopy(block)
    uvu.apply(composed)
    head_compose.apply(composed)
    _check(block, copy.deepcopy(composed))
    single = copy.deepcopy(composed).float()
    assert single.dot_products._head_0_square.dtype == torch.float32


def test_uvu_apply_composes_only_with_the_constant(monkeypatch):
    holder = torch.nn.Module()
    holder.update, holder.readout = _update(1), _readout(1)
    uvu.apply(holder)
    assert not getattr(holder.update, head_compose.MARKER, False)
    holder = torch.nn.Module()
    holder.update, holder.readout = _update(1), _readout(1)
    monkeypatch.setattr(head_compose, "COMPOSE_HEADS", True)
    names = uvu.apply(holder)
    assert "update (head composed)" in names and "readout (head composed)" in names
