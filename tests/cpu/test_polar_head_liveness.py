"""``head_liveness`` (R5): dead POLAR head degrees, proved from the e3nn instructions, pruned at setup.

A (update, with FOLD_UPDATE): ``tp_out`` paths writing degrees the readout's ``linear_1`` never reads leave the
fold, while the degree-two invariant inputs stay live. B (electron readout): dot-product paths reading the
structurally zero degree-two expansion of charges and potentials go, bitwise exactly. Blocks built like
MACE-POLAR-1 (``test_polar_head_compose``), in both layouts; float64 against MACE.
"""

from __future__ import annotations

import copy

import pytest
import torch

pytest.importorskip("mace.modules.field_blocks")
import test_polar_head_compose as compose_test  # noqa: E402

from fastmace.rewrites.polar import head_compose, head_liveness, update_fold, uvu  # noqa: E402

LAYOUTS = ["mul_ir", "ir_mul"]


def _check_close(block, rewritten, rtol=1e-11):
    inputs = compose_test._inputs(block)
    expected, got = compose_test._run(block, *inputs), compose_test._run(rewritten, *inputs)
    for e, g in zip(expected, got):
        torch.testing.assert_close(g, e, rtol=rtol, atol=1e-12)
    return got


def test_the_constant_is_off():
    assert head_liveness.PRUNE_DEAD_DEGREES is False


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("lmax", [1, 2, 3])
def test_dead_update_outputs_leave_the_fold(lmax, layout):
    block = compose_test._update(lmax, layout=layout)
    tp = block.tp_out
    dead = head_liveness.dead_output_paths(block)
    degrees = [tp.irreps_out[head_liveness._entry(tp.irreps_out, meta[4])][1].l for meta in tp._path_meta]
    assert sorted(dead) == [k for k, degree in enumerate(degrees) if degree >= 2]  # the readout reads 0e and 1o
    pruned = copy.deepcopy(block)
    uvu.apply(pruned)
    assert head_liveness.apply(pruned, fold=True) == ([""] if dead else [])
    assert update_fold.apply(pruned) == [""]
    live_rows = sum(meta[10] for k, meta in enumerate(tp._path_meta) if k not in dead)
    assert pruned._fold_out.shape[0] == live_rows
    _, grad_feats, _, _ = _check_close(block, pruned)
    if lmax >= 2:  # the degree-two inputs of the invariants stay live
        start, stop = (s for s in block.node_feats_irreps.slices()[2].indices(block.node_feats_irreps.dim)[:2])
        assert grad_feats[:, start:stop].abs().max() > 0


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("lmax", [1, 2, 3])
def test_zero_readout_paths_go_bitwise(lmax, layout):
    block = compose_test._readout(lmax, layout=layout)
    reference = copy.deepcopy(block)
    uvu.apply(reference)
    pruned = copy.deepcopy(block)
    uvu.apply(pruned)
    dead_q = head_liveness.dead_input_paths(block.linear_up_q, block.dot_products_q)
    assert len(dead_q) == max(0, lmax - 1)  # one l x l -> 0 path per degree above one
    assert head_liveness.apply(pruned) == ([""] if lmax >= 2 else [])
    inputs = compose_test._inputs(block)
    for expected, got in zip(compose_test._run(reference, *inputs), compose_test._run(pruned, *inputs)):
        assert torch.equal(got, expected)
    _check_close(block, pruned)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_pruning_composes_with_fold_and_heads(layout):
    holder = torch.nn.Module()
    holder.update, holder.readout = compose_test._update(3, layout=layout), compose_test._readout(3, layout=layout)
    rewritten = copy.deepcopy(holder)
    uvu.apply(rewritten)
    head_liveness.apply(rewritten, fold=True)
    update_fold.apply(rewritten)
    head_compose.apply(rewritten)
    _check_close(holder.update, rewritten.update)
    _check_close(holder.readout, rewritten.readout)


def test_uvu_apply_prunes_only_with_the_constant_and_keeps_the_selfcheck(monkeypatch):
    holder = torch.nn.Module()
    holder.update, holder.readout = compose_test._update(2), compose_test._readout(2)
    plain = copy.deepcopy(holder)
    uvu.apply(plain)
    assert not getattr(plain.readout, head_liveness.MARKER, False)
    monkeypatch.setattr(head_liveness, "PRUNE_DEAD_DEGREES", True)
    monkeypatch.setattr(update_fold, "FOLD_UPDATE", True)
    pruned = copy.deepcopy(holder)
    names = uvu.apply(pruned)
    assert "update (dead degrees pruned)" in names and "readout (dead degrees pruned)" in names
    assert uvu.selfcheck(holder, pruned)["passed"]  # the tensor products stay whole
    _check_close(holder.update, pruned.update)


def test_an_unfolded_update_is_left_whole():
    block = compose_test._update(2)
    assert head_liveness.apply(block, fold=False) == []
    assert not hasattr(block, "_live_out_meta")


def test_a_replaced_readout_linear_proves_nothing():
    block = compose_test._update(2)
    block.readout.linear_1 = torch.nn.Sequential(block.readout.linear_1)  # no instructions to read
    assert head_liveness.dead_output_paths(block) == []
