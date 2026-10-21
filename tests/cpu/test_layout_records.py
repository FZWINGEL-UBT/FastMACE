"""Exact layout removals of the layout audit (plans/opt-3.md OPT3-T05; docs/design/opt3-T05-layout-audit.md).

- ``direct_node_feats``: without LAMMPS, the folded MH-1 forward keeps ``linear_up``'s output and the node
  attributes as they are, instead of joining them for ``handle_lammps`` and slicing them out again (the
  slices were strided, and the streaming convolution copied its input, forward and backward).
- ``folded_constants``: the folded last radial layer's ``[W, b]`` and its two halves for the fused input
  gradient are built once, instead of a concatenation and two copies per force call.
- ``node_major_outer_sum``: the reassociated first interaction's per-slot sums (``edge_outer_sum``) are stored
  node-major, ``[nodes, (l, m), slots, k]``, so that each l's GEMM operand is a view instead of a transposed
  copy.

Both are exact: on the tiny MH-1-type model (CPU, float64) energy and forces are bitwise those without the
record, and the records are opt-in, self-checked and refused where they cannot act.
"""

import copy
import importlib.util

import pytest
import torch

from fastmace.kernels.triton import skinny_linear
from fastmace.rewrites import mh1

pytest.importorskip("mace")
pytestmark = pytest.mark.skipif(importlib.util.find_spec("cuequivariance_torch") is None,
                                reason="cuequivariance_torch is not installed")

from test_rewrites_mh1 import PACKAGE, context, energy_and_forces, registry, tiny  # noqa: E402

NAMES = ("direct_node_feats", "folded_constants", "node_major_outer_sum")
REQUIRES = {"direct_node_feats": ("fold_cutoff",), "folded_constants": ("fold_cutoff",),
            "node_major_outer_sum": ("segmented_first_sum",)}


def _rewritten(**records):
    model, batch = tiny("mh1", backend="cueq")
    regs = registry()
    out = copy.deepcopy(model)
    asked = {name: "on" for name in PACKAGE} | {name: "off" for name in NAMES} | records
    resolution = regs.resolve(out, asked, strict=False)
    applied = regs.apply(out, resolution, context(batch), original=model)
    return model, out, batch, resolution, applied


@pytest.mark.parametrize("name", NAMES)
def test_the_records_are_exact_opt_in_and_need_the_folded_forward(name):
    record = mh1.RECORDS[name]
    assert (record.default, record.exactness, record.deterministic) == ("off", "exact", True)
    assert record.requires == REQUIRES[name]
    model, _ = tiny("mh1", backend="cueq")
    asked = {other: "off" for other in registry().names()} | {name: "on"}
    assert not registry().resolve(copy.deepcopy(model), asked, strict=False).decision(name).enabled
    from fastmace.planner import fixed

    assert fixed.AUTO_RULES[f"RewriteConfig.{name}"] == "never_automatic"


@pytest.mark.parametrize("records", [{"direct_node_feats": "on"}, {"folded_constants": "on"},
                                     {"node_major_outer_sum": "on"},
                                     {"direct_node_feats": "on", "folded_constants": "on",
                                      "node_major_outer_sum": "on"}])
def test_energy_and_forces_are_bitwise_unchanged(records):
    _, without, batch, _, _ = _rewritten()
    _, model, _, resolution, applied = _rewritten(**records)
    for name in records:
        assert resolution.decision(name).enabled
        entry = next(entry for entry in applied if entry.name == name)
        assert entry.result.passed and entry.detail
    energy, forces = energy_and_forces(without, batch)
    new_energy, new_forces = energy_and_forces(model, batch)
    assert torch.equal(new_energy, energy) and torch.equal(new_forces, forces)


def test_direct_node_feats_marks_the_folded_interactions():
    _, model, _, _, _ = _rewritten(direct_node_feats="on")
    marked = [interaction for interaction in model.interactions if getattr(interaction, "direct_node_feats", False)]
    assert marked and all(getattr(i.forward, "__func__", None) is mh1.modules.cutoff_folded_forward for i in marked)


def test_direct_node_feats_keeps_lammps_runs_on_the_join():
    """With a LAMMPS class the forward still joins for handle_lammps (the record only skips an identity)."""
    import inspect

    source = inspect.getsource(mh1.modules.cutoff_folded_forward)
    assert "lammps_class is None" in source and "direct_node_feats" in source


def test_folded_constants_hold_the_folded_weight_and_its_halves():
    _, model, _, _, _ = _rewritten(folded_constants="on")
    nets = [i.conv_tp_weights for i in model.interactions if getattr(i.conv_tp_weights, "folded_constant", None)
            is not None]
    assert nets
    for net in nets:
        last = net.rest[-1]
        expected = torch.cat([last.weight, last.bias[:, None]], dim=1)
        assert torch.equal(net.folded_weight(), expected)
        main, tail = net.folded_parts
        assert main.is_contiguous() and tail.is_contiguous()
        assert torch.equal(main, expected[:, :-1]) and torch.equal(tail, expected[:, -1])


def test_the_folded_input_gradient_with_parts_equals_the_one_without():
    a = torch.randn(37, 65, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(1280, 65, dtype=torch.float64)
    parts = (weight[:, :64].contiguous(), weight[:, 64].contiguous())
    grads = []
    for kwargs in ({}, {"parts": parts}):
        out = skinny_linear.folded_linear(a, weight, **kwargs)
        (grad,) = torch.autograd.grad((out * out).sum(), a)
        grads.append((out.detach(), grad))
    assert torch.equal(grads[0][0], grads[1][0]) and torch.equal(grads[0][1], grads[1][1])


@pytest.mark.parametrize("segmented", [False, True])
def test_node_major_sums_are_the_slot_major_sums_rearranged_bitwise(segmented):
    from fastmace.kernels.triton.edge_outer_sum import edge_outer_sum

    generator = torch.Generator().manual_seed(9)
    nodes, slots, d, k, edges = 5, 3, 4, 6, 40
    a = torch.randn((edges, k), generator=generator, dtype=torch.float64).requires_grad_()
    y = torch.randn((edges, d), generator=generator, dtype=torch.float64).requires_grad_()
    key = torch.sort(torch.randint(0, nodes * slots, (edges,), generator=generator)).values
    old = edge_outer_sum(a, y, key, nodes * slots, segmented=segmented)
    new = edge_outer_sum(a, y, key, nodes * slots, segmented=segmented, node_slots=slots)
    assert new.shape == (nodes, d * slots * k)
    rearranged = old.view(nodes, slots, d, k).permute(0, 2, 1, 3).reshape(nodes, d * slots * k)
    assert torch.equal(new, rearranged)
    grad = torch.randn(new.shape, generator=generator, dtype=torch.float64)
    old_grad = grad.view(nodes, d, slots, k).permute(0, 2, 1, 3).reshape(nodes * slots, d * k)
    got = torch.autograd.grad((new * grad).sum(), (a, y))
    want = torch.autograd.grad((old * old_grad).sum(), (a, y))
    assert all(torch.equal(g, w) for g, w in zip(got, want))


def test_node_major_outer_sum_marks_the_reassociation():
    _, model, _, _, _ = _rewritten(node_major_outer_sum="on")
    assert model.interactions[0].first_reassociation.node_major
    _, without, _, _, _ = _rewritten()
    assert not without.interactions[0].first_reassociation.node_major
