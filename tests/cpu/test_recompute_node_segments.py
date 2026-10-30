"""``recompute_node_segments`` (plans/capacity.md CAP-T08): the node-local segments recomputed in the backward.

On the CPU in float64:

- ``modules.recomputed_forward`` against the stored forward of a node-local module, values and input gradients,
  with arguments passed by keyword and a ``None``; it keeps fewer tensors for the backward;
- a deep copy rebinds the recomputed forward and the stored one to the copy;
- the record on a tiny MH-1-class cuEquivariance model (naive kernels), with one region per interaction (the default)
  and with separate segments: it recomputes every ``gated_rows`` chain and product block, survives a deep copy,
  passes its self-check (and fails it for a wrong segment), and keeps the model's energy and forces bitwise in eager.
"""

import copy
import importlib.util

import pytest
import torch

from fastmace.rewrites import fusion, linear, mh1, segments
from fastmace.rewrites import registry as reg
from fastmace.rewrites.segments import modules as sm

pytest.importorskip("mace")
CUEQ = pytest.mark.skipif(importlib.util.find_spec("cuequivariance_torch") is None,
                          reason="cuequivariance_torch is not installed")

from test_rewrites_mh1 import PACKAGE, context, energy_and_forces, tiny  # noqa: E402

LINEAR = ("block_linear", "grouped_linear", "merged_residual")


class NodeLocal(torch.nn.Module):
    """A node-local stand-in: per-atom rows in, per-atom rows out, a skip term and a shared table."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 5, dtype=torch.float64)

    def forward(self, x, sc=None, table=None):
        out = torch.nn.functional.silu(self.linear(x)) * table.sum()
        return out if sc is None else out + sc


def _inputs(generator, rows=11):
    x = torch.randn((rows, 6), generator=generator, dtype=torch.float64).requires_grad_()
    sc = torch.randn((rows, 5), generator=generator, dtype=torch.float64).requires_grad_()
    table = torch.randn((3,), generator=generator, dtype=torch.float64).requires_grad_()
    return x, sc, table


def _saved(function, *args, **kwargs):
    count = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: count.append(t) or t, lambda t: t):
        out = function(*args, **kwargs)
    return out, len(count)


def test_the_recomputed_forward_equals_the_stored_one():
    generator = torch.Generator().manual_seed(4)
    module = NodeLocal()
    reference = copy.deepcopy(module)
    sm.recompute(module)
    x, sc, table = _inputs(generator)
    grad = torch.randn((11, 5), generator=generator, dtype=torch.float64)
    results = []
    for function in (reference, module):
        out = function(x, sc=sc, table=table)
        results.append((out.detach(), *torch.autograd.grad(out, (x, sc, table), grad)))
    for a, b in zip(*results):
        assert torch.equal(a, b)
    assert torch.equal(module(x, None, table=table), reference(x, None, table=table))  # positional, a None
    _, kept = _saved(module, x, sc=sc, table=table)
    _, stored = _saved(reference, x, sc=sc, table=table)
    assert kept < stored


def test_a_deep_copy_rebinds_the_recomputed_forward():
    module = sm.recompute(NodeLocal())
    twin = copy.deepcopy(module)
    assert twin.forward.__self__ is twin and twin.recomputed_inner.__self__ is twin
    assert sm.stored(twin).__self__ is twin


def test_the_record_is_rounding_deterministic_off_and_rescued_after_recompute_conv_weights():
    from fastmace.model import capabilities
    from fastmace.planner import memory
    from fastmace.rewrites import memory as terms

    record = segments.RECORDS["recompute_node_segments"]
    assert (record.exactness, record.default, record.deterministic) == ("rounding", "off", True)
    assert record.requires == ("gated_rows",)
    assert memory.MEMORY_RESCUE == ("recompute_conv_weights", "recompute_node_segments")
    assert "fastmace.rewrites.segments" in reg.RECORD_SOURCES
    model, _ = tiny("mh1")
    edge, node = record.memory(capabilities.model_dims(model), 4)
    assert edge == 0 and node < 0
    assert record.memory is terms.recompute_node_segments


# --------------------------------------------------------------------------- the tiny MH-1 model

def _registry():
    out = reg.Registry()
    for family in (mh1, linear, fusion, segments):
        for record in family.RECORDS.values():
            out.register(record)
    return out


def _rewritten(recompute):
    model, batch = tiny("mh1", backend="cueq")
    regs = _registry()
    on = {*PACKAGE, *LINEAR, "gated_rows"}
    requests = ({name: "on" if name in on else "off" for name in regs.names()}
                | {"recompute_node_segments": recompute})
    out = copy.deepcopy(model)
    resolution = regs.resolve(out, requests, strict=False)
    applied = regs.apply(out, resolution, context(batch), original=model)
    return model, out, batch, resolution, applied


REGIONS = pytest.mark.parametrize("region", ["combined", "separate"])


@CUEQ
@REGIONS
def test_the_record_recomputes_every_segment_and_passes_its_selfcheck(monkeypatch, region):
    monkeypatch.setattr(segments, "SEGMENT_REGION", region)
    original, model, _, resolution, applied = _rewritten("on")
    assert resolution.decision("recompute_node_segments").enabled
    entry = next(entry for entry in applied if entry.name == "recompute_node_segments")
    assert entry.result.passed and entry.result.max_relative_error == 0
    if region == "combined":  # one region per interaction: the chain and the reshape run inside the product's
        for interaction, product in zip(model.interactions, model.products):
            assert getattr(product, "region_widths", None) is not None
            assert interaction.gated_rows.forward.__func__ is sm.joined_inputs
            assert interaction.reshape.forward.__func__ is sm.passed_through
            assert product.forward.__func__ is sm.region_forward
    else:
        assert all(module.forward.__func__ is sm.recomputed_forward for _, module in segments.segments(model))
    twin = copy.deepcopy(model)  # the stored forwards rebind to the copy
    for module in (twin.products[1], twin.interactions[1].gated_rows):
        for name in ("recomputed_inner", "region_chain", "region_reshape"):
            bound = getattr(module, name, None)
            assert bound is None or bound.__self__ in set(twin.modules())
    product = model.products[1]
    wrapped = product.forward
    product.forward = lambda *args, **kwargs: wrapped(*args, **kwargs).roll(1, 0)  # rows mixed
    assert not segments.RECORDS["recompute_node_segments"].selfcheck(original, model, reg.Context()).passed


@CUEQ
@REGIONS
def test_energy_and_forces_are_bitwise_unchanged_in_eager(monkeypatch, region):
    monkeypatch.setattr(segments, "SEGMENT_REGION", region)
    _, without, batch, _, _ = _rewritten("off")
    _, recomputed, _, _, _ = _rewritten("on")
    energy_and_forces(without, batch)  # throwaway evaluation (docs/development.md)
    energy, forces = energy_and_forces(without, batch)
    new_energy, new_forces = energy_and_forces(recomputed, batch)
    assert torch.equal(new_energy, energy) and torch.equal(new_forces, forces)
