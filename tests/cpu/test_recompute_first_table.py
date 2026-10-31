"""``recompute_first_table`` (capacity ideas, rebuild-``a``): the first interaction's radial table rebuilt in the
backward.

On the tiny MH-1-type model (CPU, float32; the tables run through ``radial_table.evaluate``):

- the record is exact, off by default, deterministic, needs ``radial_tables`` and ``reassociate_first``, and is the
  memory rescue's second record, between ``recompute_conv_weights`` and ``recompute_node_segments``;
- its memory term is the table's output, ``[h c, c]``, per edge;
- it runs the first interaction's table and reassociation under one checkpoint, passes its self-check, keeps the
  energy and forces bitwise, and the force call no longer keeps the table's output for the backward.
"""

import copy

import pytest
import torch

from fastmace.rewrites import mh1, radial
from fastmace.rewrites import registry as reg
from fastmace.rewrites.mh1 import modules as mm

pytest.importorskip("mace")

from test_rewrites_mh1 import PACKAGE, context, energy_and_forces, tiny  # noqa: E402


def _registry():
    out = reg.Registry()
    for family in (mh1, radial):
        for record in family.RECORDS.values():
            out.register(record)
    return out


def _rewritten(recompute):
    model, batch = tiny("mh1")
    model = model.float()
    batch = {key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
             for key, value in batch.items()}
    regs = _registry()
    out = copy.deepcopy(model)
    asked = {name: "on" for name in PACKAGE} | {"radial_tables": "on", "recompute_first_table": recompute}
    asked |= {name: "off" for name in regs.names() if name not in asked}
    resolution = regs.resolve(out, asked, strict=False)
    applied = regs.apply(out, resolution, context(batch), original=model)
    return model, out, batch, resolution, applied


def test_the_record_is_exact_off_and_rescued_between_the_other_two():
    from fastmace.model import capabilities
    from fastmace.planner import fixed, memory
    from fastmace.rewrites import memory as terms

    record = radial.RECORDS["recompute_first_table"]
    assert (record.exactness, record.default, record.deterministic) == ("exact", "off", True)
    assert record.requires == ("radial_tables", "reassociate_first")
    assert memory.MEMORY_RESCUE == ("recompute_conv_weights", "recompute_first_table", "recompute_node_segments")
    assert fixed.AUTO_RULES["RewriteConfig.recompute_first_table"] == (
        "on_by_rule_12_when_the_plan_does_not_fit_with_recompute_conv_weights")
    model, _ = tiny("mh1")
    dims = capabilities.model_dims(model)
    edge, node = record.memory(dims, 4)
    assert (edge, node) == (-(dims.interactions[0].radial[-2] + 1) * 4, 0)
    assert record.memory is terms.recompute_first_table


def _saved_shapes(model, batch):
    shapes = []

    def pack(tensor):
        shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        energy_and_forces(model, batch)
    return shapes


def test_the_record_rebuilds_the_table_in_the_backward_and_keeps_every_bit():
    _, without, batch, _, _ = _rewritten("off")
    _, model, _, resolution, applied = _rewritten("on")
    assert resolution.decision("recompute_first_table").enabled
    entry = next(entry for entry in applied if entry.name == "recompute_first_table")
    assert entry.result.passed and entry.result.max_relative_error == 0, entry.result.detail
    first, *others = model.interactions
    assert first.recompute_first_table and not any(getattr(i, "recompute_first_table", False) for i in others)
    assert copy.deepcopy(model).interactions[0].recompute_first_table
    energy, forces = energy_and_forces(without, batch)
    new_energy, new_forces = energy_and_forces(model, batch)
    assert torch.equal(new_energy, energy) and torch.equal(new_forces, forces)
    table = (batch["edge_index"].shape[1], first.radial_table.coef.shape[-1] - 1)  # the table's [E, h c + 1]
    assert table in _saved_shapes(without, batch), "the plan without the record keeps the table's output"
    assert table not in _saved_shapes(model, batch)


def test_the_selfcheck_finds_a_wrong_region(monkeypatch):
    original = mm.first_table_region

    def wrong(self, *args):
        message, density = original(self, *args)
        return message * 1.01, density

    monkeypatch.setattr(mm, "first_table_region", wrong)
    with pytest.raises(reg.SelfCheckError):
        _rewritten("on")
