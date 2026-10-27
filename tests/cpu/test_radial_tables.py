"""``radial_tables`` (opt-6 OPT6-T05): certified Chebyshev tables for the MH-1 radial chain, on the tiny MH-1-type
model (CPU, float32; the tables run through ``radial_table.evaluate``, the kernels' PyTorch twin).

The record fits per-pair tables in float64, certifies that their float32 error is at most the float32 network's
(values and r-derivatives, at the tensor-product weights and the edge density), and is self-checked; the planner
keeps it opt-in and refuses it off float32.
"""

import copy

import pytest
import torch

from fastmace.rewrites import mh1, radial
from fastmace.rewrites import registry as reg

pytest.importorskip("mace")

from test_rewrites_mh1 import PACKAGE, context, energy_and_forces, tiny  # noqa: E402


def _registry():
    out = reg.Registry()
    for family in (mh1, radial):
        for record in family.RECORDS.values():
            out.register(record)
    return out


def _float32():
    model, batch = tiny("mh1")
    model = model.float()
    batch = {key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
             for key, value in batch.items()}
    return model, batch


def _rewritten(tables):
    model, batch = _float32()
    regs = _registry()
    out = copy.deepcopy(model)
    asked = {name: "on" for name in PACKAGE} | {"radial_tables": "on" if tables else "off"}
    asked |= {name: "off" for name in regs.names() if name not in asked}
    resolution = regs.resolve(out, asked, strict=False)
    applied = regs.apply(out, resolution, context(batch), original=model)
    return model, out, batch, resolution, applied


def test_the_record_is_certified_opt_in_and_deterministic():
    record = radial.RECORDS["radial_tables"]
    assert (record.exactness, record.default, record.deterministic) == ("certified", "auto", True)
    assert record.requires == ("fold_cutoff",)
    assert "certified" in reg.EXACTNESS
    from fastmace.planner import fixed

    assert fixed.AUTO_RULES["RewriteConfig.radial_tables"] == "on_with_fold_cutoff_float32_unless_reproducible"


def test_tables_are_certified_and_the_forces_follow_the_network():
    _, without, batch, _, _ = _rewritten(False)
    original, model, _, resolution, applied = _rewritten(True)
    assert resolution.decision("radial_tables").enabled
    entry = next(entry for entry in applied if entry.name == "radial_tables")
    assert entry.result.passed, entry.result.detail
    tables = [interaction.radial_table for interaction in model.interactions]
    assert all(0.0 <= table.certificate["worst"] <= 1.0 for table in tables), [t.certificate for t in tables]
    assert type(model.radial_embedding).__name__ == "LengthsEmbedding"
    energy, forces = energy_and_forces(without, batch)
    new_energy, new_forces = energy_and_forces(model, batch)
    assert float((new_forces - forces).abs().max() / forces.abs().max()) < 1e-4
    assert float((new_energy - energy).abs().max() / energy.abs().max()) < 1e-5
    # Against float64 (the pristine weights' float32 values in float64): the table model is no further away
    # than the network model, within the float32 noise of the rest of the model.
    reference = copy.deepcopy(without).double()
    batch64 = {key: value.double() if torch.is_tensor(value) and value.is_floating_point() else value
               for key, value in batch.items()}
    _, forces64 = energy_and_forces(reference, batch64)
    network_error = float((forces.double() - forces64).abs().max())
    table_error = float((new_forces.double() - forces64).abs().max())
    assert table_error <= 2 * network_error + 1e-7


def test_edges_beyond_the_cutoff_give_zero_values_and_gradients():
    _, model, _, _, _ = _rewritten(True)
    table = model.interactions[0].radial_table
    r = torch.tensor([[table.r_max], [table.r_max + 1.0], [0.5 * (table.r_lo + table.r_max)]], requires_grad=True)
    node_attrs = torch.nn.functional.one_hot(table.columns[:1].repeat(2), len(model.atomic_numbers)).float()
    edge_index = torch.tensor([[0, 0, 0], [1, 1, 1]])
    folded, density = table(r, node_attrs, edge_index)
    assert torch.equal(folded[:2], torch.zeros_like(folded[:2])) and torch.equal(density[:2], torch.zeros(2, 1))
    (grad,) = torch.autograd.grad((folded.sum() + density.sum()), r)
    assert torch.equal(grad[:2], torch.zeros(2, 1)) and bool(grad[2] != 0)


def test_the_table_gradient_is_the_derivative_of_its_values():
    _, model, _, _, _ = _rewritten(True)
    table = model.interactions[1].radial_table
    r = torch.linspace(1.2, table.r_max - 0.2, 64, dtype=torch.float32)[:, None].requires_grad_()
    node_attrs = torch.nn.functional.one_hot(table.columns[[0, -1]], len(model.atomic_numbers)).float()
    edge_index = torch.stack([torch.zeros(64, dtype=torch.long), torch.ones(64, dtype=torch.long)])
    folded, density = table(r, node_attrs, edge_index)
    (grad,) = torch.autograd.grad(folded.sum() + density.sum(), r)
    h = 1e-3
    up, up_density = table((r + h).detach(), node_attrs, edge_index)
    down, down_density = table((r - h).detach(), node_attrs, edge_index)
    numeric = ((up.sum(1) + up_density.sum(1)) - (down.sum(1) + down_density.sum(1))) / (2 * h)
    torch.testing.assert_close(grad.reshape(-1), numeric, rtol=2e-2, atol=2e-3)


def test_float64_models_are_refused():
    model, batch = tiny("mh1")
    regs = _registry()
    asked = {name: "on" for name in PACKAGE} | {"radial_tables": "on"}
    asked |= {name: "off" for name in regs.names() if name not in asked}
    resolution = regs.resolve(model, asked, strict=False)
    # Decision 62: the record's support check declines float64 models, so even "on" resolves off with the reason
    # (and "auto", the default, skips them); applying the resolution leaves the radial chain alone.
    decision = resolution.decision("radial_tables")
    assert not decision.enabled and "float32 models only" in decision.reason
    applied = regs.apply(copy.deepcopy(model), resolution, context(batch), original=model)
    assert all(entry.name != "radial_tables" for entry in applied)


def test_the_planner_refuses_radial_tables_in_float64():
    from test_planner import plan_for

    from fastmace.config import PrecisionConfig, RewriteConfig
    from fastmace.planner.fixed import PlanRefused

    with pytest.raises(PlanRefused, match="radial_tables='on' needs float32"):
        plan_for("mh1-r2scan", precision=PrecisionConfig(dtype="float64"), rewrites=RewriteConfig(radial_tables="on"))
    # Owner decision 62: on by default in float32 plans with fold_cutoff, off in float64 ones.
    assert plan_for("mh1-r2scan").rewrites.radial_tables == "on"
    assert plan_for("mh1-r2scan", precision=PrecisionConfig(dtype="float64")).rewrites.radial_tables == "off"
    assert plan_for("mh1-r2scan", rewrites=RewriteConfig(radial_tables="off")).rewrites.radial_tables == "off"
