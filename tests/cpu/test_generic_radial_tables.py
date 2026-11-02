"""``generic_radial_tables``: certified radial tables for the generic family (MACE-MP-0, OMAT-0, the density family).

The same fit, certificate and kernel as MH-1's ``radial_tables``, for a generic interaction's chain ``r -> [h c, c]``
(the split radial network's last layer input) and its edge density; float32 only, opt-in.
"""

from __future__ import annotations

import copy

import pytest
import torch

from fastmace.rewrites import generic, radial
from fastmace.rewrites import registry as reg

pytest.importorskip("mace")

from test_rewrites_mh1 import context, energy_and_forces, tiny  # noqa: E402

FAMILIES = ("residual", "density")


def _registry():
    out = reg.Registry()
    for family in (generic, radial):
        for record in family.RECORDS.values():
            out.register(record)
    return out


def _float32(family):
    model, batch = tiny(family)
    model = model.float()
    batch = {key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
             for key, value in batch.items()}
    return model, batch


def _rewritten(family, tables, model=None, batch=None):
    if model is None:
        model, batch = _float32(family)
    regs = _registry()
    out = copy.deepcopy(model)
    asked = {"generic_fused_radial_backward": "on", "generic_radial_tables": "on" if tables else "off"}
    asked |= {name: "off" for name in regs.names() if name not in asked}
    resolution = regs.resolve(out, asked, strict=False)
    applied = regs.apply(out, resolution, context(batch), original=model)
    return model, out, batch, resolution, applied


def test_the_record_is_certified_opt_in_and_on_the_radial_tables_field():
    regs = _registry()
    record = regs.record("generic_radial_tables") if hasattr(regs, "record") else None
    record = record or next(c for c in regs.candidates(_float32("residual")[0]) if c.name == "generic_radial_tables")
    assert (record.exactness, record.default) == ("certified", "off")
    assert radial.RECORDS["generic_radial_tables"]["deterministic"] is True
    assert record.requires == ("generic_fused_radial_backward",)
    assert regs.config_field("generic_radial_tables") == "radial_tables"


@pytest.mark.parametrize("family", FAMILIES)
def test_generic_tables_are_certified_and_the_forces_follow_the_network(family):
    _, without, batch, _, _ = _rewritten(family, False)
    _, model, _, resolution, applied = _rewritten(family, True)
    assert resolution.decision("generic_radial_tables").enabled
    entry = next(entry for entry in applied if entry.name == "generic_radial_tables")
    assert entry.result.passed, entry.result.detail
    tables = [interaction.radial_table for interaction in model.interactions]
    assert all(0.0 <= table.certificate["worst"] <= 1.0 for table in tables), [t.certificate for t in tables]
    assert all(table.single for table in tables)  # no distance transform: one table per interaction
    assert type(model.radial_embedding).__name__ == "LengthsEmbedding"
    energy, forces = energy_and_forces(without, batch)
    new_energy, new_forces = energy_and_forces(model, batch)
    assert float((new_forces - forces).abs().max() / forces.abs().max()) < 1e-4
    assert float((new_energy - energy).abs().max() / energy.abs().max()) < 1e-5
    # Against float64 (the float32 weights in float64): no further away than the network, within the rest's noise.
    reference = copy.deepcopy(without).double()
    batch64 = {key: value.double() if torch.is_tensor(value) and value.is_floating_point() else value
               for key, value in batch.items()}
    _, forces64 = energy_and_forces(reference, batch64)
    network_error = float((forces.double() - forces64).abs().max())
    table_error = float((new_forces.double() - forces64).abs().max())
    assert table_error <= 2 * network_error + 1e-7


def test_float64_models_are_refused():
    model, batch = tiny("residual")  # float64
    regs = _registry()
    out = copy.deepcopy(model)
    asked = {"generic_fused_radial_backward": "on"}
    asked |= {name: "off" for name in regs.names() if name not in asked}
    regs.apply(out, regs.resolve(out, asked, strict=False), context(batch), original=model)
    with pytest.raises(reg.RewriteError, match="float32 models only"):
        radial.tabulate_generic(out, context(batch))


def test_the_planner_keeps_generic_tables_opt_in_and_float32():
    from test_planner import plan_for

    from fastmace.config import PrecisionConfig, RewriteConfig
    from fastmace.errors import PlanRefused

    assert plan_for("omat0-medium").rewrites.radial_tables == "off"
    assert plan_for("omat0-medium", rewrites=RewriteConfig(radial_tables="on")).rewrites.radial_tables == "on"
    with pytest.raises(PlanRefused, match="generic model needs a float32 plan"):
        plan_for("omat0-medium", precision=PrecisionConfig(dtype="float64"),
                 rewrites=RewriteConfig(radial_tables="on"))
    assert plan_for("mh1-r2scan").rewrites.radial_tables == "on"  # MH-1's default (decision 62) unchanged
