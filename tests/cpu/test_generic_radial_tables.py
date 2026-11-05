"""``generic_radial_tables``: certified radial tables for the generic family (MACE-MP-0, OMAT-0, the density family).

The same fit, certificate and kernel as MH-1's ``radial_tables``, for a generic interaction's chain ``r -> [h c, c]``
(the split radial network's last layer input) and its edge density; float32 only; "auto" on only on the GPUs of
decision 66 (owner, 2 November 2026).
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


def test_the_record_is_certified_auto_and_on_the_radial_tables_field():
    regs = _registry()
    record = regs.record("generic_radial_tables") if hasattr(regs, "record") else None
    record = record or next(c for c in regs.candidates(_float32("residual")[0]) if c.name == "generic_radial_tables")
    assert (record.exactness, record.default) == ("certified", "auto")
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
    # "auto" (the record's default) does not take a float64 model: off at resolution, nothing raised at apply.
    auto = regs.resolve(model, {"generic_fused_radial_backward": "on"}, strict=False)
    assert not auto.decision("generic_radial_tables").enabled
    assert "float32 models only" in auto.decision("generic_radial_tables").reason


def test_the_planner_keeps_generic_tables_to_measured_gpus_and_float32(monkeypatch):
    from test_planner import GPU, plan_for

    from fastmace.config import PrecisionConfig, RewriteConfig
    from fastmace.errors import PlanRefused
    from fastmace.planner import fixed

    # Decision 66 (owner, 2 November 2026): "auto" on for the generic family on the measured GPUs only.
    assert fixed.DECISION_66_GENERIC_RADIAL_TABLES_GPUS == ("NVIDIA GeForce RTX 5060 Ti",
                                                            "NVIDIA RTX 4000 Ada Generation",
                                                            "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB",
                                                            "NVIDIA H200")
    measured = plan_for("omat0-medium")
    assert measured.rewrites.radial_tables == "on"
    assert "measured GPU, decision 66" in str(measured.decision("RewriteConfig.radial_tables")[2])
    assert plan_for("mp0-medium").rewrites.radial_tables == "on"
    monkeypatch.setattr(fixed, "DECISION_66_GENERIC_RADIAL_TABLES_GPUS", ("NVIDIA A100-SXM4-40GB",))
    other = plan_for("omat0-medium")
    _, choice, why = other.decision("RewriteConfig.radial_tables")
    assert choice == "off" and "decision 66" in str(why) and GPU.name in str(why)
    assert plan_for("omat0-medium", rewrites=RewriteConfig(radial_tables="on")).rewrites.radial_tables == "on"
    monkeypatch.undo()
    # Float64: "auto" keeps the radial chain, an explicit "on" is refused.
    assert plan_for("omat0-medium", precision=PrecisionConfig(dtype="float64")).rewrites.radial_tables == "off"
    with pytest.raises(PlanRefused, match="generic model needs a float32 plan"):
        plan_for("omat0-medium", precision=PrecisionConfig(dtype="float64"),
                 rewrites=RewriteConfig(radial_tables="on"))
    assert plan_for("mh1-r2scan").rewrites.radial_tables == "on"  # MH-1's default (decision 62) unchanged
