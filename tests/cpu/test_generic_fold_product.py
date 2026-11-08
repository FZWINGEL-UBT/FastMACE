"""generic_fold_product_linear: products.0.linear folded into its consumers on the generic family (4 November 2026).

The tiny density model (MP-0b3/MPA-0/OMAT-0's classes, cuEquivariance-converted on the CPU) with block_linear,
generic_species_skip and the fold against the stock model: energy, forces and stress in float64, the self-check, and
which models the record and the planner's support key take.
"""

import copy

import pytest
import torch
from test_rewrites_linear import energy_and_forces, relative, tiny_cueq

from fastmace.model import capabilities
from fastmace.rewrites import registry as reg
from fastmace.rewrites.generic.modules import SpeciesSkip
from fastmace.rewrites.linear.modules import BlockLinear

ON = ("generic_species_skip", "block_linear", "generic_fold_product_linear")


def _rewrite(model, batch, names=ON):
    registry = reg.default_registry()
    rewritten = copy.deepcopy(model)
    requested = {name: ("on" if name in names else "off") for name in registry.names()}
    resolution = registry.resolve(rewritten, requested, strict=False)
    context = reg.Context(inputs=reg.structure_inputs(batch), seed=0)
    applied = registry.apply(rewritten, resolution, context, original=model)
    return rewritten, applied


def _stress(model, batch):
    out = model(batch.clone().to_dict(), training=False, compute_force=True, compute_stress=True)
    return out["stress"].detach()


def test_fold_matches_the_stock_density_model():
    model, batch = tiny_cueq("density")
    rewritten, applied = _rewrite(model, batch)
    assert [entry.name for entry in applied] == list(ON)
    for entry in applied:
        assert entry.result.passed and entry.result.max_relative_error <= 1e-10, entry
    assert isinstance(rewritten.products[0].linear, torch.nn.Identity)
    second = rewritten.interactions[1]
    assert isinstance(second.skip_tp, SpeciesSkip) and len(second.skip_tp._reference) == 2
    assert isinstance(second.linear_up, BlockLinear) and len(second.linear_up._reference) == 2
    assert isinstance(rewritten.readouts[0].linear, BlockLinear)
    first = rewritten.interactions[0]  # FOLD_FIRST_DENSITY_LINEAR (round 6): linear composed into the species skip
    assert isinstance(first.linear, torch.nn.Identity)
    assert isinstance(first.skip_tp, SpeciesSkip) and len(first.skip_tp._reference) == 2
    energy_and_forces(model, batch)  # throwaway evaluation (docs/development.md)
    energy, forces = energy_and_forces(model, batch)
    new_energy, new_forces = energy_and_forces(rewritten, batch)
    assert relative(new_energy, energy) <= 1e-10 and relative(new_forces, forces) <= 1e-10
    assert relative(_stress(rewritten, batch), _stress(model, batch)) <= 1e-10


def test_first_density_linear_fold_can_be_switched_off(monkeypatch):
    from fastmace.rewrites import species

    monkeypatch.setattr(species, "FOLD_FIRST_DENSITY_LINEAR", False)
    model, batch = tiny_cueq("density")
    rewritten, applied = _rewrite(model, batch)
    assert all(entry.result.passed for entry in applied)
    assert isinstance(rewritten.interactions[0].linear, BlockLinear)
    assert isinstance(rewritten.products[0].linear, torch.nn.Identity)
    energy, forces = energy_and_forces(model, batch)
    new_energy, new_forces = energy_and_forces(rewritten, batch)
    assert relative(new_energy, energy) <= 1e-10 and relative(new_forces, forces) <= 1e-10



def test_readout_tail_fold_matches_the_stock_density_model(monkeypatch):
    from fastmace.rewrites import species

    assert species.GENERIC_FOLD_READOUT_TAIL is None and not species.GENERIC_FOLD_READOUT_TAIL_GPUS
    monkeypatch.setattr(species, "GENERIC_FOLD_READOUT_TAIL", True)
    model, batch = tiny_cueq("density")
    rewritten, applied = _rewrite(model, batch)
    for entry in applied:
        assert entry.result.passed and entry.result.max_relative_error <= 1e-10, entry
    attribute = "linear_1" if hasattr(rewritten.readouts[-1], "linear_1") else "linear"
    assert isinstance(getattr(rewritten.readouts[-1], attribute), torch.nn.Identity)
    last, product = rewritten.interactions[-1], rewritten.products[-1].linear
    assert isinstance(product, BlockLinear) and len(product._reference) == 2
    layer = product._reference[1]  # the readout's layer: the products now end at its width
    assert tuple(product.segments_out) == tuple(layer.segments_out)
    assert isinstance(last.skip_tp, SpeciesSkip) and last.skip_tp._reference[1] is layer
    assert sum(dim for _, _, dim in last.skip_tp.segments_out) == sum(m * d for _, m, d in layer.segments_out)
    energy_and_forces(model, batch)  # throwaway evaluation (docs/development.md)
    energy, forces = energy_and_forces(model, batch)
    new_energy, new_forces = energy_and_forces(rewritten, batch)
    assert relative(new_energy, energy) <= 1e-10 and relative(new_forces, forces) <= 1e-10
    assert relative(_stress(rewritten, batch), _stress(model, batch)) <= 1e-10


def test_readout_tail_fold_is_off_by_default():
    model, batch = tiny_cueq("density")
    rewritten, _ = _rewrite(model, batch)
    attribute = "linear_1" if hasattr(rewritten.readouts[-1], "linear_1") else "linear"
    assert isinstance(getattr(rewritten.readouts[-1], attribute), BlockLinear)


def test_fold_without_species_skip_is_not_applied():
    model, batch = tiny_cueq("density")
    registry = reg.default_registry()
    requested = {name: "off" for name in registry.names()} | {"block_linear": "on",
                                                               "generic_fold_product_linear": "on"}
    resolution = registry.resolve(copy.deepcopy(model), requested, strict=False)
    decision = resolution.decision("generic_fold_product_linear")
    assert not decision.enabled and "generic_species_skip" in decision.reason


def test_selfcheck_catches_a_wrong_skip_matrix():
    model, batch = tiny_cueq("density")
    rewritten, _ = _rewrite(model, batch)
    with torch.no_grad():
        rewritten.interactions[1].skip_tp.w0.mul_(1.01)
    result = reg.default_registry().selfcheck("generic_fold_product_linear", model, rewritten,
                                              reg.Context(inputs=reg.structure_inputs(batch), seed=0))
    assert not result.passed


@pytest.mark.parametrize("family, supported, reason", [
    ("density", True, "ok"),
    ("residual", False, "skip term"),  # MP-0: the first product adds the first interaction's skip term
    ("mh1", False, "not both generic"),
])
def test_record_support(family, supported, reason):
    model, _ = tiny_cueq(family)
    found = reg.default_registry()["generic_fold_product_linear"].applies_to(model)
    assert tuple(found)[0] == supported and reason in tuple(found)[1]


@pytest.mark.parametrize("family, supported, reason", [
    ("density", True, "generic"),
    ("mh1", True, "ok"),  # through species_tables (the MH-1 record)
    ("residual", False, "skip term"),
])
def test_planner_support_key(family, supported, reason):
    model, _ = tiny_cueq(family)
    found = capabilities.rewrite_support(model)["fold_product_linear"]
    assert tuple(found)[0] == supported and reason in tuple(found)[1]


def test_generic_fold_shares_the_field():
    registry = reg.default_registry()
    assert registry.config_field("generic_fold_product_linear") == "fold_product_linear"
    assert registry["generic_fold_product_linear"].exactness == "rounding"
    names = list(registry.names())
    assert names.index("generic_species_skip") < names.index("generic_fold_product_linear") \
        < names.index("tf32x3_gemms")


def test_the_planner_keeps_the_generic_fold_to_measured_gpus(monkeypatch):
    from test_planner import GPU, plan_for

    from fastmace.config import RewriteConfig
    from fastmace.planner import fixed

    # Decision 66 (5 November 2026): "auto" on for the generic family on the measured GPUs only.
    assert fixed.DECISION_66_GENERIC_FOLD_PRODUCT_GPUS == ("NVIDIA RTX 4000 Ada Generation", "NVIDIA A100-SXM4-80GB",
                                                     "NVIDIA H200")
    assert "NVIDIA A100 80GB PCIe" in fixed.DECISION_66_GENERIC_FOLD_PRODUCT_GPUS  # every A100 model
    assert GPU.name not in fixed.DECISION_66_GENERIC_FOLD_PRODUCT_GPUS
    unmeasured = plan_for("omat0-medium")
    _, choice, why = unmeasured.decision("RewriteConfig.fold_product_linear")
    assert choice == "off" and "generic fold not measured" in str(why) and GPU.name in str(why)
    asked = plan_for("omat0-medium", rewrites=RewriteConfig(fold_product_linear="on"))
    assert asked.rewrites.fold_product_linear == "on"
    monkeypatch.setattr(fixed, "DECISION_66_GENERIC_FOLD_PRODUCT_GPUS", (GPU.name,))
    measured = plan_for("omat0-medium")
    _, choice, why = measured.decision("RewriteConfig.fold_product_linear")
    assert choice == "on" and "generic fold on a measured GPU, decision 66" in str(why)
    assert plan_for("mh1-r2scan").rewrites.fold_product_linear == "on"  # MH-1's species-table fold, unchanged
