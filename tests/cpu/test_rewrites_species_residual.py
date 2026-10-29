"""``species_residual`` (fp32 profile, 28 October 2026): the first interaction's residual product as species rows.

The first interaction's residual input is ``linear_up`` of the embedding: with ``species_tables`` a per-species
table. The record adds ``linear_res`` of that table, computed once per species, to ``linear_1``'s products, so the
merged product loses those inputs and their backward. On tiny MH-1-class cuEquivariance models (cuEq's CPU
``naive`` method), float64, eager, with the MH-1 package, the linear family and gated rows on:
- energy and forces equal the model without the record to round-off, with and without ``gated_rows``;
- the merged products lose the residual rows (fewer input columns), and the gated chain reads the new rows;
- the record is opt-in, rounding class, self-checked (a perturbed residual row fails), needs species_tables and
  merged_residual, and the planner keeps it off unless asked.
"""

import copy
import importlib.util

import pytest
import torch

from fastmace.rewrites import fusion, linear, mh1, species
from fastmace.rewrites import registry as reg
from fastmace.rewrites.species import modules as sm

pytest.importorskip("mace")
pytestmark = pytest.mark.skipif(importlib.util.find_spec("cuequivariance_torch") is None,
                                reason="cuequivariance_torch is not installed")

from test_rewrites_mh1 import PACKAGE, context, energy_and_forces, relative, tiny  # noqa: E402

LINEAR = ("block_linear", "grouped_linear", "merged_residual")
SPECIES = ("species_tables", "fold_readout_tail", "fold_product_linear", "species_residual")


def registry():
    out = reg.Registry()
    for family in (mh1, linear, fusion, species):
        for record in family.RECORDS.values():
            out.register(record)
    return out


def rewritten(gated=True, **on):
    model, batch = tiny("mh1", backend="cueq")
    regs = registry()
    base = {*PACKAGE, *LINEAR, *(("gated_rows",) if gated else ())}
    requests = ({name: "on" if name in base else "off" for name in regs.names()}
                | {name: on.get(name, "off") for name in SPECIES})
    out = copy.deepcopy(model)
    resolution = regs.resolve(out, requests, strict=False)
    applied = regs.apply(out, resolution, context(batch), original=model)
    return model, out, batch, resolution, applied


def test_record_is_opt_in_rounding_and_ordered_after_the_merged_products():
    record = species.RECORDS["species_residual"]
    assert record.default == "off" and record.exactness == "rounding" and record.deterministic is True
    assert {"species_tables", "merged_residual"} <= set(record.requires)
    names = reg.default_registry().names()
    assert names.index("species_residual") > names.index("gated_rows")


@pytest.mark.parametrize("gated", [True, False])
def test_rows_replace_the_residual_inputs(gated):
    _, model, _, resolution, applied = rewritten(gated=gated, species_tables="on", species_residual="on")
    assert resolution.decision("species_residual").enabled
    first = model.interactions[0]
    rows = first.residual_linear
    assert isinstance(rows, sm.SpeciesResidualBlockLinear) and first.species_residual is True
    merged = rows.reference[0]
    # the products lose the residual inputs: fewer weight rows than the merged module's
    assert sum(getattr(rows, f"weight_{i}").shape[0] for i in range(len(rows.block_ends))) < sum(
        getattr(merged, f"weight_{i}").shape[0] for i in range(len(merged.block_ends)))
    if gated:
        assert first.gated_rows.first is rows
    result = next(item for item in applied if item.name == "species_residual").result
    assert result.passed and result.max_relative_error <= 1e-12
    assert not getattr(model.interactions[1], "species_residual", False)


@pytest.mark.parametrize("gated", [True, False])
def test_energy_and_forces_to_round_off(gated):
    _, without, batch, _, _ = rewritten(gated=gated, species_tables="on")
    _, with_rows, _, _, _ = rewritten(gated=gated, species_tables="on", species_residual="on")
    energy_and_forces(without, batch)  # throwaway evaluation (docs/development.md)
    energy, forces = energy_and_forces(without, batch)
    new_energy, new_forces = energy_and_forces(with_rows, batch)
    assert relative(new_energy, energy) <= 1e-12 and relative(new_forces, forces) <= 1e-10


def test_with_the_whole_species_family_to_round_off():
    _, without, batch, _, _ = rewritten(species_tables="on", fold_readout_tail="on", fold_product_linear="on")
    _, with_rows, _, _, _ = rewritten(species_tables="on", fold_readout_tail="on", fold_product_linear="on",
                                      species_residual="on")
    energy_and_forces(without, batch)
    energy, forces = energy_and_forces(without, batch)
    new_energy, new_forces = energy_and_forces(with_rows, batch)
    assert relative(new_energy, energy) <= 1e-12 and relative(new_forces, forces) <= 1e-10


def test_needs_species_tables():
    _, model, _, resolution, _ = rewritten(species_residual="on")
    assert not resolution.decision("species_residual").enabled
    assert not getattr(model.interactions[0], "species_residual", False)


def test_selfcheck_catches_a_wrong_row():
    original, model, batch, _, _ = rewritten(species_tables="on", species_residual="on")
    rows = model.interactions[0].residual_linear
    with torch.no_grad():
        getattr(rows, f"residual_{rows.residual_segments[0]}").add_(1e-6)
    assert not species.RECORDS["species_residual"].selfcheck(original, model, context(batch)).passed


def test_planner_keeps_it_off_unless_asked():
    from fastmace.config import RewriteConfig
    from fastmace.planner import fixed

    assert fixed.AUTO_RULES["RewriteConfig.species_residual"] == "never_automatic"
    assert RewriteConfig().species_residual == "auto"
    assert "species_residual" in fixed.REWRITE_FIELDS
