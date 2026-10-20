"""The species family of optimisation round 1 (OPT-T08; docs/design/opt-T08-linear-audit.md).

- ``species_tables`` (C1): the species-only atom-level products become per-species tables. These
  are the node embedding, every interaction's source and target embedding, and the first
  interaction's ``skip_tp`` and ``linear_up``.
- ``fold_readout_tail`` (C2): the last product's linear and the last interaction's ``skip_tp``
  are folded into the last readout's first linear layer.

On tiny MH-1-class cuEquivariance models (cuEq's CPU ``naive`` method), float64, eager, with the
MH-1 package and the linear family on:
- energy and forces equal the model without the records to round-off;
- the records are opt-in, rounding class and self-checked, and a perturbed table or folded
  weight fails the check;
- the planner keeps them off unless asked.
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

LINEAR = ("block_linear", "grouped_linear", "merged_residual", "gated_rows")
NAMES = ("species_tables", "fold_readout_tail")


def registry():
    out = reg.Registry()
    for family in (mh1, linear, fusion, species):
        for record in family.RECORDS.values():
            out.register(record)
    return out


def requests(**on):
    """The MH-1 package and the linear family on, everything else off; ``on`` names the species records."""
    regs = registry()
    base = {*PACKAGE, *LINEAR}
    return ({name: "on" if name in base else "off" for name in regs.names()}
            | {name: on.get(name, "off") for name in NAMES})


def rewritten(family="mh1", **on):
    model, batch = tiny(family, backend="cueq")
    regs = registry()
    out = copy.deepcopy(model)
    resolution = regs.resolve(out, requests(**on), strict=False)
    applied = regs.apply(out, resolution, context(batch), original=model)
    return model, out, batch, resolution, applied


def entry(applied, name):
    return next(item for item in applied if item.name == name)


# --------------------------------------------------------------------------- records


@pytest.mark.parametrize("name", NAMES)
def test_records_are_opt_in_rounding_deterministic(name):
    record = species.RECORDS[name]
    assert record.default == "off" and record.exactness == "rounding" and record.deterministic is True
    assert name in reg.default_registry().names()
    regs = registry()
    model, _ = tiny("mh1", backend="cueq")
    assert not regs.resolve(copy.deepcopy(model), requests(), strict=False).decision(name).enabled


def test_default_registry_holds_the_family_after_the_fusion_family():
    names = reg.default_registry().names()
    assert min(names.index(n) for n in NAMES) > names.index("gated_rows")


def test_planner_keeps_them_off_unless_asked():
    from fastmace.config import RewriteConfig
    from fastmace.planner import fixed

    assert fixed.AUTO_RULES["RewriteConfig.species_tables"] == "never_automatic"
    assert fixed.AUTO_RULES["RewriteConfig.fold_readout_tail"] == "never_automatic"
    assert RewriteConfig().species_tables == "auto" and RewriteConfig().fold_readout_tail == "auto"
    assert set(NAMES) <= set(fixed.REWRITE_FIELDS)


# --------------------------------------------------------------------------- species_tables (C1)


def test_species_tables_replace_the_species_only_modules():
    _, model, _, resolution, applied = rewritten(species_tables="on")
    assert resolution.decision("species_tables").enabled
    assert isinstance(model.node_embedding, sm.OneHotPassthrough)
    first = model.interactions[0]
    assert isinstance(first.skip_tp, sm.SpeciesTable) and isinstance(first.linear_up, sm.SpeciesTable)
    for interaction in model.interactions:
        assert isinstance(interaction.source_embedding, sm.SpeciesTable)
        assert isinstance(interaction.target_embedding, sm.SpeciesTable)
    result = entry(applied, "species_tables").result
    assert result.passed and result.max_relative_error <= 1e-12


def test_species_tables_keep_energy_and_forces_to_round_off():
    _, without, batch, _, _ = rewritten()
    _, with_tables, _, _, _ = rewritten(species_tables="on")
    energy_and_forces(without, batch)  # throwaway evaluation (docs/development.md)
    energy, forces = energy_and_forces(without, batch)
    new_energy, new_forces = energy_and_forces(with_tables, batch)
    assert relative(new_energy, energy) <= 1e-12 and relative(new_forces, forces) <= 1e-10


def test_species_table_gathers_rows_by_species():
    table = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    onehot = torch.nn.functional.one_hot(torch.tensor([2, 0, 3, 2]), 4).to(torch.float64)
    assert torch.equal(sm.SpeciesTable(table)(onehot), onehot @ table)
    assert torch.equal(sm.OneHotPassthrough()(onehot), onehot)


def test_species_tables_selfcheck_catches_a_wrong_table():
    original, model, _, _, _ = rewritten(species_tables="on")
    with torch.no_grad():
        model.interactions[0].skip_tp.table.mul_(1 + 1e-8)
    assert not species.RECORDS["species_tables"].selfcheck(original, model, reg.Context(
        inputs=context(tiny("mh1", backend="cueq")[1]).inputs)).passed


def test_species_tables_refuse_a_joint_embedding_model():
    supported, reason = species.RECORDS["species_tables"].applies_to(tiny("omol", backend="cueq")[0])
    assert not supported and "joint" in reason


# --------------------------------------------------------------------------- fold_readout_tail (C2)


def test_fold_readout_tail_folds_into_the_readout():
    _, model, _, resolution, applied = rewritten(fold_readout_tail="on")
    assert resolution.decision("fold_readout_tail").enabled
    readout = model.readouts[-1]
    assert isinstance(readout.linear_1, torch.nn.Identity)
    width = sum(m * d for _, m, d in model.products[-1].linear.segments_out)
    assert width == sum(m * d for _, m, d in model.interactions[-1].skip_tp.segments_out)
    result = entry(applied, "fold_readout_tail").result
    assert result.passed and result.max_relative_error <= 1e-12


def test_fold_readout_tail_keeps_energy_and_forces_to_round_off():
    _, without, batch, _, _ = rewritten()
    _, folded, _, _, _ = rewritten(fold_readout_tail="on")
    energy_and_forces(without, batch)
    energy, forces = energy_and_forces(without, batch)
    new_energy, new_forces = energy_and_forces(folded, batch)
    assert relative(new_energy, energy) <= 1e-12 and relative(new_forces, forces) <= 1e-10


def test_both_records_together_keep_energy_and_forces_to_round_off():
    _, without, batch, _, _ = rewritten()
    _, both, _, _, applied = rewritten(species_tables="on", fold_readout_tail="on")
    assert {item.name for item in applied} >= set(NAMES)
    energy_and_forces(without, batch)
    energy, forces = energy_and_forces(without, batch)
    new_energy, new_forces = energy_and_forces(both, batch)
    assert relative(new_energy, energy) <= 1e-12 and relative(new_forces, forces) <= 1e-10


def test_fold_readout_tail_selfcheck_catches_a_wrong_fold():
    original, model, _, _, _ = rewritten(fold_readout_tail="on")
    with torch.no_grad():
        model.products[-1].linear.weight_0.mul_(1 + 1e-8)
    assert not species.RECORDS["fold_readout_tail"].selfcheck(original, model, reg.Context()).passed


def test_compose_block_linears_equals_applying_both():
    _, model, _, _, _ = rewritten()
    first, second = model.products[-1].linear, model.readouts[-1].linear_1
    composed = sm.compose(first, second)
    x = torch.randn((5, sum(m * d for _, m, d in first.segments_in)), dtype=torch.float64)
    assert relative(composed(x), second(first(x))) <= 1e-13


# --------------------------------------------------------------------------- end to end


def test_simulation_with_both_records_follows_the_default_to_round_off(tmp_path, monkeypatch):
    """CPU float64, tiny MH-1, 6 Langevin steps through fm.ForceField and Simulation: the plan records
    the opt-in records, and the trajectory follows the same plan without them to round-off."""
    import numpy as np

    from _support.structures import nacl
    from _support.tiny_models import FAMILIES, tiny_mace_model
    from fastmace import integrators
    from fastmace.config import EngineConfig, RewriteConfig
    from fastmace.engine import constraints
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.simulation import Simulation

    monkeypatch.setenv("FASTMACE_CACHE", str(tmp_path / "cache"))
    path = tmp_path / "tiny-mh1.model"
    torch.save(tiny_mace_model([11, 17], **FAMILIES["mh1"]), path)
    loading._LICENCES_SHOWN.add(str(path))
    loaded = loading.load_pristine(str(path), trusted=True)
    engine = EngineConfig(device="cpu", compile_mode="none")
    planner = minimal.MinimalPlanner(engine=engine, environ={}, process=constraints.ProcessRecord())
    monkeypatch.setattr(minimal, "default_planner", lambda force_field: planner)

    def run(**on):
        atoms = nacl(repeat=1, rattle=0.05, seed=3)
        rewrites = RewriteConfig(block_linear="on", **on)
        sim = Simulation(atoms, ForceField(loaded, atoms, precision="float64", rewrites=rewrites),
                         integrator=integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01),
                         seed=11, engine=engine)
        sim.run(steps=6)
        committed = sim.committed()
        out = committed.positions.copy(), committed.energy, {key: choice for key, choice, _ in sim.plan.decisions}
        sim.close()
        return out

    positions, energy, _ = run()
    new_positions, new_energy, decisions = run(species_tables="on", fold_readout_tail="on")
    assert decisions["RewriteConfig.species_tables"] == "on" and decisions["RewriteConfig.fold_readout_tail"] == "on"
    assert float(np.abs(new_positions - positions).max()) <= 1e-12
    assert abs(new_energy - energy) <= 1e-10 * abs(energy)
