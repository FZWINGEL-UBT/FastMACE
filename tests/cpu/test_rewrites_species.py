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
NAMES = ("species_tables", "fold_readout_tail", "fold_product_linear")


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
    assert fixed.AUTO_RULES["RewriteConfig.fold_product_linear"] == "never_automatic"
    assert all(getattr(RewriteConfig(), name) == "auto" for name in NAMES)
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
    assert {item.name for item in applied} >= {"species_tables", "fold_readout_tail"}
    energy_and_forces(without, batch)
    energy, forces = energy_and_forces(without, batch)
    new_energy, new_forces = energy_and_forces(both, batch)
    assert relative(new_energy, energy) <= 1e-12 and relative(new_forces, forces) <= 1e-10


def test_fold_readout_tail_selfcheck_catches_a_wrong_fold():
    original, model, _, _, _ = rewritten(fold_readout_tail="on")
    with torch.no_grad():
        model.products[-1].linear.weight_0.mul_(1 + 1e-8)
    assert not species.RECORDS["fold_readout_tail"].selfcheck(original, model, reg.Context()).passed


# --------------------------------------------------------------------------- fold_product_linear (C3)


def test_fold_product_linear_carries_species_and_folds_the_consumers():
    _, model, _, resolution, applied = rewritten(species_tables="on", fold_product_linear="on")
    assert resolution.decision("fold_product_linear").enabled
    assert isinstance(model.products[0].linear, sm.PadColumns)
    consumers = (model.interactions[1].skip_tp, model.interactions[1].linear_up, model.readouts[0].linear)
    assert all(isinstance(c, sm.FoldedConsumer) for c in consumers)
    result = entry(applied, "fold_product_linear").result
    assert result.passed and result.max_relative_error <= 1e-12


def test_fold_product_linear_needs_species_tables():
    record = species.RECORDS["fold_product_linear"]
    assert "species_tables" in record.requires and "block_linear" in record.requires
    _, _, _, resolution, _ = rewritten(fold_product_linear="on")
    assert not resolution.decision("fold_product_linear").enabled


@pytest.mark.parametrize("tail", ["off", "on"])
def test_fold_product_linear_keeps_energy_and_forces_to_round_off(tail):
    _, without, batch, _, _ = rewritten()
    _, folded, _, _, _ = rewritten(species_tables="on", fold_product_linear="on", fold_readout_tail=tail)
    energy_and_forces(without, batch)
    energy, forces = energy_and_forces(without, batch)
    new_energy, new_forces = energy_and_forces(folded, batch)
    assert relative(new_energy, energy) <= 1e-12 and relative(new_forces, forces) <= 1e-10


def test_fold_product_linear_selfcheck_catches_a_wrong_table():
    original, model, batch, _, _ = rewritten(species_tables="on", fold_product_linear="on")
    with torch.no_grad():
        model.interactions[1].linear_up.table.add_(1e-6)
    assert not species.RECORDS["fold_product_linear"].selfcheck(original, model, context(batch)).passed


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
    new_positions, new_energy, decisions = run(species_tables="on", fold_readout_tail="on", fold_product_linear="on")
    assert all(decisions[f"RewriteConfig.{name}"] == "on" for name in NAMES)
    assert float(np.abs(new_positions - positions).max()) <= 1e-12
    assert abs(new_energy - energy) <= 1e-10 * abs(energy)


# --------------------------------------------------------------------------- conductor conditions (OPT-T08 go)


def _with_species(batch, atom, column):
    """``batch`` with atom ``atom`` given the species of one-hot column ``column`` (phase 3 swaps)."""
    out = dict(batch)
    attrs = batch["node_attrs"].clone()
    attrs[atom] = 0
    attrs[atom, column] = 1
    out["node_attrs"] = attrs
    return out


def test_tables_follow_a_species_change_between_evaluations():
    """Tables cover every element of the model and are gathered per step: changing one atom's species
    between two evaluations of the same rewritten model follows the unrewritten model."""
    _, without, batch, _, _ = rewritten()
    _, folded, _, _, _ = rewritten(species_tables="on", fold_product_linear="on", fold_readout_tail="on")
    _, tables, _, _, _ = rewritten(species_tables="on")
    current = int(batch["node_attrs"][0].argmax())
    changed = _with_species(batch, 0, (current + 1) % batch["node_attrs"].shape[1])
    energy_and_forces(without, batch)
    for state in (batch, changed, batch):
        energy, forces = energy_and_forces(without, state)
        new_energy, new_forces = energy_and_forces(folded, state)
        assert relative(new_energy, energy) <= 1e-12 and relative(new_forces, forces) <= 1e-10
        assert torch.equal(energy_and_forces(tables, state)[0], energy)
    assert not torch.equal(energy_and_forces(without, changed)[0], energy_and_forces(without, batch)[0])


def test_species_tables_alone_are_bitwise_on_the_cpu():
    """A one-hot row times a weight is the weight row, and the tables' products give the same rows: on the CPU
    (float64) energy and forces are bitwise equal. The record stays rounding class (a GPU GEMM's summation
    order may depend on the row count)."""
    _, without, batch, _, _ = rewritten()
    _, tables, _, _, _ = rewritten(species_tables="on")
    energy_and_forces(without, batch)
    energy, forces = energy_and_forces(without, batch)
    new_energy, new_forces = energy_and_forces(tables, batch)
    assert torch.equal(new_energy, energy) and torch.equal(new_forces, forces)


@pytest.mark.parametrize("family", ["omol"])
def test_fold_readout_tail_is_exact_to_round_off_on_other_families(family):
    _, without, batch, _, _ = rewritten(family)
    _, folded, _, resolution, _ = rewritten(family, fold_readout_tail="on")
    assert resolution.decision("fold_readout_tail").enabled
    energy_and_forces(without, batch)
    energy, forces = energy_and_forces(without, batch)
    new_energy, new_forces = energy_and_forces(folded, batch)
    assert relative(new_energy, energy) <= 1e-12 and relative(new_forces, forces) <= 1e-10


@pytest.mark.parametrize(("family", "words"), [
    ("omol", "joint"), ("residual", "first interaction"), ("density", "first interaction"),
    ("off", "first interaction")])
@pytest.mark.parametrize("name", ["species_tables", "fold_product_linear"])
def test_tables_and_product_fold_refuse_other_models(family, words, name):
    supported, reason = species.RECORDS[name].applies_to(tiny(family, backend="cueq")[0])
    assert not supported and words in reason


@pytest.mark.parametrize("family", ["residual", "density", "off"])
def test_fold_readout_tail_refuses_an_element_dependent_skip(family):
    supported, reason = species.RECORDS["fold_readout_tail"].applies_to(tiny(family, backend="cueq")[0])
    assert not supported and "skip_tp" in reason


def _stand_in(label):
    """A stand-in with the structure the records check, as the released checkpoint of ``label`` has it."""
    import types

    import test_planner as tp

    model_class, interactions, _, joint = tp.LABEL_FACTS[label]
    model = type(model_class, (), {})()
    model.atomic_numbers = list(range(5))
    model.node_embedding = types.SimpleNamespace(linear=object())
    model.interactions = []
    for name in interactions:
        block = type(name, (), {})()
        block.skip_tp = type("Linear" if name == tp.MH1 else "FullyConnectedTensorProduct", (), {})()
        block.linear_up = object()
        model.interactions.append(block)
    model.products = [types.SimpleNamespace(use_sc=True) for _ in interactions]
    model.readouts = ([type("LinearReadoutBlock", (), {})() for _ in interactions[:-1]]
                      + [type("NonLinearReadoutBlock", (), {})()])
    if joint:
        model.joint_embedding = object()
    return model


def test_which_of_the_19_labels_each_record_applies_to():
    """species_tables and fold_product_linear: the three MH-1 labels. fold_readout_tail: those and
    OMOL-0 (MH-1 class). POLAR-1 (PolarMACE, external fields) is refused by all three, the residual, density
    and OFF labels by their element-dependent skip (and the first interaction's class)."""
    import test_planner as tp

    applies = {name: {label for label in tp.LABELS if species.RECORDS[name].applies_to(_stand_in(label))[0]}
               for name in NAMES}
    mh1_labels = {"mh1-omat", "mh1-mp", "mh1-r2scan"}
    polar = {"polar1-s", "polar1-m", "polar1-l"}
    assert applies["species_tables"] == applies["fold_product_linear"] == mh1_labels
    assert applies["fold_readout_tail"] == mh1_labels | {"omol0-xl"}
    assert not polar & set().union(*applies.values())


def test_fold_readout_tail_refuses_an_unpinned_last_interaction():
    model = _stand_in("mp0-medium")
    model.interactions[-1] = type("RealAgnosticInteractionBlock", (), {})()
    supported, reason = species.RECORDS["fold_readout_tail"].applies_to(model)
    assert not supported and "RealAgnosticInteractionBlock" in reason
