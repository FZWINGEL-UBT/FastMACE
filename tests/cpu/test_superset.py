"""The species superset in the force field (plan task 3-T02; SPEC §8.5, §9.3 rules 2 and 8, §12.2).

On the tiny float64 models of ``_support.tiny_models`` (O, Na, Cl present; Fe in the superset
but absent, and the highest column):

- every species-indexed buffer comes from the superset, never from the atoms present: the
  reassociation's per-species table is the node embedding (plus the per-structure joint
  embedding) applied to the identity basis, also for a species with no atoms;
- the sender-group count is the superset size when the last superset species is absent;
- an atom outside the superset raises ``SpeciesNotInSuperset`` (construction and
  ``check_species``), before anything is written;
- the force field registers its derived buffers in a ``DerivedStateRegistry``; a host species or
  charge change refreshed through them equals a fresh construction, bitwise (CPU float64);
- the refresh functions keep every buffer's address.
"""

import numpy as np
import pytest
import torch

from fastmace.config import NeighbourConfig
from fastmace.errors import ModelElementError, SpeciesNotInSuperset
from fastmace.events import DerivedStateRegistry
from fastmace.model.forces import MACEForceField
from fastmace.rewrites import registry as reg

pytest.importorskip("mace")

Z_VALUES = [8, 11, 17, 26]  # Fe absent from the structure, and the model's last column
SUPERSET = (8, 11, 17, 26)
SORTED = NeighbourConfig(skin=0.5, sort_receivers="on", gpu_neighbour_list="off")
UNSORTED = NeighbourConfig(skin=0.5, sort_receivers="off", gpu_neighbour_list="off")
MH1_PACKAGE = ("factor_species", "fold_cutoff", "density_first", "joint_radial_input", "fused_folded_backward",
               "reassociate_first", "segmented_first_sum")
GENERIC = ("generic_species_skip", "generic_reassociate", "generic_segmented_first_sum",
           "generic_fused_radial_backward")
CASES = [("mh1", MH1_PACKAGE), ("residual", GENERIC), ("density", GENERIC), ("mh1", ())]
JOINT_SPECS = {
    "total_spin": {"type": "categorical", "per": "graph", "in_dim": 1, "emb_dim": 4, "num_classes": 5, "offset": 0},
    "total_charge": {"type": "categorical", "per": "graph", "in_dim": 1, "emb_dim": 4, "num_classes": 9,
                     "offset": 4},
}


def structure(seed=8):
    from _support.structures import nacl

    atoms = nacl(repeat=2, rattle=0.1, seed=seed)
    atoms.numbers[3] = 8  # three species, one of them once
    return atoms


def calculator(family="mh1", **options):
    from _support.tiny_models import FAMILIES, tiny_mace_calculator

    options = {**FAMILIES[family], "atomic_energies": np.array([-1.25, -2.5, -3.75, -5.125]), **options}
    return tiny_mace_calculator(Z_VALUES, **options)


def only(*names):
    return {name: "on" if name in names else "off" for name in reg.default_registry().names()}


def field(calc, atoms, names, species=SUPERSET, neighbours=SORTED, **options):
    return MACEForceField(calc, atoms, neighbours=neighbours, rewrites=only(*names), species=species,
                          compute_stress=True, **options)


def outputs(force_field, atoms):
    """Energy, forces and stress at ``atoms``' positions after a canonical rebuild (exact edge count)."""
    positions = torch.tensor(atoms.positions)
    force_field.rebuild(positions)
    force_field.compact(positions)
    force_field.use_edges(int(force_field.edge_count))
    out = force_field(positions)
    return [out.energy.detach().clone(), out.forces.detach().clone(), out.stress.detach().clone()]


def refresh(registry, changed):
    """Refresh every buffer that depends on ``changed`` in registration order (3-T03's ``refresh``)."""
    for name in registry.affected(changed):
        registry[name].refresh()


def mutated(atoms):
    """A swap (atoms 0 and 1), a transmutation to the absent Fe, and one atom of the absent superset species."""
    out = atoms.copy()
    numbers = out.numbers.copy()
    numbers[0], numbers[1] = atoms.numbers[1], atoms.numbers[0]
    numbers[5] = 26
    numbers[9] = 26
    out.numbers = numbers
    return out


# --------------------------------------------------------------------------- identity basis


@pytest.mark.parametrize("family, names", [("mh1", MH1_PACKAGE), ("residual", GENERIC)])
def test_absent_superset_species_takes_its_row_from_the_identity_basis(family, names):
    calc, atoms = calculator(family), structure()
    ff = field(calc, atoms, names)
    assert ff.superset == SUPERSET and ff.species_columns == (0, 1, 2, 3)
    model = getattr(ff.model, "_orig_mod", ff.model)
    interaction = model.interactions[0]
    reassociation = interaction.first_reassociation
    assert reassociation.columns.tolist() == [0, 1, 2, 3]
    with torch.no_grad():
        identity = torch.eye(len(Z_VALUES), dtype=torch.float64)
        expected = interaction.linear_up(model.node_embedding(identity))
    table = reassociation.species_feats
    assert table.shape == expected.shape
    torch.testing.assert_close(table, expected, rtol=1e-14, atol=1e-15)
    # Not "the first atom of each species": Fe has none, and its row is not atom 0's species row.
    first_species = Z_VALUES.index(int(atoms.numbers[0]))
    assert not torch.equal(table[3], table[first_species])


def test_the_identity_table_carries_the_per_structure_joint_embedding():
    """OMOL-0's per-graph joint embedding (total charge and spin) is part of every row."""
    calc, atoms = calculator("residual", embedding_specs=JOINT_SPECS), structure()
    neutral = field(calc, atoms, GENERIC, charge=0, spin_multiplicity=1)
    charged = field(calc, atoms, GENERIC, charge=2, spin_multiplicity=1)
    a = neutral.derived_buffers()["species_feats.interactions.0"]
    b = charged.derived_buffers()["species_feats.interactions.0"]
    assert a.shape == b.shape == (4, a.shape[1]) and not torch.equal(a, b)
    model = charged.model
    with torch.no_grad():
        rows = model.node_embedding(torch.eye(len(Z_VALUES), dtype=torch.float64))
        rows = rows + model.joint_embedding(torch.zeros(4, dtype=torch.int64),
                                            {name: charged.data[name] for name in JOINT_SPECS})
        expected = model.interactions[0].linear_up(rows)
    torch.testing.assert_close(b, expected, rtol=1e-14, atol=1e-15)


def test_superset_equal_to_the_structure_is_the_default():
    calc, atoms = calculator(), structure()
    default = field(calc, atoms, MH1_PACKAGE, species=None)
    explicit = field(calc, atoms, MH1_PACKAGE, species=(17, 8, 11))
    assert default.superset == explicit.superset == (8, 11, 17)
    assert default.species_columns == (0, 1, 2)
    for a, b in zip(outputs(default, atoms), outputs(explicit, atoms)):
        assert torch.equal(a, b)


def test_node_e0_equals_the_one_hot_product():
    from fastmace.model import energy as energy64

    calc, atoms = calculator(), structure()
    ff = field(calc, atoms, MH1_PACKAGE)
    expected = energy64.node_e0_float64(ff.atomic_energies, ff.data["node_attrs"])
    assert torch.equal(ff.data[energy64.NODE_E0_KEY], expected)


# --------------------------------------------------------------------------- sender groups


def test_last_superset_species_absent_keeps_the_sender_group_count():
    calc, atoms = calculator(), structure()
    ff = field(calc, atoms, MH1_PACKAGE)
    assert 26 not in atoms.numbers and ff.superset[-1] == 26
    assert ff.neighbours.group_count == len(SUPERSET) == 4
    groups = ff.neighbours.sender_groups
    assert groups.tolist() == [SUPERSET.index(int(z)) for z in atoms.numbers]
    # Structure-only superset: still its size, not max() + 1 of what is present.
    small = field(calc, atoms, MH1_PACKAGE, species=None)
    assert small.neighbours.group_count == 3


# --------------------------------------------------------------------------- membership


def test_an_atom_outside_the_superset_raises_at_construction():
    calc, atoms = calculator(), structure()
    with pytest.raises(SpeciesNotInSuperset) as caught:
        field(calc, atoms, MH1_PACKAGE, species=(11, 17))
    assert caught.value.reasons[0].rule == 2 and caught.value.reasons[0].code == "species-not-in-superset"
    assert "O" in str(caught.value)


def test_a_superset_species_without_embedding_raises():
    calc, atoms = calculator(), structure()
    with pytest.raises(ModelElementError, match="Xe"):
        field(calc, atoms, MH1_PACKAGE, species=(8, 11, 17, 54))


def test_check_species_raises_before_anything_is_written():
    calc, atoms = calculator(), structure()
    ff = field(calc, atoms, MH1_PACKAGE)
    before = {name: tensor.clone() for name, tensor in ff.derived_buffers().items()}
    numbers = atoms.numbers.copy()
    ff.check_species(numbers)  # fine
    numbers[2] = 54
    with pytest.raises(SpeciesNotInSuperset, match="Xe"):
        ff.check_species(numbers)
    with pytest.raises(ValueError, match="atoms"):
        ff.check_species(numbers[:-1])
    for name, tensor in ff.derived_buffers().items():
        assert torch.equal(tensor, before[name]), name
    assert torch.equal(ff.atomic_numbers.cpu(), torch.as_tensor(atoms.numbers))


# --------------------------------------------------------------------------- registry and refresh


def test_registers_its_derived_buffers():
    calc, atoms = calculator(), structure()
    ff = field(calc, atoms, MH1_PACKAGE)
    registry = DerivedStateRegistry()
    ff.register_derived(registry)
    names = set(registry.names())
    assert {"force_field.node_attrs", "force_field.node_e0", "force_field.sender_groups",
            "force_field.total_charge", "force_field.total_spin",
            "force_field.species_feats.interactions.0"} <= names
    for name in ("force_field.node_attrs", "force_field.node_e0", "force_field.sender_groups"):
        assert registry[name].depends_on == ("species",)
    for name in ("force_field.total_charge", "force_field.total_spin", "force_field.species_feats.interactions.0"):
        assert registry[name].depends_on == ("charges",)
    assert registry.names().index("force_field.total_charge") < registry.names().index(
        "force_field.species_feats.interactions.0")


@pytest.mark.parametrize("family, names", CASES, ids=lambda v: "+".join(v) if isinstance(v, tuple) else v)
def test_host_species_change_equals_a_fresh_construction(family, names):
    calc, atoms = calculator(family), structure()
    ff = field(calc, atoms, names)
    registry = DerivedStateRegistry()
    ff.register_derived(registry)
    addresses = {name: registry[name].buffer.data_ptr() for name in registry.names()}
    outputs(ff, atoms)
    target = mutated(atoms)
    ff.check_species(target.numbers)
    ff.atomic_numbers.copy_(torch.as_tensor(target.numbers))  # the transaction's write (3-T03)
    refresh(registry, ["species"])
    got = outputs(ff, target)
    fresh = field(calc, target, names)
    expected = outputs(fresh, target)
    for a, b in zip(got, expected):
        assert torch.equal(a, b), f"max |difference| {(a - b).abs().max():.3e}"
    for name, tensor in fresh.derived_buffers().items():
        assert torch.equal(ff.derived_buffers()[name], tensor), name
    assert {name: registry[name].buffer.data_ptr() for name in registry.names()} == addresses


def test_host_charge_change_equals_a_fresh_construction():
    """Total charge follows the per-atom charges; the joint-embedding table follows the total charge."""
    calc, atoms = calculator("residual", embedding_specs=JOINT_SPECS), structure()
    ff = field(calc, atoms, GENERIC, charge=0, spin_multiplicity=1)
    registry = DerivedStateRegistry()
    ff.register_derived(registry)
    outputs(ff, atoms)
    charges = np.zeros(len(atoms))
    charges[[2, 7]] = 1.0  # two formal charges of +1: total charge 2, spin parity unchanged
    ff.charges.copy_(torch.as_tensor(charges))
    refresh(registry, ["charges"])
    got = outputs(ff, atoms)
    charged = atoms.copy()
    charged.set_initial_charges(charges)
    fresh = field(calc, charged, GENERIC, charge=2, spin_multiplicity=1)
    assert float(ff.data["total_charge"].reshape(-1)[0]) == 2.0
    for a, b in zip(got, outputs(fresh, charged)):
        assert torch.equal(a, b), f"max |difference| {(a - b).abs().max():.3e}"


def test_rewrites_follow_the_superset_columns():
    calc, atoms = calculator("residual"), structure()
    ff = field(calc, atoms, GENERIC)
    skips = [i.skip_tp for i in ff.model.interactions if type(i.skip_tp).__name__ == "SpeciesSkip"]
    assert skips and all(skip.columns.tolist() == [0, 1, 2, 3] for skip in skips)
    assert ff.model.interactions[0].first_reassociation.columns.tolist() == [0, 1, 2, 3]
    assert ff.info()["species_columns"] == [0, 1, 2, 3] and ff.info()["superset"] == list(SUPERSET)



def test_refreshes_read_the_engine_state_when_given():
    """3-T03's engine passes its SimulationState: the transaction then writes state.species only."""
    from fastmace.state import SimulationState

    calc, atoms = calculator(), structure()
    ff = field(calc, atoms, MH1_PACKAGE)
    state = SimulationState.from_atoms(atoms, ff.device, 1)
    registry = DerivedStateRegistry()
    ff.register_derived(registry, state)
    assert ff.atomic_numbers is state.species
    target = mutated(atoms)
    state.species.copy_(torch.as_tensor(target.numbers))
    refresh(registry, ["species"])
    fresh = field(calc, target, MH1_PACKAGE)
    for a, b in zip(outputs(ff, target), outputs(fresh, target)):
        assert torch.equal(a, b)
    with pytest.raises(ValueError, match="differs"):
        ff.use_sources(torch.as_tensor(atoms.numbers, dtype=torch.int64))


def test_a_host_total_charge_change_sets_the_baseline():
    calc, atoms = calculator("residual", embedding_specs=JOINT_SPECS), structure()
    ff = field(calc, atoms, GENERIC, charge=0, spin_multiplicity=1)
    registry = DerivedStateRegistry()
    ff.register_derived(registry)
    ff.set_total_charge(2)
    refresh(registry, ["charges"])
    fresh = field(calc, atoms, GENERIC, charge=2, spin_multiplicity=1)
    for a, b in zip(outputs(ff, atoms), outputs(fresh, atoms)):
        assert torch.equal(a, b)


@pytest.mark.parametrize("joint", [False, True])
def test_every_force_field_tensor_equals_a_fresh_construction(joint):
    """3-T03's registry check compares every tensor the force field holds, scratch buffers included."""
    if joint:
        calc, names, options = calculator("residual", embedding_specs=JOINT_SPECS), GENERIC, dict(spin_multiplicity=1)
    else:
        calc, names, options = calculator(), MH1_PACKAGE, {}
    atoms = structure()
    ff = field(calc, atoms, names, neighbours=UNSORTED, **({"charge": 0} if joint else {}), **options)
    registry = DerivedStateRegistry()
    ff.register_derived(registry)
    target = mutated(atoms)
    charges = np.zeros(len(atoms))
    if joint:
        charges[[2, 7]] = 1.0
        target.set_initial_charges(charges)
    ff.atomic_numbers.copy_(torch.as_tensor(target.numbers))
    ff.charges.copy_(torch.as_tensor(charges))
    refresh(registry, ["species", "charges"])
    fresh = field(calc, target, names, neighbours=UNSORTED, **({"charge": 2} if joint else {}), **options)
    outputs(ff, target)
    outputs(fresh, target)
    tensors = {name: value for name, value in vars(fresh).items() if torch.is_tensor(value)}
    assert {"_species_index", "_sender_groups", "_charge_sum", "_charge_offset"} <= set(tensors)
    for name, value in tensors.items():
        assert torch.equal(vars(ff)[name], value), name
