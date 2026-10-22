"""The species superset's refresh functions on the GPU (plan task 3-T02; SPEC §8.5), fast tier.

Tiny float64 models (O, Na, Cl present; Fe in the superset, absent and the last column) on CUDA, eager:
every refresh function the force field registers is captured in one CUDA graph (they run on the
device, in place, at fixed addresses, without host synchronisation, so 3-T09 can replay them inside
an attempt graph). After a species and charge change written into ``atomic_numbers`` and ``charges``
and one replay, the derived buffers equal a fresh construction's, and so do energy, forces and stress
(within float64 rounding: the GPU's scatter sums are not bitwise reproducible).
"""

import numpy as np
import pytest
import torch

from fastmace.config import NeighbourConfig
from fastmace.events import DerivedStateRegistry
from fastmace.model.forces import MACEForceField
from fastmace.rewrites import registry as reg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

Z_VALUES = [8, 11, 17, 26]
NEIGHBOURS = NeighbourConfig(skin=0.5, sort_receivers="on", gpu_neighbour_list="on")
MH1_PACKAGE = ("factor_species", "fold_cutoff", "density_first", "joint_radial_input", "fused_folded_backward",
               "reassociate_first", "segmented_first_sum")
GENERIC = ("generic_species_skip", "generic_reassociate", "generic_fused_radial_backward")
JOINT_SPECS = {
    "total_spin": {"type": "categorical", "per": "graph", "in_dim": 1, "emb_dim": 4, "num_classes": 5, "offset": 0},
    "total_charge": {"type": "categorical", "per": "graph", "in_dim": 1, "emb_dim": 4, "num_classes": 9,
                     "offset": 4},
}
CASES = [("mh1", MH1_PACKAGE, {}), ("residual", GENERIC, {"embedding_specs": JOINT_SPECS})]


def structure():
    from _support.structures import nacl

    atoms = nacl(repeat=2, rattle=0.1, seed=8)
    atoms.numbers[3] = 8
    return atoms


def build(calc, atoms, names, charge):
    requests = {name: "on" if name in names else "off" for name in reg.default_registry().names()}
    return MACEForceField(calc, atoms, neighbours=NEIGHBOURS, rewrites=requests, species=tuple(Z_VALUES),
                          compute_stress=True, charge=charge, spin_multiplicity=1 if charge is not None else None)


def outputs(field, atoms):
    positions = torch.tensor(atoms.positions, device="cuda")
    field.rebuild(positions)
    field.compact(positions)
    field.use_edges(int(field.edge_count))
    out = field(positions)
    return [out.energy.detach().clone(), out.forces.detach().clone(), out.stress.detach().clone()]


@pytest.mark.parametrize("family, names, options", CASES, ids=[case[0] for case in CASES])
def test_captured_refresh_equals_a_fresh_construction(family, names, options):
    from _support.tiny_models import FAMILIES, tiny_mace_calculator

    calc = tiny_mace_calculator(Z_VALUES, device="cuda", **{**FAMILIES[family], **options,
                                                            "atomic_energies": np.array([-1.25, -2.5, -3.75, -5.125])})
    joint = "embedding_specs" in options
    atoms = structure()
    field = build(calc, atoms, names, 0 if joint else None)
    registry = DerivedStateRegistry()
    field.register_derived(registry)
    outputs(field, atoms)
    addresses = {name: registry[name].buffer.data_ptr() for name in registry.names()}

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):  # warm-up outside the capture, as the engine does
        for name in registry.names():
            registry[name].refresh()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for name in registry.names():
            registry[name].refresh()

    target = atoms.copy()
    numbers = target.numbers.copy()
    numbers[0], numbers[1] = atoms.numbers[1], atoms.numbers[0]
    numbers[[5, 9]] = 26  # the absent, last superset species
    target.numbers = numbers
    charges = np.zeros(len(atoms))
    if joint:
        charges[[2, 7]] = 1.0
        target.set_initial_charges(charges)
    field.check_species(target.numbers)
    field.atomic_numbers.copy_(torch.as_tensor(target.numbers))
    field.charges.copy_(torch.as_tensor(charges))
    graph.replay()
    torch.cuda.synchronize()
    assert {name: registry[name].buffer.data_ptr() for name in registry.names()} == addresses

    fresh = build(calc, target, names, 2 if joint else None)
    for name, tensor in fresh.derived_buffers().items():
        torch.testing.assert_close(field.derived_buffers()[name], tensor, rtol=1e-14, atol=0, msg=name)
    assert field.neighbours.group_count == len(Z_VALUES)
    for got, expected in zip(outputs(field, target), outputs(fresh, target)):
        torch.testing.assert_close(got, expected, rtol=1e-11, atol=1e-12)
