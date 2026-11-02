"""``host_batch`` without MACE's neighbour list: every entry but the edges is MACE's own batch, bitwise.

The force field replaces the batch's edges with its own lists before anything reads them, so the build skips
MACE's host neighbour search (about 4.7 kB per atom at 74 edges per atom, the build's host peak at scale).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastmace.model.forces import BATCH_EDGE_KEYS, host_batch

Z_VALUES = [8, 11, 17, 26]


def calculator():
    from _support.tiny_models import FAMILIES, tiny_mace_calculator

    return tiny_mace_calculator(Z_VALUES, **FAMILIES["mh1"])


def mace_batch(calc, atoms):
    """MACE's own batch (``host_batch`` before it skipped the neighbour search)."""
    from mace import data as mace_data
    from mace.tools import torch_geometric, torch_tools

    calc.arrays_keys.update({calc.charges_key: "charges"})
    keyspec = mace_data.KeySpecification(info_keys=calc.info_keys, arrays_keys=calc.arrays_keys)
    with torch_tools.default_dtype(calc.default_dtype):
        config = mace_data.config_from_atoms(atoms, key_specification=keyspec, head_name=calc.head)
        graph = mace_data.AtomicData.from_config(config, z_table=calc.z_table, cutoff=calc.r_max,
                                                 heads=calc.available_heads)
    return torch_geometric.Batch.from_data_list([graph]).to_dict()


def structures():
    from ase import Atoms

    from _support.structures import nacl

    crystal = nacl(repeat=2, rattle=0.1, seed=8)
    slab = nacl(repeat=2, rattle=0.1, seed=9)
    slab.pbc = (True, True, False)
    molecule = Atoms("NaCl2O", positions=[(0, 0, 0), (2.4, 0, 0), (-2.4, 0.3, 0), (0.1, 2.0, 0.4)], pbc=False)
    return {"crystal": crystal, "slab": slab, "molecule": molecule}


@pytest.mark.parametrize("name", ["crystal", "slab", "molecule"])
def test_host_batch_is_maces_batch_without_its_edges(name):
    calc, atoms = calculator(), structures()[name]
    got, expected = host_batch(calc, atoms.copy()), mace_batch(calc, atoms.copy())
    assert set(got) == set(expected)
    assert int(expected["edge_index"].shape[1]) > 0
    for key, value in expected.items():
        if key in BATCH_EDGE_KEYS:
            assert got[key].shape[0 if key != "edge_index" else 1] == 0
            assert got[key].dtype == value.dtype and got[key].shape[1 if key != "edge_index" else 0] == (
                value.shape[1 if key != "edge_index" else 0])
        elif torch.is_tensor(value):
            assert got[key].dtype == value.dtype and torch.equal(got[key], value), key
        else:
            assert np.array_equal(np.asarray(got[key]), np.asarray(value)), key
