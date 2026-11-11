"""``tools/bench/tiled_capacity.py`` ``supercell``: the chunked builder gives ASE's atoms (``supercell_ase``:
``Atoms.repeat``, ``set_cell(scale_atoms=True)``, ``wrap``) bitwise, at chunk sizes that split the copies."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("torch")

TOOLS = Path(__file__).resolve().parents[2] / "tools" / "bench"


@pytest.fixture(scope="module")
def capacity():
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location("_tiled_capacity_supercell", TOOLS / "tiled_capacity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("structure", ["SrTiO3", "NbCl5"])
@pytest.mark.parametrize("repeat", [(1, 1, 1), (3, 4, 5), (7, 2, 3)])
@pytest.mark.parametrize("rattle, strain", [(0.0, 0.0), (0.05, 0.0), (0.05, 0.02)])
@pytest.mark.parametrize("chunk", [1, 7, 1 << 17])
def test_supercell_is_bitwise_ases(capacity, structure, repeat, rattle, strain, chunk):
    want = capacity.supercell_ase(structure, repeat, rattle, strain)
    got = capacity.supercell(structure, repeat, rattle, strain, chunk=chunk)
    assert len(got) == len(want)
    assert got.positions.dtype == np.float64 and got.positions.flags.c_contiguous
    np.testing.assert_array_equal(got.positions, want.positions)
    np.testing.assert_array_equal(got.numbers, want.numbers)
    np.testing.assert_array_equal(got.cell.array, want.cell.array)
    np.testing.assert_array_equal(got.pbc, want.pbc)
    assert sorted(got.arrays) == sorted(want.arrays)
    for name in want.arrays:
        np.testing.assert_array_equal(got.arrays[name], want.arrays[name])
