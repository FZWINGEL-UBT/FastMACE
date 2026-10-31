"""The tiled three-sweep force call (fastmace.model.tiled; docs/design/tiled-backend.md) on CPU.

A tiny float64 MH-1-class model: the tiled call must give the whole-system call's energy, forces and stress to
float64 rounding, for tile sizes from a few atoms (many tiles, every atom in some halo) to the whole cell, with
the stock model and with MH-1's CPU rewrite package (reassociated first interaction, segmented first sum).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from fastmace.config import EngineConfig, NeighbourConfig
from fastmace.model.forces import MACEForceField
from fastmace.model.tiled import TileConfig, TiledForceField, plan_tiles
from fastmace.rewrites import registry as reg

pytest.importorskip("mace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NEIGHBOURS = NeighbourConfig(skin=0.5, sort_receivers="on", gpu_neighbour_list="off")
EAGER = EngineConfig(compile_mode="none")
Z_VALUES = [8, 11, 17, 26]
MH1_PACKAGE = ("factor_species", "fold_cutoff", "density_first", "joint_radial_input", "fused_folded_backward",
               "reassociate_first", "segmented_first_sum")


def structure(seed=8):
    from _support.structures import nacl

    atoms = nacl(repeat=2, rattle=0.1, seed=seed)
    atoms.numbers[3] = 8  # three species, one of them once
    return atoms


def strained(atoms, seed=3, amplitude=0.02):
    e = np.random.default_rng(seed).uniform(-amplitude, amplitude, (3, 3))
    out = atoms.copy()
    out.set_cell(atoms.cell.array @ (np.eye(3) + 0.5 * (e + e.T)), scale_atoms=True)
    return out


def calculator():
    from _support.tiny_models import FAMILIES, tiny_mace_calculator

    return tiny_mace_calculator(Z_VALUES, **FAMILIES["mh1"])


def only(*names):
    return {name: "on" if name in names else "off" for name in reg.default_registry().names()}


def relative(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    return float(np.abs(got - expected).max()) / float(np.abs(expected).max())


@pytest.mark.parametrize("mode, bucket", [("eager", 0.15), ("padded", 0.15), ("padded", 1.0)])
@pytest.mark.parametrize("names", [(), MH1_PACKAGE], ids=["stock", "mh1-rewrites"])
@pytest.mark.parametrize("receivers", [5, 16, 1000])
def test_tiled_call_equals_the_whole_system_call(names, receivers, mode, bucket):
    """Eager tiles at their own shapes, or every tile in a padded slot of its edge bucket (masked receivers and
    halo rows, padding edges beyond the cutoff and beyond the streaming count); ``bucket=1``: one slot."""
    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*names))
    assert [entry.name for entry in field.applied] == list(names)
    tiled = TiledForceField(field, TileConfig(receivers, mode=mode, bucket=bucket))
    for current in (atoms, strained(atoms)):
        expected = field.evaluate(current, stress=True)
        got = tiled.evaluate(current, stress=True)
        assert got[0] == pytest.approx(expected[0], rel=1e-13, abs=0)
        assert relative(got[1], expected[1]) <= 1e-11
        assert relative(got[2], expected[2]) <= 1e-11
        assert tiled.last["max_receivers"] <= receivers
        if receivers < len(atoms):
            assert tiled.last["tiles"] > 1 and tiled.last["max_halo"] > 0
        if mode == "padded":
            slots = list(tiled.last["slots"].values())
            assert tiled.last["max_receivers"] + 1 <= max(r for r, _, _ in slots) <= receivers + 1
            assert max(h for _, h, _ in slots) >= tiled.last["max_halo"]
            assert max(e for _, _, e in slots) >= tiled.last["max_edges"]
            if bucket >= 1:
                assert len(slots) == 1


def test_padded_slots_are_reused_and_grow_only_when_a_tile_does_not_fit():
    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, rewrites=only(*MH1_PACKAGE))
    tiled = TiledForceField(field, TileConfig(8, mode="padded", headroom=0.5, bucket=0.3))
    tiled.evaluate(atoms)
    slots = dict(tiled._slots)
    tiled.evaluate(atoms)
    assert all(tiled._slots[key] is slot for key, slot in slots.items())
    smaller = TiledForceField(field, TileConfig(8, mode="padded", headroom=0.0))
    expected = smaller.evaluate(atoms)
    key = next(iter(smaller._slots))
    smaller._slots[key] = smaller._new_slot((2, 1, 1))  # too small for any tile
    got = smaller.evaluate(atoms)
    assert smaller._slots[key]["sizes"][0] > 2
    assert relative(got[1], expected[1]) <= 1e-12


def test_tiled_call_without_stress():
    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, rewrites=only(*MH1_PACKAGE))
    energy, forces, stress = TiledForceField(field, TileConfig(8)).evaluate(atoms)
    expected = field.evaluate(atoms)
    assert stress is None and expected[2] is None
    assert energy == pytest.approx(expected[0], rel=1e-13, abs=0)
    assert relative(forces, expected[1]) <= 1e-11


@pytest.mark.parametrize("receivers, edges", [(1, 10**9), (7, 10**9), (64, 60), (1000, 10**9)])
def test_plan_tiles_covers_every_atom_once_within_the_caps(receivers, edges):
    rng = np.random.default_rng(0)
    n = 64
    cell = torch.tensor(np.diag([8.0, 9.0, 10.0]) + rng.uniform(-0.5, 0.5, (3, 3)), dtype=torch.float64)
    positions = torch.tensor(rng.uniform(0, 1, (n, 3)), dtype=torch.float64) @ cell
    degree = torch.tensor(rng.integers(1, 20, n))
    row_ptr = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(degree, 0)])
    tiles, grid = plan_tiles(positions, cell, row_ptr, receivers, edges)
    seen = torch.cat(tiles)
    assert sorted(seen.tolist()) == list(range(n))
    for tile in tiles:
        assert torch.all(tile[1:] > tile[:-1])  # sorted: receiver-major edge order is kept
        assert len(tile) <= receivers
        assert len(tile) == 1 or int(degree[tile].sum()) <= edges
    assert len(grid) == 3


def test_refuses_compiled_and_unsupported_fields():
    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER)
    with pytest.raises(ValueError, match="compiled force field"):
        TiledForceField(field, TileConfig(mode="compiled"))
    with pytest.raises(ValueError, match="receivers"):
        TileConfig(0)
    with pytest.raises(ValueError, match="mode"):
        TileConfig(mode="fast")
    with pytest.raises(ValueError, match="bucket"):
        TileConfig(bucket=0)
    with pytest.raises(ValueError, match="CUDA"):
        TiledForceField(MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER), TileConfig(mode="graphs"))


def big_structure(seed=8):
    """NaCl 3x3x3 (16.9 A): at the tiny model's 4 A cutoff, at least three cutoff-wide bins per axis."""
    from _support.structures import nacl

    atoms = nacl(repeat=3, rattle=0.1, seed=seed)
    atoms.numbers[3] = 8
    return atoms


@pytest.mark.parametrize("names", [(), MH1_PACKAGE], ids=["stock", "mh1-rewrites"])
def test_tile_neighbours_are_bitwise_the_global_list(names):
    """Each tile's edges from the cell bins (fastmace.neighbors.tile_list) are the global skin list's edges for
    those receivers after compact at the same positions: the same pairs, unit shifts and shifts, in the same order
    (with MH-1's rewrites the list sorts by sender group too)."""
    from fastmace.neighbors.tile_list import TileNeighbours

    calc, atoms = calculator(), big_structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*names))
    moved = atoms.copy()
    moved.positions += np.random.default_rng(4).uniform(-0.12, 0.12, atoms.positions.shape)
    positions = field._prepare_one_off(moved)
    count = int(field.neighbours.count)
    expected = (field.neighbours.edge_index[:, :count], field.neighbours.unit_shifts[:count],
                field.neighbours.shifts[:count])
    bins = TileNeighbours.like(field.neighbours)
    bins.bin(positions)
    chunks = [bins.edges(chunk) for chunk in torch.arange(len(atoms)).split(37)]
    sender, receiver, unit, shifts = (torch.cat(part) for part in zip(*chunks))
    assert torch.equal(torch.stack([sender, receiver]), expected[0])
    assert torch.equal(unit, expected[1]) and torch.equal(shifts, expected[2])


@pytest.mark.parametrize("mode", ["eager", "padded"])
def test_tiled_call_with_tile_neighbours_equals_the_whole_system_call(mode):
    calc, atoms = calculator(), big_structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*MH1_PACKAGE))
    tiled = TiledForceField(field, TileConfig(40, mode=mode, neighbours="tiles"))
    for current in (atoms, strained(atoms)):
        expected = field.evaluate(current, stress=True)
        got = tiled.evaluate(current, stress=True)
        assert got[0] == pytest.approx(expected[0], rel=1e-13, abs=0)
        assert relative(got[1], expected[1]) <= 1e-11
        assert relative(got[2], expected[2]) <= 1e-11
        assert tiled.last["tiles"] > 1


def test_tile_neighbours_refuse_small_cells():
    from fastmace.neighbors.tile_list import TileNeighbours

    with pytest.raises(ValueError, match="bins"):
        TileNeighbours(4.0, torch.eye(3, dtype=torch.float64) * 11.0, torch.float64)
    with pytest.raises(ValueError, match="neighbours"):
        TileConfig(neighbours="cells")

