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



# ---------------------------------------------------------------- species tables at the call's row count


def test_force_field_builds_its_species_tables_at_call_rows(monkeypatch):
    """``call_rows``: the reassociation's identity basis has the model call's row count, not the atom count, so
    a tiled field's construction never evaluates N rows; ``species_tables(rows)`` gives the tables at any count."""
    from fastmace.rewrites.mh1 import modules as m

    calc, atoms = calculator(), structure()
    whole = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, rewrites=only(*MH1_PACKAGE))
    rows = []
    identity_attrs = m.identity_attrs
    monkeypatch.setattr(m, "identity_attrs", lambda columns, width, n, *a: rows.append(n) or identity_attrs(
        columns, width, n, *a))
    small = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, rewrites=only(*MH1_PACKAGE),
                           call_rows=7)
    assert rows and set(rows) == {7}
    assert small.call_rows == 7 and whole.call_rows == len(atoms)
    (_, _, table, _), = small._tables
    assert torch.equal(table, whole.species_tables(7)[0])
    assert torch.equal(whole._tables[0][2], small.species_tables(len(atoms))[0])
    with pytest.raises(ValueError, match="call_rows"):
        MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, rewrites=only(*MH1_PACKAGE), call_rows=0)


@pytest.mark.parametrize("mode", ["eager", "padded"])
def test_tiled_tiles_read_species_tables_at_their_own_rows(mode):
    """Each tile's interaction 0 reads the table evaluated at the tile's node rows (the padded slot's, or the
    eager tile's own), so the table is bitwise the features that tile's embedding gives its atoms."""
    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*MH1_PACKAGE), call_rows=3)
    tiled = TiledForceField(field, TileConfig(5, mode=mode))
    reassociation = tiled.model.interactions[0].first_reassociation
    seen = []
    first_forward = tiled._first_forward

    def recording(leaf, eps, attrs, local, r):
        seen.append((attrs.shape[0], reassociation.species_feats.detach().clone()))
        return first_forward(leaf, eps, attrs, local, r)

    tiled._first_forward = recording
    tiled.evaluate(atoms, stress=True)
    rows_seen = {rows for rows, _ in seen}
    if mode == "eager":
        assert len(rows_seen) > 1  # tiles of several sizes
    else:
        assert rows_seen == {r + h for r, h, _ in tiled.last["slots"].values()}
    for rows, table in seen:
        assert torch.equal(table, field.species_tables(rows)[0])


@pytest.mark.parametrize("mode", ["eager", "padded"])
def test_tiled_call_does_not_depend_on_the_fields_call_rows(mode):
    """The tiled call binds its own tables: bitwise the same whatever row count the field was built with, and
    the field's own call afterwards is bitwise what it was before."""
    calc, atoms = calculator(), structure()
    results = []
    for call_rows in (None, 3):
        field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                               rewrites=only(*MH1_PACKAGE), call_rows=call_rows)
        before = field.evaluate(atoms, stress=True)
        results.append(TiledForceField(field, TileConfig(5, mode=mode)).evaluate(atoms, stress=True))
        after = field.evaluate(atoms, stress=True)
        assert after[0] == before[0]
        assert np.array_equal(after[1], before[1]) and np.array_equal(after[2], before[2])
    assert results[0][0] == results[1][0]
    assert np.array_equal(results[0][1], results[1][1]) and np.array_equal(results[0][2], results[1][2])


# ---------------------------------------------------------------- tile size from the workspace budget


@pytest.mark.parametrize("available, chosen", [(10**9, 16), (900, 8), (879, 4)])  # 8 x 100 x 1.1 = 880
def test_budget_receivers_take_the_largest_tile_whose_workspace_fits(available, chosen):
    """``budget_receivers``: after the first call (at ``receivers``), the call measures its workspace per
    receiver and takes the largest candidate whose predicted workspace, with 10 % headroom, fits the memory left
    after the tiled state; the slots are then planned again. The forces are the fixed-size call's."""
    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*MH1_PACKAGE))
    tiled = TiledForceField(field, TileConfig(4, mode="padded", budget_receivers=(8, 16)))
    tiled._memory = lambda: (100 * tiled.receivers, available)  # 100 bytes per receiver
    first = tiled.evaluate(atoms, stress=True)
    assert tiled.receivers == chosen and tiled.last["receivers"] == 4
    again = tiled.evaluate(atoms, stress=True)
    assert tiled.last["receivers"] == chosen and tiled.last["max_receivers"] <= chosen
    assert max(r for r, _, _ in tiled.last["slots"].values()) <= chosen + 1
    fixed = TiledForceField(field, TileConfig(chosen, mode="padded")).evaluate(atoms, stress=True)
    for got in (first, again):
        assert got[0] == pytest.approx(fixed[0], rel=1e-13, abs=0)
        assert relative(got[1], fixed[1]) <= 1e-11 and relative(got[2], fixed[2]) <= 1e-11
    tiled._memory = lambda: (0, 10**12)
    tiled.evaluate(atoms)
    assert tiled.receivers == chosen  # chosen once, after the first call


def test_budget_receivers_stay_without_a_memory_measurement():
    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, rewrites=only(*MH1_PACKAGE))
    tiled = TiledForceField(field, TileConfig(4, budget_receivers=(8, 16)))
    tiled.evaluate(atoms)  # CPU: no device memory to measure
    assert tiled.receivers == 4
    with pytest.raises(ValueError, match="budget_receivers"):
        TileConfig(4, budget_receivers=(0,))


# ---------------------------------------------------------------- the tiled state on the host


@pytest.mark.parametrize("mode", ["eager", "padded"])
@pytest.mark.parametrize("names", [(), MH1_PACKAGE], ids=["stock", "mh1-rewrites"])
def test_host_state_equals_device_state_bitwise(names, mode):
    """``TileConfig.state="host"``: x, sc and their adjoints live in host memory; each tile reads and writes its
    rows there. The same gathers, assignments and index_add_ as on the device: bitwise the same call."""
    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*names))
    results = []
    for state in ("device", "host"):
        tiled = TiledForceField(field, TileConfig(5, mode=mode, state=state))
        for current in (atoms, strained(atoms)):
            results.append(tiled.evaluate(current, stress=True))
        assert all(t.device.type == "cpu" for t in tiled._g["state"].tensors.values())
        assert tiled.last["host_bytes"] == (tiled.last["global_bytes"] if state == "host" else 0)
    for device, host in zip(results[:2], results[2:]):
        assert device[0] == host[0]
        assert np.array_equal(device[1], host[1]) and np.array_equal(device[2], host[2])


def test_host_state_refuses_graphs_and_unknown_values():
    with pytest.raises(ValueError, match="state"):
        TileConfig(5, state="disk")
    with pytest.raises(ValueError, match="host"):  # a captured graph cannot read host memory
        TileConfig(5, mode="graphs", state="host")


# ---------------------------------------------------------------- the species index instead of the one-hot


def test_tile_rows_from_the_species_index_are_the_one_hot_rows():
    """``TileConfig.species="index"``: a tile's one-hot rows are built from the field's species index, bitwise
    the stored rows (padding rows read atom 0)."""
    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, rewrites=only(*MH1_PACKAGE))
    stored = field.data["node_attrs"].clone()
    tiled = TiledForceField(field, TileConfig(5, species="index"))
    nodes = torch.tensor([0, 3, 3, 7, len(atoms) - 1, 0, 1])
    rows = tiled._attrs(nodes)
    assert rows.dtype == stored.dtype and torch.equal(rows, stored[nodes])


@pytest.mark.parametrize("mode", ["eager", "padded"])
def test_tiled_call_from_the_species_index_equals_the_one_hot_call_bitwise(mode):
    """The tiled call with the species index gives bitwise the call that reads the stored one-hot rows; the
    field's device one-hot is released, and the field's own whole-system call then refuses."""
    calc, atoms = calculator(), structure()
    results = []
    for species in ("one_hot", "index"):
        field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                               rewrites=only(*MH1_PACKAGE))
        tiled = TiledForceField(field, TileConfig(5, mode=mode, species=species))
        tiled.evaluate(atoms, stress=True)  # TorchScript's first call in a process rounds differently
        results.append(tiled.evaluate(strained(atoms), stress=True))
        if species == "index":
            assert field.node_attrs_released and tiled.last["species"] == "index"
            with pytest.raises(RuntimeError, match="node_attrs"):
                field.evaluate(atoms)
    assert results[0][0] == results[1][0]
    assert np.array_equal(results[0][1], results[1][1]) and np.array_equal(results[0][2], results[1][2])


def test_species_index_refuses_unknown_values():
    with pytest.raises(ValueError, match="species"):
        TileConfig(5, species="bits")


def test_field_built_with_its_one_hot_on_the_host_gives_bitwise_the_released_field():
    """``MACEForceField(release_node_attrs=True)`` never puts the one-hot on the device (the construction's
    species check and rewrites read it on the host, E0 is looked up there): the tiled call with the species index
    is bitwise the one of a field released after construction, and the field refuses its own call."""
    calc, atoms = calculator(), structure()
    results = []
    for at_build in (False, True):
        field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                               rewrites=only(*MH1_PACKAGE), release_node_attrs=at_build)
        assert field.node_attrs_released == at_build
        tiled = TiledForceField(field, TileConfig(5, mode="padded", species="index"))
        tiled.evaluate(atoms, stress=True)  # TorchScript's first call in a process rounds differently
        results.append(tiled.evaluate(strained(atoms), stress=True))
        with pytest.raises(RuntimeError, match="node_attrs"):
            field.evaluate(atoms)
    assert results[0][0] == results[1][0]
    assert np.array_equal(results[0][1], results[1][1]) and np.array_equal(results[0][2], results[1][2])
