"""The tiled three-sweep force call (fastmace.model.tiled; docs/design/tiled-backend.md) on CPU.

A tiny float64 MH-1-class model: the tiled call must give the whole-system call's energy, forces and stress to
float64 rounding, for tile sizes from a few atoms (many tiles, every atom in some halo) to the whole cell, with
the stock model and with MH-1's CPU rewrite package (reassociated first interaction, segmented first sum).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from fastmace.config import EngineConfig, NeighbourConfig
from fastmace.model.forces import MACEForceField
from fastmace.model.tiled import TileConfig, TiledForceField, _split_block, plan_tiles
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


def test_padded_slot_kept_while_its_tiles_fit_and_grown_by_a_quarter_of_the_largest():
    """The fit test takes the tiles' own sizes: a slot exactly as large as its largest tile (one padding row more)
    is kept, although its headroom is used up (in MD the tiles' sizes move a little every step, and each growth
    compiles the slot again). A slot that a tile exceeds grows in that size by at least a quarter of the largest
    slot's, so a small slot whose tiles keep growing changes once, not every few steps."""
    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, rewrites=only(*MH1_PACKAGE))
    tiled = TiledForceField(field, TileConfig(8, mode="padded", headroom=0.5, bucket=0.3))
    tiled.evaluate(atoms)
    step = tiled._edge_step
    for key, (r, h, _) in tiled.slot_needs.items():
        tiled._slots[key] = tiled._new_slot((r + 1, h, key * step))  # no headroom left
    changes = tiled.slot_changes
    tiled.evaluate(atoms)
    assert tiled.slot_changes == changes
    assert all(tiled._slots[key]["sizes"] == (r + 1, h, key * step) for key, (r, h, _) in tiled.slot_needs.items())
    key, (r, h, _) = next(iter(tiled.slot_needs.items()))
    tiled._slots[key] = tiled._new_slot((r + 1, h - 1, key * step))  # one halo row short
    largest = max(slot["sizes"][1] for slot in tiled._slots.values())
    tiled.evaluate(atoms)
    assert tiled.slot_changes == changes + 1
    assert tiled._slots[key]["sizes"][:1] == (r + 1,)  # the sizes that fit stay
    assert tiled._slots[key]["sizes"][1] >= h - 1 + math.ceil(0.25 * largest)


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


@pytest.mark.parametrize("receivers, edges", [(1, 10**9), (5, 40), (16, 100), (64, 7), (500, 2**62)])
def test_block_split_by_the_caps_is_the_per_atom_greedy(receivers, edges):
    """``plan_tiles`` with degrees splits each block by one search per piece over the cumulative degrees: the same
    pieces as the per-atom greedy rule it replaces (a piece ends before the atom that would take it to ``receivers``
    atoms or past ``edges`` edges, and holds at least one atom). Degrees with zeros and above the edge cap; blocks
    that start inside the array."""
    rng = np.random.default_rng(1)
    degree = rng.integers(0, 30, 500)
    cumulative = np.concatenate([[0], np.cumsum(degree)])
    for start, stop in ((0, 500), (37, 300), (120, 121), (200, 200)):
        expected, piece, edge_sum = [], start, 0
        for i in range(start, stop):
            if i > piece and (i - piece >= receivers or edge_sum + degree[i] > edges):
                expected.append((piece, i))
                piece, edge_sum = i, 0
            edge_sum += degree[i]
        if stop > piece:
            expected.append((piece, stop))
        assert _split_block(cumulative, start, stop, receivers, edges) == expected


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
            assert field.data["node_attrs"].is_meta  # the one-hot is dropped, not copied to the host
            with pytest.raises(RuntimeError, match="node_attrs"):
                field.evaluate(atoms)
    assert results[0][0] == results[1][0]
    assert np.array_equal(results[0][1], results[1][1]) and np.array_equal(results[0][2], results[1][2])


def test_species_index_releases_the_whole_system_buffers_and_looks_e0_up_bitwise():
    """With the species index the field keeps only what the tiles read (tiled capacity, lever 5): every other
    per-atom entry of ``data`` is a meta tensor, ``batch`` a zero-stride view, and ``node_e0()`` gives bitwise the
    stored float64 E0 column from the species index; the tiled call's heads are one shared element."""
    from fastmace.model import energy as energy64

    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, rewrites=only(*MH1_PACKAGE))
    stored = field.data[energy64.NODE_E0_KEY].clone()
    tiled = TiledForceField(field, TileConfig(5, mode="padded", species="index", neighbours="tiles"))
    assert field.whole_system_released and field.data[energy64.NODE_E0_KEY].is_meta
    assert torch.equal(field.node_e0(), stored)
    assert field.data["batch"].stride(0) == 0 and not bool(field.data["batch"].any())
    n = len(atoms)
    for key, value in field.data.items():
        if torch.is_tensor(value) and value.dim() and value.shape[0] == n and key not in ("positions", "batch"):
            assert value.is_meta, key
    tiled.evaluate(atoms)
    assert tiled._g["heads"].stride(0) == 0 and tiled._g["arange"].shape[0] <= n + 1


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
        if at_build:  # dropped after the construction: its shape and dtype only, no storage
            attrs = field.data["node_attrs"]
            assert attrs.is_meta and attrs.shape[0] == len(atoms) and attrs.dtype == field.dtype
        tiled = TiledForceField(field, TileConfig(5, mode="padded", species="index"))
        tiled.evaluate(atoms, stress=True)  # TorchScript's first call in a process rounds differently
        results.append(tiled.evaluate(strained(atoms), stress=True))
        with pytest.raises(RuntimeError, match="node_attrs"):
            field.evaluate(atoms)
    assert results[0][0] == results[1][0]
    assert np.array_equal(results[0][1], results[1][1]) and np.array_equal(results[0][2], results[1][2])


# ---------------------------------------------------------------- the host state aliased


@pytest.mark.parametrize("mode", ["eager", "padded"])
@pytest.mark.parametrize("names", [(), MH1_PACKAGE], ids=["stock", "mh1-rewrites"])
def test_aliased_host_state_is_bitwise_the_host_state(names, mode):
    """``TileConfig.alias_state``: dE/dsc overwrites sc's rows (one tile reads a receiver's sc and writes its
    dE/dsc), dE/dx takes x's rows once an atom's last tile is done (a window holds the open atoms' sums, added in
    the same order): bitwise the unaliased host state's call, with two of the four state tensors."""
    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*names))
    results, sizes = [], []
    for alias in (False, True):
        tiled = TiledForceField(field, TileConfig(5, mode=mode, state="host", alias_state=alias))
        tiled.evaluate(atoms, stress=True)  # TorchScript's first call in a process rounds differently
        for current in (strained(atoms), atoms):
            results.append(tiled.evaluate(current, stress=True))
        sizes.append(sorted(tiled._g["state"].tensors))
    assert sizes == [["gsc", "gx", "sc", "x"], ["sc", "x"]]
    for plain, aliased in zip(results[:2], results[2:]):
        assert plain[0] == aliased[0]
        assert np.array_equal(plain[1], aliased[1]) and np.array_equal(plain[2], aliased[2])


def test_aliased_state_needs_the_host_state():
    with pytest.raises(ValueError, match="alias_state"):
        TileConfig(5, alias_state=True)


# ---------------------------------------------------------------- the tile lists built once per call


@pytest.mark.parametrize("mode", ["eager", "padded"])
def test_tile_lists_built_once_per_call_are_bitwise_the_rebuilt_ones(mode, monkeypatch):
    """``TileConfig.list_cache``: each tile's edges are found once per call (``TileNeighbours.edges``) and kept on
    the host in a compact form (sender, receiver counts, unit shifts), from which the later sweeps rebuild them
    exactly: bitwise the call that searches them in every sweep."""
    from fastmace.neighbors.tile_list import TileNeighbours

    calc, atoms = calculator(), big_structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*MH1_PACKAGE))
    searches = []
    edges = TileNeighbours.edges
    monkeypatch.setattr(TileNeighbours, "edges", lambda self, receivers: searches.append(1) or edges(self, receivers))
    results = []
    for cache in (False, True):
        tiled = TiledForceField(field, TileConfig(40, mode=mode, neighbours="tiles", list_cache=cache,
                                                  list_reuse=False))
        tiled.evaluate(atoms, stress=True)  # TorchScript's first call in a process rounds differently
        searches.clear()
        results.append(tiled.evaluate(strained(atoms), stress=True))
        tiles = tiled.last["tiles"]
        assert len(searches) == (tiles if cache else (3 + (mode == "padded")) * tiles)
        assert (tiled.last["list_cache_bytes"] > 0) == cache
        if cache:  # int32 senders, one-byte shift codes (every cell longer than the cutoff), narrow counts
            assert tiled.last["list_cache_bytes"] <= 5 * tiled.last["edges"] + len(atoms)
    assert results[0][0] == results[1][0]
    assert np.array_equal(results[0][1], results[1][1]) and np.array_equal(results[0][2], results[1][2])


@pytest.mark.parametrize("stress", [False, True], ids=["fixed", "stress"])
@pytest.mark.parametrize("mode", ["eager", "padded"])
def test_tile_lists_kept_within_half_the_skin_are_bitwise_a_fresh_search(mode, stress, monkeypatch):
    """``TileConfig.list_reuse``: the tiles' edges are searched within cutoff + skin and kept across calls with the
    plan; while no atom has moved more than skin / 2 a call filters them at the cutoff (pairs cross it), bitwise a
    fresh search on the same tiles; a larger move searches again."""
    from fastmace.model import tiled as tiled_module

    calc, atoms = calculator(), big_structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=stress,
                           rewrites=only(*MH1_PACKAGE))
    kept = TiledForceField(field, TileConfig(40, mode=mode, neighbours="tiles", list_reuse=True))
    fresh = TiledForceField(field, TileConfig(40, mode=mode, neighbours="tiles", list_reuse=False))
    x0 = field._prepare_one_off(atoms).clone()
    rng = np.random.default_rng(5)
    direction = torch.tensor(rng.normal(size=(len(atoms), 3)), dtype=x0.dtype)
    direction /= direction.norm(dim=1, keepdim=True)
    skin = field.neighbours.skin
    for tiled in (kept, fresh):
        tiled(x0)  # TorchScript's first call in a process rounds differently
    assert kept.last["list_reused"] is False and kept.list_rebuilds == 1
    edges0 = kept.last["edges"]
    x1 = x0 + 0.4 * skin * direction
    got = kept(x1)
    assert kept.last["list_reused"] is True and kept.last["edges"] != edges0
    tiles = [t.to(x0.device) for t in kept._kept["tiles"]]
    grid = kept._kept["grid"]
    original = tiled_module.plan_tiles
    monkeypatch.setattr(tiled_module, "plan_tiles", lambda *a, **k: (tiles, grid))
    expected = fresh(x1)
    monkeypatch.setattr(tiled_module, "plan_tiles", original)
    assert fresh.last["edges"] == kept.last["edges"]
    assert torch.equal(got.energy, expected.energy) and torch.equal(got.forces, expected.forces)
    kept(x0 + 0.6 * skin * direction)
    assert kept.last["list_reused"] is False and kept.list_rebuilds == 2
    kept.receivers = 30  # a new receiver cap (the budget's switch) plans again
    kept(x0 + 0.6 * skin * direction)
    assert kept.last["list_reused"] is False and kept.list_rebuilds == 3


def test_tile_lists_not_kept_below_the_host_floor():
    """``list_reuse`` is the default for tile lists; a host with less than ``list_reuse_floor`` available keeps
    nothing (neither the kept lists nor the list cache: the later sweeps search again), says why in the call's
    report and gives bitwise the plain call's results, which keep the cache."""
    from fastmace.model.tiled import _host_available

    if _host_available() is None:
        pytest.skip("no /proc/meminfo")
    assert TileConfig(40, neighbours="tiles").list_reuse
    calc, atoms = calculator(), big_structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, rewrites=only(*MH1_PACKAGE))
    starved = TiledForceField(field, TileConfig(40, neighbours="tiles", list_reuse_floor=2**62))
    plain = TiledForceField(field, TileConfig(40, neighbours="tiles", list_reuse=False))
    x0 = field._prepare_one_off(atoms).clone()
    results = []
    for tiled in (starved, plain):
        tiled(x0)  # TorchScript's first call in a process rounds differently
        results.append(tiled(x0))
    assert starved.last["list_reuse"].startswith("off: ") and "below the floor" in starved.last["list_reuse"]
    assert starved.last["list_reused"] is False and starved._kept is None
    assert plain.last["list_reuse"] == "off"
    assert starved.last["list_cache"].startswith("off: ") and "below the floor" in starved.last["list_cache"]
    assert starved.last["list_cache_bytes"] == 0
    assert plain.last["list_cache"] == "on" and plain.last["list_cache_bytes"] > 0
    assert torch.equal(results[0].energy, results[1].energy) and torch.equal(results[0].forces, results[1].forces)


def test_list_cache_unit_shifts_one_byte_codes_or_int16_and_back_exactly():
    """The list cache keeps unit shifts in {-1, 0, 1}^3 as one byte (an index into the 27 combinations), larger
    ones (a cell shorter than the cutoff) as int16, and gives back the float unit shifts exactly; receiver counts at
    the narrowest integer dtype that holds them."""
    from fastmace.model.tiled import _narrow_counts, _pack_units, _unpack_units

    rng = np.random.default_rng(0)
    for low, high, dtype in ((-1, 1, torch.uint8), (-2, 2, torch.int16), (0, 0, torch.uint8)):
        unit = torch.tensor(rng.integers(low, high + 1, (50, 3)), dtype=torch.float32)
        packed = _pack_units(unit)
        assert packed.dtype == dtype
        back = _unpack_units(packed, torch.float32, unit.device)
        assert back.dtype == torch.float32 and torch.equal(back, unit)
    assert _pack_units(torch.full((2, 3), 2.0**15)) is None
    assert _pack_units(torch.zeros((0, 3))).dtype == torch.uint8
    for top, dtype in ((255, torch.uint8), (256, torch.int16), (2**15, torch.int32)):
        counts = torch.tensor([0, 3, top])
        assert _narrow_counts(counts).dtype == dtype and torch.equal(_narrow_counts(counts).long(), counts)


def test_compiled_tiles_default_to_the_workspace_budget():
    assert TileConfig(4096, mode="compiled").budget_receivers == (8192, 16384)
    assert TileConfig(4096).budget_receivers == (8192, 16384)  # "auto": taken only once its tiles compile
    assert TileConfig(4096, mode="eager").budget_receivers == ()
    assert TileConfig(4096, mode="compiled", budget_receivers=()).budget_receivers == ()


# ---------------------------------------------------------------- compiled tiles by default


def test_auto_mode_keeps_eager_tiles_for_an_uncompiled_field():
    """``TileConfig.mode="auto"`` (the default): an eager force field's tiles stay eager, with the reason in the
    call's report."""
    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, rewrites=only(*MH1_PACKAGE))
    assert TileConfig(5).mode == "auto"
    tiled = TiledForceField(field, TileConfig(5))
    tiled.evaluate(atoms)
    assert tiled.last["mode"] == "eager" and "not compiled" in tiled.last["mode_reason"]


@pytest.mark.parametrize("headroom, chosen", [(2.0, "compiled"), (1.0, "eager")])
def test_auto_mode_compiles_the_tiles_when_the_residency_fits(headroom, chosen):
    """A compiled force field's first call runs eager and measures its peak; the later calls compile when the
    memory left at that peak holds the compiled residency and a 1 GiB margin (tiled-compiled-default.md)."""
    from fastmace.model import tiled as tiled_module

    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, rewrites=only(*MH1_PACKAGE))
    tiled = TiledForceField(field, TileConfig(5))
    tiled._auto = True  # as for a compiled force field
    gib = 2**30
    tiled._choose_mode((int(3 * gib), int((3 + headroom) * gib)))  # workspace, available after the call
    assert tiled.mode == chosen
    assert (tiled_module.COMPILED_RESIDENCY + tiled_module.COMPILED_MARGIN < headroom * gib) == (chosen == "compiled")
    tiled._choose_mode(None)  # chosen once
    assert tiled.mode == chosen


# ---------------------------------------------------------------- the host state's traffic deferred


@pytest.mark.parametrize("mode", ["eager", "padded"])
@pytest.mark.parametrize("alias", [False, True], ids=["plain", "aliased"])
def test_deferred_host_state_writes_are_bitwise_the_immediate_ones(mode, alias):
    """``TileConfig.overlap``: a tile's host-state writes are applied after the next tile's work is queued (on CUDA
    that work overlaps them), in the same order; every sweep ends with all of them applied. Bitwise the call that
    applies them at once."""
    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*MH1_PACKAGE))
    results = []
    for overlap in (False, True):
        tiled = TiledForceField(field, TileConfig(5, mode=mode, state="host", alias_state=alias, overlap=overlap))
        tiled.evaluate(atoms, stress=True)  # TorchScript's first call in a process rounds differently
        results.append(tiled.evaluate(strained(atoms), stress=True))
        assert not tiled._g["state"]._pending
    assert results[0][0] == results[1][0]
    assert np.array_equal(results[0][1], results[1][1]) and np.array_equal(results[0][2], results[1][2])


# ---------------------------------------------------------------- the whole-system stages in chunks


def test_binning_and_tile_plan_in_chunks_are_bitwise_unchunked(monkeypatch):
    """``TileNeighbours.bin`` and ``plan_tiles`` work in chunks of ``CHUNK`` atoms (the call's whole-system
    transients): with 7-atom chunks the bins, the order, the tiles and the edges are bitwise those of one chunk;
    and the plan without degrees (``row_ptr`` None) splits blocks exactly as with all-zero degrees."""
    from fastmace.neighbors import tile_list
    from fastmace.neighbors.tile_list import TileNeighbours

    calc, atoms = calculator(), big_structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, rewrites=only(*MH1_PACKAGE))
    positions = field._prepare_one_off(atoms)
    cell = torch.as_tensor(atoms.cell.array)
    n = len(atoms)
    results = []
    for chunk in (tile_list.CHUNK, 7):
        monkeypatch.setattr(tile_list, "CHUNK", chunk)
        bins = TileNeighbours.like(field.neighbours)
        bins.bin(positions)
        tiles, _ = plan_tiles(positions, cell, None, 40, 2**62)
        degrees, _ = plan_tiles(positions, cell, torch.zeros(n + 1, dtype=torch.int64), 40, 2**62)
        assert [t.tolist() for t in tiles] == [t.tolist() for t in degrees]
        results.append((bins.offsets, bins.atom_bin, bins.order, tiles, [bins.edges(t) for t in tiles]))
    (o1, b1, r1, t1, e1), (o2, b2, r2, t2, e2) = results
    assert torch.equal(o1, o2) and torch.equal(b1, b2) and torch.equal(r1, r2)
    assert [t.tolist() for t in t1] == [t.tolist() for t in t2]
    for x, y in zip(e1, e2):
        assert all(torch.equal(a, b) for a, b in zip(x, y))


@pytest.mark.parametrize("mode", ["eager", "padded"])
def test_out_of_memory_at_a_budget_raised_cap_retries_at_the_lower_cap(mode, monkeypatch):
    """A call that runs out of device memory at a receiver cap the budget raised (its first measurement, eager,
    under-predicts a compiled tile's workspace) plans again at the next lower cap and retries; the lower cap stays,
    and the call's report says why. Bitwise a field built at that cap. An error at the configured cap is raised."""
    calc, atoms = calculator(), big_structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, rewrites=only(*MH1_PACKAGE))
    expected_field = TiledForceField(field, TileConfig(20, mode=mode, budget_receivers=(40,)))
    expected_field.evaluate(atoms)  # TorchScript's first call in a process rounds differently
    expected = expected_field.evaluate(atoms)
    tiled = TiledForceField(field, TileConfig(20, mode=mode, budget_receivers=(40,)))
    tiled.evaluate(atoms)
    tiled.receivers, tiled._budget_raised = 40, True  # as the budget's switch after a first call would
    tiled._slots.clear()
    step = TiledForceField._first

    def starved(self, *args, **kwargs):
        if self.receivers > 20:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (test)")
        return step(self, *args, **kwargs)

    monkeypatch.setattr(TiledForceField, "_first", starved)
    got = tiled.evaluate(atoms)
    assert tiled.receivers == 20 and "40 -> 20" in tiled.last["budget_fallback"]
    assert got[0] == expected[0] and np.array_equal(got[1], expected[1])
    with pytest.raises(torch.cuda.OutOfMemoryError):
        tiled.receivers = 40  # the configured cap itself: nothing lower to fall back to
        tiled._budget_raised = False
        tiled.evaluate(atoms)
