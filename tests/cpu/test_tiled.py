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


def test_a_tile_whose_bucket_has_no_slot_runs_in_a_neighbouring_slot_that_fits():
    """In MD a tile's edges drift across its bucket's edge by a few per cent. A tile whose bucket has no slot runs
    in the bucket above's or below's slot when it fits there, instead of compiling a new slot; the same call."""
    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, rewrites=only(*MH1_PACKAGE))
    tiled = TiledForceField(field, TileConfig(8, mode="padded", headroom=0.5, bucket=0.3))
    expected = tiled.evaluate(atoms)
    top = max(tiled._slots)
    slot = tiled._slots.pop(top)  # the top bucket's tiles drifted one bucket up
    tiled._slots[top + 1] = slot
    changes = tiled.slot_changes
    got = tiled.evaluate(atoms)
    assert top not in tiled._slots and tiled._slots[top + 1] is slot
    assert tiled.slot_changes == changes
    assert top + 1 in tiled.slot_needs
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


@pytest.mark.parametrize("split", [1, 2])
@pytest.mark.parametrize("names", [(), MH1_PACKAGE], ids=["stock", "mh1-rewrites"])
def test_tile_neighbours_are_bitwise_the_global_list(names, split):
    """Each tile's edges from the cell bins (fastmace.neighbors.tile_list) are the global skin list's edges for
    those receivers after compact at the same positions: the same pairs, unit shifts and shifts, in the same order
    (with MH-1's rewrites the list sorts by sender group too), with cutoff-wide bins and a +-1 stencil and with
    half-cutoff bins and a +-2 stencil (the default on this cell)."""
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
    bins = TileNeighbours.like(field.neighbours, split=split)
    assert TileNeighbours.like(field.neighbours).split == 2 and bins.split == split
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
    # The smallest accepted cell (three cutoff-wide bins) holds six half-cutoff bins: the +-2 stencil.
    assert TileNeighbours(4.0, torch.eye(3, dtype=torch.float64) * 12.0, torch.float64).split == 2
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


def test_compiled_graphs_refuse_other_modes_host_state_lanes_and_the_cpu():
    for kwargs, match in ((dict(mode="padded"), "mode"), (dict(mode="graphs"), "mode"),
                          (dict(mode="compiled", state="host"), "host"), (dict(mode="auto", lanes="on"), "lanes")):
        with pytest.raises(ValueError, match=match):
            TileConfig(5, compiled_graphs=True, **kwargs)
    TileConfig(5, mode="auto", compiled_graphs=True)
    calc, atoms = calculator(), structure()
    with pytest.raises(ValueError, match="CUDA"):
        TiledForceField(MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER),
                        TileConfig(mode="auto", compiled_graphs=True))


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
            assert field.atomic_numbers.device.type == "cpu"  # the species source on the host
            index = field._species_index.clone()
            field._refresh_node_attrs()  # a species event's refresh still looks up from it
            assert torch.equal(field._species_index, index)
            with pytest.raises(RuntimeError, match="node_attrs"):
                field.evaluate(atoms)
    assert results[0][0] == results[1][0]
    assert np.array_equal(results[0][1], results[1][1]) and np.array_equal(results[0][2], results[1][2])


def test_species_index_releases_the_whole_system_buffers_and_looks_e0_up_bitwise():
    """With the species index the field keeps only what the tiles read (tiled capacity, lever 5): every other
    per-atom entry of ``data`` is a meta tensor, ``batch`` a zero-stride view, and ``node_e0()`` gives bitwise the
    stored float64 E0 column from the species index; the tiled call's heads are one shared element."""
    from fastmace.model import energy as energy64

    calc, atoms = calculator(), big_structure()
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
    results, layouts = [], []
    # (one-hot released at the construction, whole system released at the construction): the third never puts
    # the whole-system call's per-atom buffers on the device; afterwards it is the same field.
    for at_build, whole in ((False, False), (True, False), (True, True)):
        field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                               rewrites=only(*MH1_PACKAGE), release_node_attrs=at_build,
                               release_whole_system=whole)
        assert field.node_attrs_released == at_build and field.whole_system_released == whole
        if at_build:  # dropped after the construction: its shape and dtype only, no storage
            attrs = field.data["node_attrs"]
            assert attrs.is_meta and attrs.shape[0] == len(atoms) and attrs.dtype == field.dtype
        tiled = TiledForceField(field, TileConfig(5, mode="padded", species="index"))
        layouts.append({key: (tuple(value.shape), value.dtype, value.device.type, value.stride())
                        for key, value in field.data.items() if torch.is_tensor(value)}
                       | {name: (tuple(t.shape), t.dtype, t.device.type) for name, t in (
                           ("atomic_numbers", field.atomic_numbers), ("charges", field.charges),
                           ("species", field._species_index), ("groups", field._sender_groups))})
        tiled.evaluate(atoms, stress=True)  # TorchScript's first call in a process rounds differently
        results.append(tiled.evaluate(strained(atoms), stress=True))
        with pytest.raises(RuntimeError, match="node_attrs"):
            field.evaluate(atoms)
    assert layouts[0] == layouts[1] == layouts[2]
    for other in results[1:]:
        assert results[0][0] == other[0]
        assert np.array_equal(results[0][1], other[1]) and np.array_equal(results[0][2], other[2])


def test_field_without_its_own_list_gives_bitwise_the_tile_list_call():
    """``MACEForceField(global_list=False)``: the field's own list stays one empty slot (construction and every
    one-off call), so the tiled call with tile lists is bitwise the one of a field with its list; the field's own
    call and ``TileConfig.neighbours="global"`` refuse it."""
    calc, atoms = calculator(), big_structure()
    results = []
    for own in (True, False):
        field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                               rewrites=only(*MH1_PACKAGE), release_node_attrs=True, release_whole_system=True,
                               global_list=own)
        assert (field.neighbours.capacity == 1 and int(field.neighbours.count) == 0) != own
        if not own:
            with pytest.raises(ValueError, match="global_list"):
                TiledForceField(field, TileConfig(12, neighbours="global", species="index"))
        tiled = TiledForceField(field, TileConfig(12, mode="padded", neighbours="tiles", species="index"))
        tiled.evaluate(atoms, stress=True)  # TorchScript's first call in a process rounds differently
        results.append(tiled.evaluate(strained(atoms), stress=True))
        assert (int(field.neighbours.count) == 0) != own
    assert results[0][0] == results[1][0]
    assert np.array_equal(results[0][1], results[1][1]) and np.array_equal(results[0][2], results[1][2])
    bare = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, rewrites=only(), global_list=False)
    with pytest.raises(RuntimeError, match="global_list"):
        bare.evaluate(atoms)


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
    # Sweeps 2 and 3 from the call's filtered lists (and sweep 1 in padded mode, after its sizing pass's search).
    assert kept.last["list_cache_hits"] == (3 if mode == "padded" else 2) * kept.last["tiles"]
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


@pytest.mark.parametrize("reach", [20.0, 900.0], ids=["small-cell", "large-coordinates"])
def test_filter32_keeps_exactly_the_float64_tests_edges(reach):
    """``TILE_FILTER32``: the float32 test with the float64 one in its band keeps the same edges, units and shifts as
    the float64 test, also for pairs within 1e-7 A of the cutoff and coordinates up to 900 A."""
    from fastmace.model.tiled import _filter32, _filter32_band
    from fastmace.neighbors.skin import times_3x3

    torch.manual_seed(3)
    cutoff, n, e = 5.0, 4000, 60000
    cell = torch.tensor([[11.3, 0.0, 0.0], [0.4, 12.1, 0.0], [0.2, -0.3, 10.7]], dtype=torch.float64)
    positions64 = (torch.rand(n, 3, dtype=torch.float64) * 2 - 1) * (reach - 15)
    sender, receiver = torch.randint(0, n, (e,)), torch.randint(0, n // 2, (e,))
    codes = torch.randint(0, 27, (e,))
    table = torch.stack([torch.arange(27) // 9, torch.arange(27) // 3 % 3, torch.arange(27) % 3], 1).double() - 1
    shifts27 = times_3x3(table, cell)
    unit = table[codes]
    # the first n / 2 pairs: each its own sender (atoms n / 2 .. n - 1, never a receiver), placed at the cutoff
    # +- up to 1e-7 A from its receiver along a random direction
    near = torch.arange(n // 2)
    sender[near] = near + n // 2
    direction = torch.nn.functional.normalize(torch.randn(n // 2, 3, dtype=torch.float64), dim=1)
    length = cutoff + (torch.rand(n // 2, dtype=torch.float64) * 2 - 1) * 1e-7
    positions64 = positions64.clone()
    positions64[sender[near]] = positions64[receiver[near]] + shifts27[codes[near]] - direction * length.unsqueeze(1)
    reach_now = float(positions64.abs().max()) + float(shifts27.abs().max())
    band = _filter32_band(cutoff, reach_now)
    got = _filter32(positions64.float(), positions64, shifts27, band, cutoff, sender, receiver, unit, codes,
                    torch.float32)
    vectors = positions64[receiver] - positions64[sender] + times_3x3(unit, cell)
    keep = (vectors * vectors).sum(dim=1) < cutoff ** 2
    expected = sender[keep], receiver[keep], unit[keep], times_3x3(unit, cell)[keep].to(torch.float32)
    lengths = vectors.norm(dim=1)
    assert int(((lengths - cutoff).abs() < 1e-6).sum()) > 1000  # many pairs at the cutoff
    for a, b in zip(got, expected):
        assert torch.equal(a, b)


@pytest.mark.parametrize("mode", ["eager", "padded"])
def test_filter32_tiled_call_is_bitwise_the_float64_filter(mode, monkeypatch):
    """``TILE_FILTER32`` in the tiled call: a call reusing its kept lists gives bitwise the call with the float64
    filter (the same edges in the same order)."""
    from fastmace.model import tiled as tiled_module

    calc, atoms = calculator(), big_structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*MH1_PACKAGE))
    x0 = field._prepare_one_off(atoms).clone()
    rng = np.random.default_rng(7)
    direction = torch.tensor(rng.normal(size=(len(atoms), 3)), dtype=x0.dtype)
    direction /= direction.norm(dim=1, keepdim=True)
    x1 = x0 + 0.4 * field.neighbours.skin * direction
    results = {}
    for fast in (False, True):
        monkeypatch.setattr(tiled_module, "TILE_FILTER32", fast)
        tiled = TiledForceField(field, TileConfig(40, mode=mode, neighbours="tiles", list_reuse=True))
        tiled(x0)
        tiled(x0)  # TorchScript's first call in a process rounds differently
        results[fast] = tiled(x1)
        assert tiled.last["list_reused"] is True
    assert torch.equal(results[True].energy, results[False].energy)
    assert torch.equal(results[True].forces, results[False].forces)
    assert torch.equal(results[True].stress, results[False].stress)


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


# ---------------------------------------------------------------- the wavefront schedule


@pytest.mark.parametrize("group", [1, 3, 5])
def test_wavefront_schedule_runs_every_sweep_once_after_what_it_reads(group):
    """``wavefront_schedule``: each tile's three sweeps once; sweep 2 in tile order, ``group`` tiles in a run; a
    tile's sweep 1 before every sweep 2 that needs it, its sweep 3 after the sweep 2 of its ``ready`` tile; the peak
    is the most receivers between their sweeps 1 and 3. A group changes sweep at most three times."""
    from fastmace.model.tiled import wavefront_schedule

    # A ring of 16 tiles (periodic): tile t reads its neighbours' x; its receivers sit in their halos too.
    tiles = 16
    needs = [sorted({(t - 1) % tiles, t, (t + 1) % tiles}) for t in range(tiles)]
    ready = [tiles - 1 if t == 0 else min(t + 1, tiles - 1) for t in range(tiles)]
    counts = [3] * tiles
    steps, peak = wavefront_schedule(needs, ready, counts, group)
    assert sorted(steps) == sorted((s, t) for s in (1, 2, 3) for t in range(tiles))
    position = {step: i for i, step in enumerate(steps)}
    assert [t for s, t in steps if s == 2] == list(range(tiles))
    for t in range(tiles):
        assert all(position[(1, u)] < position[(2, t)] for u in needs[t])
        assert position[(3, t)] > position[(2, ready[t])]
    live = most = 0
    for sweep, t in steps:
        live += counts[t] if sweep == 1 else -counts[t] if sweep == 3 else 0
        most = max(most, live)
    assert peak == most < sum(counts)
    def changes(order):
        return sum(a[0] != b[0] for a, b in zip(order, order[1:]))

    assert changes(steps) <= 3 * -(-tiles // group)
    if group > 1:
        assert changes(steps) < changes(wavefront_schedule(needs, ready, counts)[0])


@pytest.mark.parametrize("mode, group", [("eager", 1), ("padded", 1), ("padded", 3)])
@pytest.mark.parametrize("overlap", [False, True], ids=["immediate", "deferred"])
def test_wavefront_call_equals_the_three_sweep_call(mode, group, overlap):
    """``TileConfig.wavefront``: the pooled rows hold bitwise the state of the three plain sweeps, so the node
    energies (and the energy) are bitwise; the forces and the stress are the same sums in another order. The pool
    holds fewer rows than atoms, and the lists of a tile are dropped after its sweep 3."""
    calc, atoms = calculator(), big_structure().repeat((4, 1, 1))  # 10 tile layers along the long axis
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*MH1_PACKAGE))
    results = []
    for wave in ("off", "on"):
        tiled = TiledForceField(field, TileConfig(12, mode=mode, neighbours="tiles", state="host", alias_state=True,
                                                  overlap=overlap, wavefront=wave, wavefront_group=group,
                                                  list_reuse=False, boundary_skin=False))
        tiled.evaluate(atoms, stress=True)  # TorchScript's first call in a process rounds differently
        for current in (strained(atoms), atoms):
            results.append(tiled.evaluate(current, stress=True))
        assert tiled.last["wavefront"]["on"] == (wave == "on")
        assert not tiled._g["state"]._pending
        assert tiled._g["forces"] is None  # handed out: the call keeps no reference to its forces
        assert tiled._g["state"]._first is None and tiled._g["state"]._last is None  # freed after the window plan
        assert (tiled.last["list_cache_bytes"] == 0) == (wave == "on")  # each tile's lists dropped after its sweep 3
    assert tiled.last["grid"][0] >= 8
    assert tiled.last["wavefront"]["pool_rows"] < 0.8 * len(atoms)
    assert tiled.last["tiles"] > 8
    for plain, wave in zip(results[:2], results[2:]):
        assert plain[0] == wave[0]
        assert relative(wave[1], plain[1]) <= 1e-13
        assert relative(wave[2], plain[2]) <= 1e-13


@pytest.mark.parametrize("available, on", [(2**40, False), (2**20, True)])
def test_wavefront_auto_only_where_the_plain_state_would_not_fit(available, on, monkeypatch):
    """``wavefront="auto"``: from ``WAVEFRONT_LAYERS`` grid layers, on only when the plain state would leave the host
    below the floor (decided once per atom count)."""
    from fastmace.model import tiled as tiled_module

    monkeypatch.setattr(tiled_module, "WAVEFRONT_LAYERS", 8)
    monkeypatch.setattr(tiled_module, "_host_available", lambda: available)
    calc, atoms = calculator(), big_structure().repeat((4, 1, 1))
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, rewrites=only(*MH1_PACKAGE))
    tiled = TiledForceField(field, TileConfig(12, neighbours="tiles", state="host", alias_state=True,
                                              list_reuse=False, list_reuse_floor=2**21))
    tiled.evaluate(atoms)
    assert tiled.last["wavefront"]["on"] is on
    assert tiled._wave_auto == {len(atoms): on}


@pytest.mark.parametrize("short", [False, True])
def test_wavefront_auto_below_its_layers_only_where_the_plain_state_does_not_fit_at_all(short, monkeypatch):
    """Below ``WAVEFRONT_LAYERS`` grid layers "auto" stays off while the plain state fits the host (only the floor
    is short), and runs from ``WAVEFRONT_MIN_LAYERS`` when it does not fit at all."""
    from fastmace.model import tiled as tiled_module

    monkeypatch.setattr(tiled_module, "WAVEFRONT_LAYERS", 100)
    calc, atoms = calculator(), big_structure().repeat((4, 1, 1))
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, rewrites=only(*MH1_PACKAGE))
    tiled = TiledForceField(field, TileConfig(12, neighbours="tiles", state="host", alias_state=True,
                                              list_reuse=False, list_reuse_floor=2**40))
    plain = 1.25 * len(atoms) * tiled._state_width() * (8 if str(tiled.dtype) == "torch.float64" else 4)
    available = int(plain / 2) if short else int(plain * 2)  # below the floor either way
    monkeypatch.setattr(tiled_module, "_host_available", lambda: available)
    tiled.evaluate(atoms)
    assert tiled._wave_auto == {len(atoms): True} and tiled._wave_short == {len(atoms): short}
    assert tiled.last["wavefront"]["on"] is short


@pytest.mark.parametrize("chunk", [1, 3, 7, 1000])
def test_the_skin_check_by_chunks_decides_as_on_the_whole_arrays(chunk):
    """``moved_within``: the list reuse's displacement test by chunks of atoms gives the whole arrays' decision, at
    the limit itself too (a displacement equal to the limit keeps the lists)."""
    import torch

    from fastmace.model.tiled import moved_within

    generator = torch.Generator().manual_seed(3)
    reference = torch.rand(20, 3, generator=generator, dtype=torch.float64) * 10
    for index in (0, 6, 19):
        for step in (0.25, 0.5, 0.5000001, 2.0):
            moved = reference.clone()
            moved[index, 1] += step
            for positions in (moved, moved.to(torch.float32)):
                whole = float(((positions.to(torch.float64) - reference) ** 2).sum(dim=1).max()) <= 0.25
                assert moved_within(positions, reference, 0.5, chunk) is whole
    assert moved_within(reference, reference, 0.0, chunk)

def test_senders_packed_as_ranks_into_the_tiles_senders_unpack_exactly():
    """``_pack_senders``: int16 ranks into the tile's distinct int32 senders while they number at most 65,536 (the
    full range of the ranks used), else the int32 senders; both unpack to the same int64 senders, in order."""
    import torch

    from fastmace.model.tiled import _pack_senders, _unpack_senders

    generator = torch.Generator().manual_seed(5)
    for distinct, packed_dtype in ((1, torch.int16), (2**16, torch.int16), (2**16 + 1, torch.int32)):
        pool = torch.randperm(10 * distinct, generator=generator)[:distinct] + 2**30
        sender = torch.cat([pool, pool[torch.randint(distinct, (500,), generator=generator)]])
        sender = sender[torch.randperm(sender.shape[0], generator=generator)]
        dictionary, codes = _pack_senders(sender)
        assert codes.dtype == packed_dtype
        assert torch.equal(_unpack_senders(dictionary, codes), sender)
    dictionary, codes = _pack_senders(torch.empty(0, dtype=torch.int64))
    assert _unpack_senders(dictionary, codes).shape == (0,)


def test_wavefront_with_and_without_kept_lists_is_the_same_call():
    """The wavefront with the kept lists (``list_reuse``, filtered again at the cutoff) and without them, each with
    and without the call's cache (a tile's lists from its sweep 1 to its sweep 3 only: none left after the call;
    with kept lists the filtered ones, so sweeps 2 and 3 filter none), gives bitwise the same call."""
    calc, atoms = calculator(), big_structure().repeat((4, 1, 1))
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*MH1_PACKAGE))
    arms = [(False, True), (True, True), (True, False)]  # (list_reuse, list_cache)
    results = []
    for reuse, cache in arms:
        tiled = TiledForceField(field, TileConfig(12, mode="padded", neighbours="tiles", state="host",
                                                  alias_state=True, wavefront="on", wavefront_group=2,
                                                  list_reuse=reuse, list_cache=cache, boundary_skin=False))
        tiled.evaluate(atoms, stress=True)  # TorchScript's first call in a process rounds differently
        for current in (strained(atoms), atoms):
            results.append(tiled.evaluate(current, stress=True))
            assert tiled.last["list_cache_bytes"] == 0
            assert (tiled.last["list_kept_bytes"] > 0) == reuse
            assert (tiled.last["list_cache_hits"] == 2 * tiled.last["tiles"]) == cache  # sweeps 2 and 3
    for arm in range(1, len(arms)):
        for one, two in zip(results[:2], results[2 * arm:2 * arm + 2]):
            assert one[0] == two[0]
            assert np.array_equal(np.asarray(two[1]), np.asarray(one[1]))
            assert np.array_equal(np.asarray(two[2]), np.asarray(one[2]))


def test_wavefront_band_cache_of_kept_lists_after_motion_is_a_fresh_search(monkeypatch):
    """The band cache of the filtered kept lists (wavefront, ``list_reuse``): on a call that reuses the kept lists
    after a move within skin / 2 (pairs cross the cutoff), sweeps 2 and 3 read the tiles' lists filtered at the
    current positions in sweep 1, bitwise a fresh search on the same tiles; the next call filters again. With the
    host below the floor on that call, the cache is dropped and the kept lists are still reused, bitwise the same."""
    from fastmace.model import tiled as tiled_module

    calc, atoms = calculator(), big_structure().repeat((4, 1, 1))
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*MH1_PACKAGE))
    # state_dtype None: with no host room "auto" would store the state in float16 (not this test's subject)
    config = dict(mode="padded", neighbours="tiles", state="host", alias_state=True, wavefront="on",
                  wavefront_group=2, boundary_skin=False, state_dtype=None)
    cached = TiledForceField(field, TileConfig(12, list_reuse=True, **config))
    starved = TiledForceField(field, TileConfig(12, list_reuse=True, **config))
    fresh = TiledForceField(field, TileConfig(12, list_reuse=False, **config))
    x0 = field._prepare_one_off(atoms).clone()
    rng = np.random.default_rng(11)
    direction = torch.tensor(rng.normal(size=(len(atoms), 3)), dtype=x0.dtype)
    direction /= direction.norm(dim=1, keepdim=True)
    moves = [x0 + 0.4 * field.neighbours.skin * direction, x0 - 0.3 * field.neighbours.skin * direction]
    for tiled in (cached, starved, fresh):
        tiled(x0)  # TorchScript's first call in a process rounds differently
    edges0 = cached.last["edges"]
    tiles = [t.to(x0.device) for t in cached._kept["tiles"]]
    original = tiled_module.plan_tiles
    for x in moves:
        got = cached(x)
        assert cached.last["list_reused"] is True and cached.last["list_cache"] == "on"
        assert cached.last["list_cache_hits"] == 2 * cached.last["tiles"] and cached.last["list_cache_bytes"] == 0
        with monkeypatch.context() as patch:
            patch.setattr(tiled_module, "_host_available", lambda: 0)  # below the floor while the cache fills
            low = starved(x)
        assert starved.last["list_reused"] is True and starved._kept is not None
        assert starved.last["list_cache"].startswith("off: ") and starved.last["list_cache_hits"] == 0
        with monkeypatch.context() as patch:
            patch.setattr(tiled_module, "plan_tiles", lambda *a, **k: (tiles, cached._kept["grid"]))
            expected = fresh(x)
        assert tiled_module.plan_tiles is original
        assert fresh.last["edges"] == cached.last["edges"] == starved.last["edges"]
        for one in (got, low):
            assert torch.equal(one.energy, expected.energy) and torch.equal(one.forces, expected.forces)
            assert torch.equal(one.stress, expected.stress)
        assert cached.last["edges"] != edges0  # pairs crossed the cutoff


def test_boundary_skin_plan_splits_the_first_and_last_layer_at_the_reach():
    """``plan_tiles(reach=...)``: along the first axis, layer 0 holds the atoms within ``reach`` of the periodic
    boundary from below, the last layer those within ``reach`` of it from above, the k layers between them the
    plain layers less the skins; every atom once."""
    rng = np.random.default_rng(1)
    n, reach = 3000, 2.5
    cell = torch.tensor(np.diag([60.0, 12.0, 12.0]) + rng.uniform(-0.5, 0.5, (3, 3)), dtype=torch.float64)
    frac = torch.tensor(rng.uniform(0, 1, (n, 3)), dtype=torch.float64)
    positions = frac @ cell
    row_ptr = torch.arange(n + 1, dtype=torch.int64) * 10
    plain, grid = plan_tiles(positions, cell, row_ptr, 200, 10**9)
    tiles, split = plan_tiles(positions, cell, row_ptr, 200, 10**9, reach=reach)
    assert split == (grid[0] + 2, grid[1], grid[2]) and grid[0] >= 4
    assert sorted(torch.cat(tiles).tolist()) == list(range(n))
    spacing = 1.0 / float(torch.linalg.norm(torch.linalg.inv(cell).T[0]))
    skin = reach / spacing
    first = frac[:, 0]
    low = torch.cat([t for t in tiles if bool(torch.all(first[t] < skin))])
    high = torch.cat([t for t in tiles if bool(torch.all(first[t] >= 1 - skin))])
    assert sorted(low.tolist()) == torch.nonzero(first < skin).flatten().tolist()
    assert sorted(high.tolist()) == torch.nonzero(first >= 1 - skin).flatten().tolist()
    assert len(tiles) >= len(plain)
    # Without the reach, or with a reach above half a layer, the plan is the plain one.
    assert plan_tiles(positions, cell, row_ptr, 200, 10**9, reach=0.6 * spacing / grid[0])[1] == grid


def test_wavefront_with_the_boundary_skin_is_the_same_call_in_another_tile_order():
    """``TileConfig.boundary_skin``: the wavefront's plan splits the first and last layer at the cutoff (the tiles'
    reach without kept lists); another tile plan, so the energy, forces and stress agree to the summation order, and
    the row pool holds no more rows than without the split."""
    calc, atoms = calculator(), big_structure().repeat((4, 1, 1))  # 8 tile layers of 8.4 A at 32 receivers
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*MH1_PACKAGE))
    results, reports = [], []
    for skin in (False, True):
        tiled = TiledForceField(field, TileConfig(32, mode="padded", neighbours="tiles", state="host",
                                                  alias_state=True, wavefront="on", wavefront_group=2,
                                                  list_reuse=False, boundary_skin=skin))
        tiled.evaluate(atoms, stress=True)  # TorchScript's first call in a process rounds differently
        results.append(tiled.evaluate(strained(atoms), stress=True))
        reports.append(tiled.last)
    off, on = reports
    assert off["wavefront"]["boundary_skin"] == 0.0
    assert on["wavefront"]["boundary_skin"] == pytest.approx(float(field.neighbours.cutoff))
    assert on["grid"] == (off["grid"][0] + 2, *off["grid"][1:])
    assert on["wavefront"]["pool_rows"] <= off["wavefront"]["pool_rows"]
    assert results[1][0] == pytest.approx(results[0][0], rel=1e-13, abs=0)
    assert relative(results[1][1], results[0][1]) <= 1e-12
    assert relative(results[1][2], results[0][2]) <= 1e-12


def test_wavefront_needs_the_aliased_state():
    with pytest.raises(ValueError, match="wavefront"):
        TileConfig(5, state="host", wavefront="on")
    with pytest.raises(ValueError, match="wavefront"):
        TileConfig(5, wavefront="sometimes")


# ---------------------------------------------------------------- two lanes for sweep 2


@pytest.mark.parametrize("wave", [False, True], ids=["three-sweeps", "wavefront"])
def test_lane_steps_pair_sweep_two_after_what_it_reads(wave):
    """``lane_steps``: every sweep of every tile once; sweep 2 in tile order, in pairs of consecutive tiles; a pair
    after the sweep 1 of every tile either of its tiles needs, a sweep 3 after the pair holding its ``ready`` tile;
    the peak is the most receivers between their sweeps 1 and 3."""
    from fastmace.model.tiled import lane_steps, wavefront_schedule

    tiles = 9  # odd: the last sweep-2 tile has no partner
    needs = [sorted({(t - 1) % tiles, t, (t + 1) % tiles}) for t in range(tiles)]
    ready = [tiles - 1 if t == 0 else min(t + 1, tiles - 1) for t in range(tiles)]
    counts = [3 + t % 2 for t in range(tiles)]
    if wave:
        steps, _ = wavefront_schedule(needs, ready, counts)
    else:
        steps = [(s, t) for s in (1, 2, 3) for t in range(tiles)]
    items, peak = lane_steps(steps, counts)
    flat = [(item[0], t) for item in items for t in item[1:]]
    assert sorted(flat) == sorted((s, t) for s in (1, 2, 3) for t in range(tiles))
    pairs = [item[1:] for item in items if item[0] == 2]
    assert [t for pair in pairs for t in pair] == list(range(tiles))
    assert all(len(pair) == 2 for pair in pairs[:-1]) and len(pairs[-1]) == 1
    position = {}
    for i, item in enumerate(items):
        for t in item[1:]:
            position[(item[0], t)] = i
    for t in range(tiles):
        if wave:
            assert all(position[(1, u)] < position[(2, t)] for u in needs[t])
            assert position[(3, t)] > position[(2, ready[t])]
        else:
            assert position[(1, tiles - 1)] < position[(2, t)] < position[(3, 0)]
    live = most = 0
    for sweep, t in flat:
        live += counts[t] if sweep == 1 else -counts[t] if sweep == 3 else 0
        most = max(most, live)
    assert peak == most


@pytest.mark.parametrize("wave", ["off", "on"])
def test_lanes_call_equals_one_lane(wave):
    """``TileConfig.lanes="on"`` off a CUDA device runs sweep 2's pairs one after the other, the second tile of a pair
    on the second lane's own padded slot and strain gradient: the node energies (and the energy) bitwise one lane's,
    the forces and the stress the same sums in another order."""
    calc, atoms = calculator(), big_structure().repeat((4, 1, 1))
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*MH1_PACKAGE))
    results = []
    for lanes in ("off", "on"):
        tiled = TiledForceField(field, TileConfig(12, mode="padded", neighbours="tiles", state="host",
                                                  alias_state=True, wavefront=wave, wavefront_group=3, lanes=lanes,
                                                  list_reuse=False))
        tiled.evaluate(atoms, stress=True)  # TorchScript's first call in a process rounds differently
        for current in (strained(atoms), atoms):
            results.append(tiled.evaluate(current, stress=True))
        assert tiled.last["lanes"] == ("on: one after the other (no CUDA device)" if lanes == "on" else "off")
        assert tiled.last["wavefront"]["on"] == (wave == "on")
        assert not tiled._g["state"]._pending
    assert tiled._lane_slots and all(slot is not tiled._slots[key] for key, slot in tiled._lane_slots.items())
    assert tiled.last["tiles"] > 8
    for one, two in zip(results[:2], results[2:]):
        assert one[0] == two[0]
        assert relative(two[1], one[1]) <= 1e-13
        assert relative(two[2], one[2]) <= 1e-13


@pytest.mark.parametrize("wave, lanes", [("off", "off"), ("on", "off"), ("on", "on")],
                         ids=["three-sweeps", "wavefront", "wavefront-lanes"])
def test_prefetch_call_is_bitwise_the_call_without(wave, lanes, monkeypatch):
    """``TileConfig.prefetch`` builds the next tile's graph and gathers its host rows ahead: the same values summed in
    the same order, so the energy, the forces and the stress are bitwise those without it; tiles read rows that
    were prefetched."""
    from fastmace.model import tiled as tiled_module

    calc, atoms = calculator(), big_structure().repeat((4, 1, 1))
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*MH1_PACKAGE))
    taken = []
    rows = tiled_module._State.rows

    def counting(self, name, index):
        taken.append(name in self.prefetched)
        return rows(self, name, index)

    monkeypatch.setattr(tiled_module._State, "rows", counting)
    results = []
    for prefetch in (False, True):
        tiled = TiledForceField(field, TileConfig(12, mode="padded", neighbours="tiles", state="host",
                                                  alias_state=True, wavefront=wave, wavefront_group=3, lanes=lanes,
                                                  list_reuse=False,
                                                  prefetch=prefetch))
        tiled.evaluate(atoms, stress=True)  # TorchScript's first call in a process rounds differently
        taken.clear()
        for current in (strained(atoms), atoms):
            results.append(tiled.evaluate(current, stress=True))
        assert any(taken) == prefetch
        assert not tiled._g["state"].prefetched and not tiled._g["state"]._pending
    for one, two in zip(results[:2], results[2:]):
        assert one[0] == two[0]
        assert np.array_equal(np.asarray(two[1]), np.asarray(one[1]))
        assert np.array_equal(np.asarray(two[2]), np.asarray(one[2]))


@pytest.mark.parametrize("mode", ["eager", "padded"])
def test_virial_from_edge_gradients_is_the_strain_leafs_gradient(mode, monkeypatch):
    """``TILE_VIRIAL``: sweeps 2 and 3 take the strain gradient as sym(V^T dE/dV) of their edges instead of a strain
    leaf in the graph: the energy and the forces bitwise (eps = 0 adds exact zeros), the stress the same to float64
    rounding, and the whole-system call's."""
    from fastmace.model import tiled as tiled_module

    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*MH1_PACKAGE))
    results = {}
    for virial in (False, True):
        monkeypatch.setattr(tiled_module, "TILE_VIRIAL", virial)
        tiled = TiledForceField(field, TileConfig(9, mode=mode))
        tiled.evaluate(atoms, stress=True)  # TorchScript's first call in a process rounds differently
        results[virial] = [tiled.evaluate(current, stress=True) for current in (strained(atoms), atoms)]
    for current, leaf, edges in zip((strained(atoms), atoms), results[False], results[True]):
        assert edges[0] == leaf[0]
        assert np.array_equal(np.asarray(edges[1]), np.asarray(leaf[1]))
        assert relative(edges[2], leaf[2]) <= 1e-13
        assert relative(edges[2], field.evaluate(current, stress=True)[2]) <= 1e-11


@pytest.mark.parametrize("virial", [False, True], ids=["strain-leaf", "virial"])
@pytest.mark.parametrize("mode", ["eager", "padded"])
def test_epilogue_is_bitwise_the_eager_epilogue_outside_compiled_mode(mode, virial, monkeypatch):
    """``TILE_EPILOGUE``: sweep 1's masks and sweep 3's objective inside the forwards, and the sweeps' edge masks and
    virial as one epilogue, run as the same products outside mode "compiled": energy, forces and stress bitwise the
    call without (with the strain leaf and with ``TILE_VIRIAL``)."""
    from fastmace.model import tiled as tiled_module

    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*MH1_PACKAGE))
    monkeypatch.setattr(tiled_module, "TILE_VIRIAL", virial)
    results = {}
    for fused in (False, True):
        monkeypatch.setattr(tiled_module, "TILE_EPILOGUE", fused)
        tiled = TiledForceField(field, TileConfig(9, mode=mode))
        tiled.evaluate(atoms, stress=True)  # TorchScript's first call in a process rounds differently
        results[fused] = [tiled.evaluate(current, stress=True) for current in (strained(atoms), atoms)]
        assert tiled.last["tiles"] > 1
    for plain, fused in zip(results[False], results[True]):
        assert fused[0] == plain[0]
        assert np.array_equal(np.asarray(fused[1]), np.asarray(plain[1]))
        assert np.array_equal(np.asarray(fused[2]), np.asarray(plain[2]))


def test_virial_drops_the_padding_edges_by_their_mask():
    """``_virial``: sym(V^T G) over the edges whose mask is 1; a padding edge's vector and gradient add nothing."""
    from fastmace.model.tiled import _virial

    rng = np.random.default_rng(1)
    v = torch.from_numpy(rng.normal(size=(40, 3)))
    g = torch.from_numpy(rng.normal(size=(40, 3)))
    mask = torch.ones(40, dtype=torch.float64)
    mask[-7:] = 0
    expected = v[:-7].T @ g[:-7]
    expected = 0.5 * (expected + expected.T)
    assert torch.allclose(_virial(v, g, mask), expected, rtol=1e-14, atol=1e-14)
    assert torch.allclose(_virial(v[:-7], g[:-7]), expected, rtol=1e-14, atol=1e-14)


def test_prefetch_with_the_state_on_the_device_is_bitwise_the_call_without(monkeypatch):
    """``TILE_PREFETCH_DEVICE``: with the state on the device, prefetch builds the next tile's graph ahead (no rows):
    the same graphs, so the call is bitwise the one without; off, the device state builds none ahead."""
    from fastmace.model import tiled as tiled_module

    calc, atoms = calculator(), big_structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*MH1_PACKAGE))
    events = []
    for name in ("_tile_graph", "_fill", "_scatter"):
        method = getattr(tiled_module.TiledForceField, name)

        def recording(self, *args, _method=method, _name=name, **kwargs):
            events.append(_name)
            return _method(self, *args, **kwargs)

        monkeypatch.setattr(tiled_module.TiledForceField, name, recording)
    results, counts = {}, {}
    for ahead in (False, True):
        monkeypatch.setattr(tiled_module, "TILE_PREFETCH_DEVICE", ahead)
        tiled = TiledForceField(field, TileConfig(12, mode="padded", neighbours="tiles", list_reuse=False))
        tiled.evaluate(atoms, stress=True)  # TorchScript's first call in a process rounds differently
        events.clear()
        results[ahead] = [tiled.evaluate(current, stress=True) for current in (strained(atoms), atoms)]
        counts[ahead] = events.count("_tile_graph")
        # Built ahead: the next tile's graph between a tile's fill (and body) and its scatter.
        assert any(pair == ("_tile_graph", "_scatter") for pair in zip(events, events[1:])) == ahead
        assert not tiled._g["state"].prefetched
    assert counts[True] == counts[False]  # each tile's graph built as often, ahead or not
    for one, two in zip(results[False], results[True]):
        assert one[0] == two[0]
        assert np.array_equal(np.asarray(two[1]), np.asarray(one[1]))
        assert np.array_equal(np.asarray(two[2]), np.asarray(one[2]))


@pytest.mark.parametrize("mode, budget", [("eager", 10**12), ("padded", 10**12), ("padded", 30000)],
                         ids=["eager", "padded", "padded-partly"])
def test_graph_cache_is_bitwise_the_call_that_builds_every_graph(mode, budget, monkeypatch):
    """``TILE_GRAPH_CACHE``: each tile's graph from its first build to the end of the call (the slot-key pass, or
    an eager call's sweep 1), at the slot's halo start, gives bitwise the same call; with a budget below the pieces'
    size only some tiles are kept, the rest built again; nothing is kept after the call."""
    from fastmace.model import tiled as tiled_module

    calc, atoms = calculator(), big_structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*MH1_PACKAGE))
    built = []
    graph = tiled_module.TiledForceField._tile_graph

    def counting(self, *args, **kwargs):
        built.append(1)
        return graph(self, *args, **kwargs)

    monkeypatch.setattr(tiled_module.TiledForceField, "_tile_graph", counting)
    monkeypatch.setattr(tiled_module, "TILE_GRAPH_CACHE", 0.25)  # None: off the speed GPUs (and off CUDA)
    results, counts = {}, {}
    for cached in (False, True):
        monkeypatch.setattr(tiled_module.TiledForceField, "_free_bytes",
                            lambda self, cached=cached: budget / 0.25 if cached else None)
        tiled = TiledForceField(field, TileConfig(12, mode=mode, neighbours="tiles", list_reuse=False))
        tiled.evaluate(atoms, stress=True)  # TorchScript's first call in a process rounds differently
        results[cached], counts[cached] = [], []
        for current in (strained(atoms), atoms):
            built.clear()
            results[cached].append(tiled.evaluate(current, stress=True))
            counts[cached].append((len(built), tiled.last["tiles"]))
            kept = tiled.last["graph_cache"]
            assert (kept["tiles"] > 0) == cached and kept["bytes"] <= kept["budget"]
            assert tiled._pieces is None
    for (plain, tiles), (fewer, same) in zip(counts[False], counts[True]):
        assert same == tiles and plain == tiles * (4 if mode == "padded" else 3)
        if budget > 10**9:
            assert fewer == tiles  # one build per tile and call
        else:
            assert tiles < fewer < plain
    for one, two in zip(results[False], results[True]):
        assert one[0] == two[0]
        assert np.array_equal(np.asarray(two[1]), np.asarray(one[1]))
        assert np.array_equal(np.asarray(two[2]), np.asarray(one[2]))


def test_speed_switches_left_at_none_are_on_only_for_the_speed_gpus(monkeypatch):
    """``TILE_VIRIAL``, ``TILE_PREFETCH_DEVICE`` and ``TILE_GRAPH_CACHE`` at None (the defaults) are on only on
    ``TILED_SPEED_GPUS`` (decision 66), the graph cache at ``TILE_GRAPH_CACHE_FRACTION``; a set value wins anywhere."""
    from fastmace.model import tiled as tiled_module

    assert tiled_module.TILE_VIRIAL is None and tiled_module.TILE_PREFETCH_DEVICE is None
    assert tiled_module.TILE_EPILOGUE is None and tiled_module.TILE_FILTER32 is None
    assert tiled_module.TILE_GRAPH_CACHE is None and "NVIDIA GeForce RTX 5060 Ti" in tiled_module.TILED_SPEED_GPUS
    for name in ("NVIDIA RTX 4000 Ada Generation", "NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"):  # 10 Nov A/Bs
        assert name in tiled_module.TILED_SPEED_GPUS
    assert "NVIDIA H200" not in tiled_module.TILED_SPEED_GPUS  # not measured
    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*MH1_PACKAGE))
    tiled = TiledForceField(field, TileConfig(5))
    assert tiled._speed_gpu is False  # a CPU device
    assert tiled._speed(None) is False and tiled._speed(None, 0.25) == 0.0
    assert tiled._speed(True) is True and tiled._speed(0.5, 0.25) == 0.5
    tiled._speed_gpu = True
    assert tiled._speed(None) is True and tiled._speed(None, 0.25) == 0.25 and tiled._speed(False) is False


def test_tile_block_rows_follows_the_module_unless_the_constant_is_set(monkeypatch):
    """``TILE_BLOCK_ROWS`` at None (the default) is the module's ``tile_block_rows``, on where nothing set it (the
    tiled call no longer sets it: on under the wavefront too since the host-state worker); True / False win
    everywhere."""
    import types

    from fastmace.rewrites.generic import modules as generic

    assert generic.TILE_BLOCK_ROWS is None
    unset, off = types.SimpleNamespace(), types.SimpleNamespace(tile_block_rows=False)
    assert generic.tile_block_rows(unset) is True and generic.tile_block_rows(off) is False
    monkeypatch.setattr(generic, "TILE_BLOCK_ROWS", True)
    assert generic.tile_block_rows(off) is True
    monkeypatch.setattr(generic, "TILE_BLOCK_ROWS", False)
    assert generic.tile_block_rows(unset) is False


@pytest.mark.parametrize("budget", [10**12, 3_000], ids=["all-kept", "some-kept"])
def test_fill_skip_leaves_the_graph_caches_tiles_out_of_the_list_cache(budget, monkeypatch):
    """``TILE_FILL_SKIP``: a tile the graph cache takes is not packed into the host list cache; the call is bitwise
    the call that fills it, also when a tile built for the cache did not fit and is built again."""
    from fastmace.model import tiled as tiled_module

    calc, atoms = calculator(), big_structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*MH1_PACKAGE))
    monkeypatch.setattr(tiled_module, "TILE_GRAPH_CACHE", 0.25)
    monkeypatch.setattr(tiled_module.TiledForceField, "_free_bytes", lambda self: budget / 0.25)
    results, filled = {}, {}
    for skip in (False, True):
        monkeypatch.setattr(tiled_module, "TILE_FILL_SKIP", skip)
        tiled = TiledForceField(field, TileConfig(12, mode="padded", neighbours="tiles", list_reuse=False))
        tiled.evaluate(atoms, stress=True)  # TorchScript's first call in a process rounds differently
        results[skip] = [tiled.evaluate(current, stress=True) for current in (strained(atoms), atoms)]
        filled[skip] = tiled.last["list_cache_bytes"]
        assert tiled.last["graph_cache"]["tiles"] > 0
    assert filled[False] > 0
    assert (filled[True] == 0) if budget > 10**9 else (0 <= filled[True] < filled[False])
    for one, two in zip(results[False], results[True]):
        assert one[0] == two[0]
        assert np.array_equal(np.asarray(two[1]), np.asarray(one[1]))
        assert np.array_equal(np.asarray(two[2]), np.asarray(one[2]))


@pytest.mark.parametrize("budget", [10**12, 3_000], ids=["all-kept", "some-kept"])
def test_graph_cache_under_the_wavefront_is_bitwise_the_call_that_builds_every_graph(budget, monkeypatch):
    """``TILE_GRAPH_CACHE_WAVE``: the wavefront keeps each tile's graph from its search pass to its sweep 3 (then
    drops it), bitwise the call that builds every graph four times; with a small budget only some tiles are kept."""
    from fastmace.model import tiled as tiled_module

    calc, atoms = calculator(), big_structure().repeat((4, 1, 1))
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*MH1_PACKAGE))
    built = []
    graph = tiled_module.TiledForceField._tile_graph

    def counting(self, *args, **kwargs):
        built.append(1)
        return graph(self, *args, **kwargs)

    monkeypatch.setattr(tiled_module.TiledForceField, "_tile_graph", counting)
    monkeypatch.setattr(tiled_module, "TILE_GRAPH_CACHE", 0.25)
    monkeypatch.setattr(tiled_module.TiledForceField, "_free_bytes", lambda self: budget / 0.25)
    results, counts = {}, {}
    for wave_cache in (False, True):
        monkeypatch.setattr(tiled_module, "TILE_GRAPH_CACHE_WAVE", wave_cache)
        tiled = TiledForceField(field, TileConfig(12, mode="padded", neighbours="tiles", state="host",
                                                  alias_state=True, wavefront="on", wavefront_group=2,
                                                  boundary_skin=False))
        tiled.evaluate(atoms, stress=True)  # TorchScript's first call in a process rounds differently
        results[wave_cache], counts[wave_cache] = [], []
        for current in (strained(atoms), atoms):
            built.clear()
            results[wave_cache].append(tiled.evaluate(current, stress=True))
            counts[wave_cache].append((len(built), tiled.last["tiles"]))
            assert (tiled.last["graph_cache"]["tiles"] > 0) == wave_cache and tiled._pieces is None
    for (plain, tiles), (fewer, same) in zip(counts[False], counts[True]):
        assert same == tiles and plain == 4 * tiles
        assert fewer == tiles if budget > 10**9 else tiles < fewer < plain
    for one, two in zip(results[False], results[True]):
        assert one[0] == two[0]
        assert np.array_equal(np.asarray(two[1]), np.asarray(one[1]))
        assert np.array_equal(np.asarray(two[2]), np.asarray(one[2]))


@pytest.mark.parametrize("wave", [False, True], ids=["host-state", "wavefront"])
def test_host_worker_is_bitwise_the_call_that_writes_on_the_calling_thread(wave, monkeypatch):
    """``TILE_HOST_WORKER``: host state with overlap applies each tile's deferred writes on a worker thread, every
    read waiting for the batches handed over before it; bitwise the call that applies them itself."""
    from fastmace.model import tiled as tiled_module

    calc, atoms = calculator(), big_structure().repeat((4, 1, 1))
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*MH1_PACKAGE))
    config = dict(mode="padded", neighbours="tiles", state="host", alias_state=wave, boundary_skin=False)
    if wave:
        config.update(wavefront="on", wavefront_group=2)
    results = {}
    for worker in (False, True):
        monkeypatch.setattr(tiled_module, "TILE_HOST_WORKER", worker)
        tiled = TiledForceField(field, TileConfig(12, **config))
        tiled.evaluate(atoms, stress=True)  # TorchScript's first call in a process rounds differently
        results[worker] = [tiled.evaluate(current, stress=True) for current in (strained(atoms), atoms)]
        assert tiled.last["host_worker"] is worker and tiled.last["tiles"] > 2
        assert tiled.last["copy_stream"] is False  # TILE_COPY_STREAM: CUDA only
        state = tiled._g["state"]
        assert state._handed == [] and state._pending == []
        assert (state._executor is not None) == worker
    for one, two in zip(results[False], results[True]):
        assert one[0] == two[0]
        assert np.array_equal(np.asarray(two[1]), np.asarray(one[1]))
        assert np.array_equal(np.asarray(two[2]), np.asarray(one[2]))


@pytest.mark.parametrize("wave", [False, True], ids=["alias", "wavefront"])
@pytest.mark.parametrize("store", ["bfloat16", "float16"])
def test_state_dtype_stores_the_host_state_in_16_bits_and_rounds_only_it(store, wave):
    """``TileConfig.state_dtype``: x, sc and their aliased adjoints on the host in 16 bits (half the bytes; the alias
    window stays in the call's dtype), each value rounded once where it is written. The call stays within the
    stored values' rounding of the call that keeps them in its own dtype."""
    calc, atoms = calculator(), big_structure().repeat((4, 1, 1))
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*MH1_PACKAGE))
    config = dict(mode="padded", neighbours="tiles", state="host", alias_state=True, boundary_skin=False,
                  wavefront="on" if wave else "off", wavefront_group=2)
    results, sizes = {}, {}
    for dtype in (None, store):
        tiled = TiledForceField(field, TileConfig(12, state_dtype=dtype, **config))
        tiled.evaluate(atoms, stress=True)
        results[dtype] = tiled.evaluate(strained(atoms), stress=True)
        state = tiled._g["state"]
        assert {str(t.dtype) for t in state.tensors.values()} == {f"torch.{dtype}" if dtype else str(tiled.dtype)}
        sizes[dtype] = sum(t.numel() * t.element_size() for t in state.tensors.values())
        assert state._window is None or state._window.dtype == tiled.dtype
    assert sizes[store] * (4 if str(tiled.dtype) == "torch.float64" else 2) == sizes[None]
    (e0, f0, s0), (e1, f1, s1) = results[None], results[store]
    f0, f1, s0, s1 = (np.asarray(v, dtype=np.float64) for v in (f0, f1, s0, s1))
    bound = 2.0**-6 if store == "bfloat16" else 2.0**-9  # a few units of the stored format's rounding
    assert f1.shape == f0.shape and np.abs(f1 - f0).max() <= bound * np.abs(f0).max()
    assert np.abs(s1 - s0).max() <= bound * max(np.abs(s0).max(), 1e-12)
    assert abs(e1 - e0) <= bound * abs(e0)
    assert not np.array_equal(f1, f0)  # the stored values were rounded


def test_wavefront_runs_on_the_force_fields_own_list():
    """The wavefront with ``neighbours="global"`` (the force/stress gate's tiled production): its first call has no
    tile list cache to drop lists from (an AttributeError before); the energy is bitwise the plain sweeps'."""
    calc, atoms = calculator(), big_structure().repeat((4, 1, 1))
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*MH1_PACKAGE))
    results = []
    for wave in ("off", "on"):
        tiled = TiledForceField(field, TileConfig(12, mode="padded", state="host", alias_state=True, wavefront=wave,
                                                  wavefront_group=2, boundary_skin=False))
        results.append(tiled.evaluate(atoms, stress=True))
    (e0, f0, s0), (e1, f1, s1) = results
    assert e1 == e0
    assert np.allclose(np.asarray(f1), np.asarray(f0), rtol=0, atol=1e-10)
    assert np.allclose(np.asarray(s1), np.asarray(s0), rtol=0, atol=1e-10)


def test_state_dtype_auto_drops_the_kept_lists_before_it_rounds_the_state(monkeypatch):
    """``TileConfig.state_dtype="auto"`` (the default): the call's dtype while the host state (the pool's rows and
    a quarter for the alias window) fits above the floor; else, when that makes it fit, the kept lists go (bitwise
    the same call, searched again); else float16."""
    from fastmace.model import tiled as tiled_module

    calc, atoms = calculator(), big_structure().repeat((4, 1, 1))
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*MH1_PACKAGE))
    floor = 2**20
    config = dict(mode="padded", neighbours="tiles", state="host", alias_state=True, boundary_skin=False,
                  wavefront="on", wavefront_group=2, list_reuse_floor=floor)

    def run(room, **extra):
        monkeypatch.setattr(tiled_module, "_host_available", lambda: floor + int(room))
        tiled = TiledForceField(field, TileConfig(12, **dict(config, **extra)))
        return tiled, tiled.evaluate(atoms, stress=True)

    tiled, plain = run(10**15)
    assert TileConfig(5).state_dtype == "auto" and tiled.last["state_dtype"] == "float64"
    assert tiled.last["state_dtype_reason"] == "the call's dtype: fits" and tiled._kept is not None
    state = tiled._g["state"]
    rows, width = state._pool_rows, sum(int(t.shape[1]) for t in state.tensors.values())
    full, half = 1.25 * rows * width * 8, rows * width * 2 + 0.25 * rows * width * 8
    lists = tiled.last["list_kept_bytes"]
    assert lists > 0
    tiled, dropped = run(full - lists / 2)
    assert tiled.last["state_dtype"] == "float64" and tiled._kept is None
    assert tiled.list_reuse_state.startswith("off: the host state")
    assert dropped[0] == plain[0] and np.array_equal(np.asarray(dropped[1]), np.asarray(plain[1]))
    tiled, rounded = run(half + 1, list_reuse=False)
    assert tiled.last["state_dtype"] == "float16" and tiled.last["state_dtype_reason"].startswith("float16")
    f0, f1 = np.asarray(plain[1]), np.asarray(rounded[1])
    assert not np.array_equal(f1, f0) and np.abs(f1 - f0).max() <= 2.0**-9 * np.abs(f0).max()


def test_host_room_is_the_memory_cgroups_where_they_limit_it(tmp_path):
    """``_cgroup_room``: a batch job's memory cgroup (v2: the job's limit one level up from the step's, v1: the
    memory controller's) leaves the limit less the usage less the inactive file cache; no limit, no room."""
    from fastmace.model.tiled import _cgroup_room

    root = tmp_path / "v2"
    job, step = root / "slurm" / "job_1", root / "slurm" / "job_1" / "step_0"
    step.mkdir(parents=True)
    (root / "slurm" / "memory.max").write_text("max\n")
    (root / "slurm" / "memory.current").write_text("5\n")
    (job / "memory.max").write_text(f"{120 * 2**30}\n")
    (job / "memory.current").write_text(f"{30 * 2**30}\n")
    (job / "memory.stat").write_text(f"anon 1\ninactive_file {10 * 2**30}\n")
    (step / "memory.max").write_text("max\n")
    (step / "memory.current").write_text(f"{29 * 2**30}\n")
    own = tmp_path / "cgroup-v2"
    own.write_text("0::/slurm/job_1/step_0\n")
    assert _cgroup_room(str(root), str(own)) == 100 * 2**30
    own.write_text("0::/\n")
    assert _cgroup_room(str(root), str(own)) is None
    v1 = tmp_path / "v1" / "memory" / "job_2"
    v1.mkdir(parents=True)
    (v1 / "memory.limit_in_bytes").write_text("1000\n")
    (v1 / "memory.usage_in_bytes").write_text("400\n")
    (v1 / "memory.stat").write_text("total_inactive_file 100\n")
    own.write_text("12:cpuset:/job_2\n11:memory:/job_2\n")
    assert _cgroup_room(str(tmp_path / "v1"), str(own)) == 700
    assert _cgroup_room(str(tmp_path), str(tmp_path / "missing")) is None


def test_state_dtype_needs_the_aliased_host_state():
    with pytest.raises(ValueError, match="state_dtype"):
        TileConfig(5, state_dtype="float32")
    TileConfig(5, state_dtype="auto")  # the default: ignored without the aliased host state
    with pytest.raises(ValueError, match="alias_state"):
        TileConfig(5, state="host", state_dtype="bfloat16")
    TileConfig(5, state="host", alias_state=True, state_dtype="float16")


def test_graph_cache_budget_leaves_the_tiles_workspace_and_stays_out_of_a_measured_call(monkeypatch):
    """``TILE_GRAPH_CACHE``'s budget: none in a call that measures its workspace (the kept graphs would count as
    tile workspace and lower the receiver cap) or under the wavefront; else the fraction of the free memory less
    the tiles' predicted workspace (the measured workspace per receiver times the cap, plus 10 %), never below 0."""
    from fastmace.model import tiled as tiled_module

    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=True,
                           rewrites=only(*MH1_PACKAGE))
    tiled = TiledForceField(field, TileConfig(10))
    monkeypatch.setattr(tiled_module, "TILE_GRAPH_CACHE", 0.25)
    monkeypatch.setattr(tiled_module.TiledForceField, "_free_bytes", lambda self: 10_000)
    tiled._graph_budget(False, measure=True)
    assert tiled._pieces is None and tiled._piece_budget == 0
    tiled._graph_budget(True)
    assert tiled._pieces is None  # TILE_GRAPH_CACHE_WAVE at None: off the speed GPUs
    monkeypatch.setattr(tiled_module, "TILE_GRAPH_CACHE_WAVE", True)
    tiled._graph_budget(True)
    assert tiled._pieces == {}
    monkeypatch.setattr(tiled_module, "TILE_GRAPH_CACHE_WAVE", None)
    tiled._graph_budget(False)  # no workspace measured yet
    assert tiled._pieces == {} and tiled._piece_budget == 2_500
    tiled._per_receiver = 400.0  # 1.1 * 400 * 10 = 4,400 bytes for the tiles
    tiled._graph_budget(False)
    assert tiled._piece_budget == int(0.25 * (10_000 - 1.1 * 400.0 * 10))  # 1,399: 4,400.000000000001 in floats
    tiled._per_receiver = 4_000.0
    tiled._graph_budget(False)
    assert tiled._pieces == {} and tiled._piece_budget == 0


def test_backward_chunk_scale_multiplies_each_modules_own_chunk_once():
    """``TileConfig.backward_chunk_scale``: the chunked backward's edges per chunk times the scale, from each module's
    own value (a second tiled field on the same model does not scale it twice); off CUDA "auto" is 1."""
    with pytest.raises(ValueError, match="backward_chunk_scale"):
        TileConfig(5, backward_chunk_scale=0)
    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, rewrites=only(*MH1_PACKAGE))
    model = getattr(field.model, "_orig_mod", field.model)
    model.interactions[0].chunked_backward_edges = 128
    for scale, edges in ((4, 512), (2, 256), (None, 128)):
        tiled = TiledForceField(field, TileConfig(5, backward_chunk_scale=scale))
        assert model.interactions[0].chunked_backward_edges == edges
        assert tiled.backward_chunk_scale == (scale or 1)


def test_lanes_need_padded_tiles_and_stay_off_without_a_cuda_device():
    with pytest.raises(ValueError, match="lanes"):
        TileConfig(5, mode="eager", lanes="on")
    with pytest.raises(ValueError, match="lanes"):
        TileConfig(5, lanes="sometimes")
    calc, atoms = calculator(), big_structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER, rewrites=only(*MH1_PACKAGE))
    for mode, state in (("padded", "off: no CUDA device"), ("eager", "off: eager tiles")):
        tiled = TiledForceField(field, TileConfig(20, mode=mode))
        tiled.evaluate(atoms)
        assert tiled.last["lanes"] == state


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
