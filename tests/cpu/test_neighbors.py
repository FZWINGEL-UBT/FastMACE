"""Skin neighbour list, pair search and edge buckets on the CPU in float64 (plan task 1a-T09).

Pairs, image shifts and counts are checked against ASE's ``neighbor_list`` on the three test
structures, rattled and strained. ``FASTMACE_BASELINE`` (the exported research-engine tag
``research-engine-baseline-20261017``) adds bitwise comparisons with its ``SkinNeighbourList``.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.neighborlist import neighbor_list

from _support.structures import STRUCTURES, nacl, nbcl5
from fastmace.config import NeighbourConfig
from fastmace.neighbors.buckets import edge_bucket, edge_capacity, round_up_edges
from fastmace.neighbors.pair_search import candidate_pairs
from fastmace.neighbors.skin import SkinNeighbourList, det_3x3, inverse_3x3, times_3x3

CUTOFF = 5.0
F64 = torch.float64
SHEAR = np.array([[1.0, 0.02, 0.0], [0.02, 0.98, 0.01], [0.0, 0.01, 1.015]])


def structures():
    """(id, atoms) for the three structures, rattled, plus strained and minimum-image variants."""
    cases = []
    for name, build in STRUCTURES.items():
        cases.append((f"{name}-rattled", build(rattle=0.05, seed=3)))
        strained = build(rattle=0.05, seed=4)
        strained.set_cell(strained.cell.array @ SHEAR, scale_atoms=True)
        cases.append((f"{name}-strained", strained))
    cases.append(("NbCl5x2-rattled", nbcl5(repeat=2, rattle=0.05, seed=5)))
    return cases


CASES = structures()
CASE_IDS = [name for name, _ in CASES]


def make(atoms, *, skin=0.3, dtype=F64, sort=False, groups=None, variable_cell=False, gpu=False, **config):
    return SkinNeighbourList(atoms.cell.array, CUTOFF, "cpu", dtype, NeighbourConfig(skin=skin, **config),
                             sort_receivers=sort, gpu_build=gpu, sender_groups=groups,
                             variable_cell=variable_cell)


def ase_pairs(atoms, cutoff, positions=None):
    atoms = atoms.copy()
    if positions is not None:
        atoms.positions = positions
    i, j, shifts = neighbor_list("ijS", atoms, cutoff)
    return {(int(a), int(b), *map(int, s)) for a, b, s in zip(i, j, shifts)}


def candidate_set(nl):
    n = nl.num_candidates
    return {(int(i), int(j), *map(int, s)) for (i, j), s in
            zip(nl.candidate_index[:, :n].T.tolist(), nl.candidate_unit_shifts[:n].round().tolist())}


def candidate_shifts(nl, dtype=F64):
    """The candidates' image shifts as compact() computes them (not stored since cap-nlist)."""
    cell = nl.cell_t if nl.variable_cell else nl.build_cell
    return times_3x3(nl.candidate_unit_shifts.to(F64), cell).to(dtype)


def edge_set(nl):
    n = int(nl.count)
    return {(int(i), int(j), *map(int, s)) for (i, j), s in
            zip(nl.edge_index[:, :n].T.tolist(), nl.unit_shifts[:n].round().tolist())}


# --------------------------------------------------------------------------- against ASE


@pytest.mark.parametrize("gpu", [False, True], ids=["matscipy", "torch-build"])
@pytest.mark.parametrize("atoms", [a for _, a in CASES], ids=CASE_IDS)
def test_candidates_and_edges_match_ase(atoms, gpu):
    nl = make(atoms, skin=0.4)
    (nl.build_gpu if gpu else nl.build_cpu)(torch.tensor(atoms.positions))
    assert candidate_set(nl) == ase_pairs(atoms, CUTOFF + 0.4)
    assert nl.num_candidates == len(ase_pairs(atoms, CUTOFF + 0.4))

    moved = atoms.positions + np.random.default_rng(7).uniform(-0.09, 0.09, atoms.positions.shape)
    x = torch.tensor(moved)
    assert not bool(nl.needs_rebuild(x))
    nl.compact(x)
    expected = ase_pairs(atoms, CUTOFF, moved)
    assert edge_set(nl) == expected
    assert int(nl.count) == len(expected)
    n = int(nl.count)
    torch.testing.assert_close(nl.shifts[:n], nl.unit_shifts[:n] @ torch.tensor(atoms.cell.array),
                               rtol=0, atol=1e-12)
    # Everything after `count` lies beyond the cutoff, the padding beyond 2 (cutoff + skin).
    rest = nl.edge_index[:, n:]
    r = torch.linalg.norm(x[rest[1]] - x[rest[0]] + nl.shifts[n:], dim=1)
    assert bool((r >= CUTOFF).all())
    padding = candidate_shifts(nl)[nl.num_candidates:]
    assert len(padding) > 0
    assert bool((torch.linalg.norm(padding, dim=1) > 2 * (CUTOFF + 0.4)).all())
    assert bool((nl.candidate_index[:, nl.num_candidates:] == 0).all())


def test_positions_outside_the_cell():
    atoms = nacl(rattle=0.05, seed=1)
    rng = np.random.default_rng(4)
    positions = atoms.positions + rng.integers(-2, 3, atoms.positions.shape) @ atoms.cell.array
    for gpu in (False, True):
        nl = make(atoms)
        (nl.build_gpu if gpu else nl.build_cpu)(torch.tensor(positions))
        assert candidate_set(nl) == ase_pairs(atoms, CUTOFF + 0.3, positions)
        nl.compact(torch.tensor(positions))
        assert edge_set(nl) == ase_pairs(atoms, CUTOFF, positions)


def test_build_dispatch():
    atoms = nacl(rattle=0.05, seed=2)
    for gpu in (False, True):  # gpu_build only takes effect on a CUDA device
        nl = make(atoms, gpu=gpu)
        assert nl.build(atoms.positions) is True
        assert nl.build(torch.tensor(atoms.positions)) is False
        assert nl.rebuilds == 2
        assert candidate_set(nl) == ase_pairs(atoms, CUTOFF + 0.3)


def test_unresolved_auto_is_refused():
    atoms = nacl()
    with pytest.raises(ValueError, match="sort_receivers"):
        SkinNeighbourList(atoms.cell.array, CUTOFF, "cpu", F64, NeighbourConfig())
    nl = SkinNeighbourList(atoms.cell.array, CUTOFF, "cpu", F64,
                           NeighbourConfig(sort_receivers="on", gpu_neighbour_list="off"))
    assert nl.sort_receivers and not nl.gpu_build


def test_pair_search_cpu_screen():
    atoms = nbcl5(rattle=0.05)
    images = torch.tensor(np.stack(np.meshgrid(*[np.arange(-1, 2)] * 3, indexing="ij"), -1).reshape(-1, 3),
                          dtype=F64)
    vectors = images @ torch.tensor(atoms.cell.array)
    wrapped = torch.tensor(atoms.get_positions(wrap=True))
    image, sender, receiver = candidate_pairs(wrapped, vectors, 4.0)
    d = torch.linalg.norm(wrapped[receiver] - wrapped[sender] + vectors[image], dim=1)
    assert bool((d < 4.0 + 0.011).all())  # the float32 screen keeps its 0.01 Å margin
    found = set(zip(image.tolist(), sender.tolist(), receiver.tolist()))
    assert len(found) == len(image)  # no pair twice
    every = torch.linalg.norm(wrapped[None, None, :] - wrapped[None, :, None] + vectors[:, None, None], dim=-1)
    assert set(map(tuple, torch.nonzero(every < 4.0).tolist())) <= found  # every pair within range


# --------------------------------------------------------------------------- buckets and capacity


def test_bucket_width():
    assert edge_bucket(6912, 0.0, 256) == 256     # the fixed step before 7 October 2026
    assert edge_bucket(6912, 0.05, 256) == 512    # NaCl 3x3x3
    assert edge_bucket(51084, 0.05, 256) == 2560  # NbCl5
    assert edge_bucket(100, 0.05, 256) == 256     # never below the multiple
    assert edge_bucket(6912, 0.05, 128) == 384


def test_round_up():
    assert round_up_edges(7000, 10**6, 512) == 7168
    assert round_up_edges(7168, 10**6, 512) == 7168
    assert round_up_edges(0, 10**6, 512) == 512
    assert round_up_edges(7000, 7100, 512) == 7100  # capped at capacity
    assert round_up_edges(7000, 10**6, 256) == 7168


def test_edge_capacity():
    assert edge_capacity(1000, 1.15, 256) == 1280
    assert edge_capacity(1024, 1.0, 256) == 1024
    assert edge_capacity(1, 1.15, 256) == 256
    assert edge_capacity(1000, 1.15, 100) == 1200


def test_capacity_is_padded_and_grows():
    atoms = nacl(rattle=0.05)
    nl = make(atoms, skin=0.4, variable_cell=True)
    assert nl.build(atoms.positions) is True
    assert nl.capacity % 256 == 0 and nl.capacity > nl.num_candidates
    assert nl.capacity == edge_capacity(nl.num_candidates, 1.15, 256)
    assert nl.build(atoms.positions) is False  # same size: no reallocation
    address = nl.edge_index.data_ptr()
    nl.compact(torch.tensor(atoms.positions))
    assert nl.edge_index.data_ptr() == address
    nl.set_cell(torch.tensor(atoms.cell.array * 0.8))
    assert nl.build(atoms.positions * 0.8) is True
    assert nl.edge_index.data_ptr() != address


def test_config_multiple_and_headroom():
    atoms = nacl(rattle=0.05)
    nl = make(atoms, edge_multiple=100, capacity_headroom=1.5)
    nl.build(atoms.positions)
    assert nl.capacity == edge_capacity(nl.num_candidates, 1.5, 100)


# --------------------------------------------------------------------------- receiver sorting


def rows(nl):
    n = nl.num_candidates
    return list(zip(nl.candidate_index[0, :n].tolist(), nl.candidate_index[1, :n].tolist(),
                    map(tuple, nl.candidate_unit_shifts[:n].tolist()), map(tuple, candidate_shifts(nl)[:n].tolist())))


@pytest.mark.parametrize("name", list(STRUCTURES))
def test_receiver_sort_is_a_stable_permutation(name):
    atoms = STRUCTURES[name](rattle=0.05, seed=1)
    groups = atoms.numbers
    unsorted, by_receiver, by_group = make(atoms), make(atoms, sort=True), make(atoms, sort=True, groups=groups)
    for nl in (unsorted, by_receiver, by_group):
        nl.build_cpu(atoms.positions)
    base = rows(unsorted)
    order = sorted(range(len(base)), key=lambda k: (base[k][1], k))
    assert rows(by_receiver) == [base[k] for k in order]
    order = sorted(range(len(base)), key=lambda k: (base[k][1], groups[base[k][0]], k))
    assert rows(by_group) == [base[k] for k in order]
    assert sorted(rows(by_receiver)) == sorted(base)  # a permutation: nothing lost or duplicated


@pytest.mark.parametrize("groups", [False, True])
def test_torch_build_sorted_by_receiver(groups):
    atoms = nbcl5(rattle=0.05, seed=2)
    group = atoms.numbers % 3 if groups else None
    nl = make(atoms, sort=True, groups=group)
    nl.build_gpu(torch.tensor(atoms.positions))
    reference = make(atoms, sort=True, groups=group)
    reference.build_cpu(atoms.positions)
    assert candidate_set(nl) == candidate_set(reference)
    n = nl.num_candidates
    sender, receiver = nl.candidate_index[:, :n]
    key = receiver * 3 + torch.as_tensor(group)[sender] if groups else receiver
    assert bool((key[1:] >= key[:-1]).all())
    nl.compact(torch.tensor(atoms.positions))
    count = int(nl.count)
    edges = nl.edge_index[1, :count]
    assert bool((edges[1:] >= edges[:-1]).all())


def test_receiver_order_survives_rebuild_and_compaction():
    atoms = STRUCTURES["NaNbO3"](rattle=0.05, seed=2)
    nl = make(atoms, skin=0.5, sort=True)
    for seed in (2, 3, 4):
        positions = atoms.positions +np.random.default_rng(seed).normal(scale=0.03, size=(len(atoms), 3))
        nl.build(positions)
        x = torch.tensor(positions)
        nl.compact(x)
        count = int(nl.count)
        receivers = nl.edge_index[1, :count]
        assert bool((receivers[1:] >= receivers[:-1]).all())
        first, first_units = nl.edge_index[:, :count].clone(), nl.unit_shifts[:count].clone()
        nl.compact(x)  # deterministic
        assert torch.equal(first, nl.edge_index[:, :count])
        assert torch.equal(first_units, nl.unit_shifts[:count])


# --------------------------------------------------------------------------- variable cell


def test_3x3_helpers():
    rng = np.random.default_rng(0)
    m = torch.tensor(rng.normal(size=(3, 3)) + 3 * np.eye(3))
    rows_ = torch.tensor(rng.normal(size=(5, 3)))
    torch.testing.assert_close(times_3x3(rows_, m), rows_ @ m, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(det_3x3(m), torch.linalg.det(m), rtol=1e-13, atol=0)
    torch.testing.assert_close(inverse_3x3(m), torch.linalg.inv(m), rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("name", list(STRUCTURES))
def test_compacted_edges_follow_the_cell(name):
    """After a strain and moves the rebuild test allows, compaction finds exactly the cutoff pairs."""
    atoms = STRUCTURES[name](rattle=0.05, seed=1)
    nl = make(atoms, skin=0.8, variable_cell=True)
    nl.build(atoms.positions)
    rng = np.random.default_rng(2)
    for strain in (np.diag([0.99, 1.01, 0.995]), np.array([[1.0, 0.01, 0], [0.01, 0.99, 0], [0, 0, 1.005]])):
        moved = atoms.copy()
        moved.set_cell(atoms.cell.array @ strain, scale_atoms=True)
        moved.positions += rng.uniform(-0.05, 0.05, moved.positions.shape)
        nl.set_cell(torch.tensor(moved.cell.array))
        x = torch.tensor(moved.positions)
        assert not bool(nl.needs_rebuild(x))
        nl.compact(x)
        assert edge_set(nl) == ase_pairs(moved, CUTOFF)
        n = int(nl.count)
        torch.testing.assert_close(nl.shifts[:n], nl.unit_shifts[:n] @ torch.tensor(moved.cell.array))


def test_strain_alone_triggers_a_rebuild():
    atoms = nacl(rattle=0.05)
    nl = make(atoms, skin=0.3, variable_cell=True)
    nl.build(atoms.positions)
    x = torch.tensor(atoms.positions)
    nl.set_cell(torch.tensor(atoms.cell.array * 1.01))
    assert not bool(nl.needs_rebuild(x * 1.01))
    nl.set_cell(torch.tensor(atoms.cell.array * 0.95))  # 5% compression: pairs from beyond the skin enter
    assert bool(nl.needs_rebuild(x * 0.95))
    nl.build(x * 0.95)  # takes the current cell
    np.testing.assert_allclose(nl.cell, atoms.cell.array * 0.95)
    torch.testing.assert_close(nl.build_cell, torch.tensor(atoms.cell.array * 0.95))
    assert not bool(nl.needs_rebuild(x * 0.95))


@pytest.mark.parametrize("name", list(STRUCTURES))
def test_deformation_aware_test_is_safe(name):
    """Random strains and moves: whenever no rebuild is asked for, the compacted edges are exact."""
    atoms = STRUCTURES[name](rattle=0.05, seed=3)
    nl = make(atoms, skin=0.6, variable_cell=True)
    nl.build(atoms.positions)
    rng = np.random.default_rng(11)
    accepted = rejected = 0
    for step in range(12):
        strain = np.eye(3) + rng.normal(scale=0.004 * (1 + step), size=(3, 3))
        moved = atoms.copy()
        moved.set_cell(atoms.cell.array @ strain, scale_atoms=True)
        moved.positions += rng.uniform(-0.08, 0.08, moved.positions.shape)
        nl.set_cell(torch.tensor(moved.cell.array))
        x = torch.tensor(moved.positions)
        if bool(nl.needs_rebuild(x)):
            rejected += 1
            continue
        accepted += 1
        nl.compact(x)
        assert edge_set(nl) == ase_pairs(moved, CUTOFF)
    assert accepted > 0 and rejected > 0


def test_fraction_scales_the_test():
    atoms = nacl(rattle=0.05)
    for variable_cell in (False, True):
        nl = make(atoms, skin=0.4, variable_cell=variable_cell)
        nl.build(atoms.positions)
        x = torch.tensor(atoms.positions)
        x[0, 0] += 0.15  # 0.75 of skin / 2
        assert not bool(nl.needs_rebuild(x))
        assert bool(nl.needs_rebuild(x, fraction=0.7))


def test_torch_build_uses_the_current_cell():
    atoms = nacl(rattle=0.05)
    reference = SkinNeighbourList(atoms.cell.array * 0.97, CUTOFF, "cpu", F64, NeighbourConfig(skin=0.4),
                                  sort_receivers=False, gpu_build=False)
    nl = make(atoms, skin=0.4, variable_cell=True)
    positions = atoms.positions * 0.97
    nl.build_gpu(torch.tensor(atoms.positions))  # caches built for the original cell
    nl.set_cell(torch.tensor(atoms.cell.array * 0.97))
    reference.build_cpu(positions)
    nl.build_gpu(torch.tensor(positions))
    assert candidate_set(nl) == candidate_set(reference)


def test_buffers_and_edges():
    atoms = nacl(rattle=0.05)
    nl = make(atoms, variable_cell=True)
    nl.build(atoms.positions)
    nl.compact(torch.tensor(atoms.positions))
    buffers = nl.buffers()
    assert {"edge_index", "shifts", "unit_shifts", "count", "reference_positions", "cell", "build_cell"} <= set(
        buffers)
    assert all(torch.is_tensor(value) for value in buffers.values())
    edge_index, shifts, unit_shifts = nl.edges(300)
    assert edge_index.shape == (2, 300) and shifts.shape == unit_shifts.shape == (300, 3)
    assert edge_index.data_ptr() == nl.edge_index.data_ptr()
    assert "cell" not in make(atoms).buffers()


# --------------------------------------------------------------------------- baseline equality

BASELINE = os.environ.get("FASTMACE_BASELINE")
needs_baseline = pytest.mark.skipif(not BASELINE, reason="FASTMACE_BASELINE is not set")


@pytest.fixture(scope="module")
def baseline():
    root = Path(BASELINE) / "Experiments" / "Langevin_Thermostat"
    sys.path.insert(0, str(root))
    try:
        from langevin import gpu_langevin
    finally:
        sys.path.remove(str(root))
    return gpu_langevin


BUFFERS = ("candidate_unit_shifts", "edge_index", "unit_shifts", "shifts", "count", "reference_positions")


def assert_same_state(ours, theirs):
    assert (ours.capacity, ours.num_candidates, ours.rebuilds) == (theirs.capacity, theirs.num_candidates,
                                                                   theirs.rebuilds)
    for name in BUFFERS:
        a, b = getattr(ours, name), getattr(theirs, name)
        assert a.dtype == b.dtype and torch.equal(a, b), name
    # Candidates (cap-nlist): int32 indices; the shifts compact() computes are the baseline's stored ones.
    assert torch.equal(ours.candidate_index.to(torch.int64), theirs.candidate_index)
    assert torch.equal(candidate_shifts(ours), theirs.candidate_shifts64)
    assert torch.equal(candidate_shifts(ours, theirs.candidate_shifts.dtype), theirs.candidate_shifts)


@needs_baseline
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
@pytest.mark.parametrize("mode", ["unsorted", "sorted", "groups"])
@pytest.mark.parametrize("gpu", [False, True], ids=["matscipy", "torch-build"])
@pytest.mark.parametrize("atoms", [a for _, a in CASES], ids=CASE_IDS)
def test_equals_baseline(baseline, atoms, gpu, mode, dtype):
    groups = atoms.numbers % 3 if mode == "groups" else None
    sort = mode != "unsorted"
    ours = make(atoms, skin=0.4, dtype=dtype, sort=sort, groups=groups)
    theirs = baseline.SkinNeighbourList(atoms.cell.array, CUTOFF, 0.4, "cpu", dtype, sort_by_receiver=sort,
                                        sender_groups=groups)
    rng = np.random.default_rng(9)
    positions = atoms.positions
    for _ in range(3):
        if gpu:
            assert ours.build_gpu(torch.tensor(positions)) == theirs.build_gpu(torch.tensor(positions))
        else:
            assert ours.build_cpu(positions) == theirs.build(positions)
        moved = positions + rng.uniform(-0.1, 0.1, positions.shape)
        x = torch.tensor(moved)
        ours.compact(x)
        theirs.compact(x)
        assert_same_state(ours, theirs)
        assert bool(ours.needs_rebuild(x)) == bool(theirs.max_displacement_exceeded(x))
        assert bool(ours.needs_rebuild(x, 0.5)) == bool(theirs.max_displacement_exceeded(x, 0.5))
        positions = moved + rng.uniform(-0.3, 0.3, positions.shape)


@needs_baseline
@pytest.mark.parametrize("gpu", [False, True], ids=["matscipy", "torch-build"])
@pytest.mark.parametrize("name", list(STRUCTURES))
def test_variable_cell_equals_baseline(baseline, name, gpu):
    atoms = STRUCTURES[name](rattle=0.05, seed=6)
    ours = make(atoms, skin=0.5, sort=True, variable_cell=True)
    theirs = baseline.SkinNeighbourList(atoms.cell.array, CUTOFF, 0.5, "cpu", F64, sort_by_receiver=True,
                                        variable_cell=True)
    rng = np.random.default_rng(10)
    if gpu:
        assert ours.build_gpu(torch.tensor(atoms.positions)) == theirs.build_gpu(torch.tensor(atoms.positions))
    else:
        assert ours.build_cpu(atoms.positions) == theirs.build(atoms.positions)
    for _ in range(4):
        strain = np.eye(3) + rng.normal(scale=0.01, size=(3, 3))
        cell = torch.tensor(atoms.cell.array @ strain)
        x = torch.tensor(atoms.positions @ strain + rng.uniform(-0.05, 0.05, atoms.positions.shape))
        ours.set_cell(cell)
        theirs.set_cell(cell)
        assert bool(ours.needs_rebuild(x)) == bool(theirs.max_displacement_exceeded(x))
        if gpu:
            assert ours.build_gpu(x) == theirs.build_gpu(x)
        else:
            assert ours.build_cpu(x.numpy()) == theirs.build(x.numpy())
        ours.compact(x)
        theirs.compact(x)
        assert_same_state(ours, theirs)
        assert torch.equal(ours.build_cell, theirs.build_cell)
        np.testing.assert_array_equal(ours.cell, theirs.cell)


@needs_baseline
def test_buckets_equal_baseline(baseline):
    for count in (0, 1, 100, 6912, 7000, 51084, 123457):
        for fraction in (0.0, 0.05, 0.13):
            assert edge_bucket(count, fraction, 256) == baseline.edge_bucket(count, fraction)
        for bucket in (256, 512, 2560):
            for capacity in (7100, 10**6):
                assert round_up_edges(count, capacity, bucket) == baseline.round_up_edges(count, capacity, bucket)
    rng = np.random.default_rng(0)
    m = torch.tensor(rng.normal(size=(3, 3)) + 2 * np.eye(3))
    r = torch.tensor(rng.normal(size=(7, 3)))
    assert torch.equal(times_3x3(r, m), baseline._times_3x3(r, m))
    assert torch.equal(det_3x3(m), baseline.det_3x3(m))
    assert torch.equal(inverse_3x3(m), baseline.inverse_3x3(m))
