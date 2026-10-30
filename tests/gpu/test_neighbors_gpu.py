"""Skin neighbour list on CUDA: the Triton build equals the CPU build, and the device stages capture.

``FASTMACE_BASELINE`` (the exported research-engine tag ``research-engine-baseline-20261017``)
adds bitwise comparisons with its ``SkinNeighbourList.build_gpu`` on the same GPU.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from _support.structures import STRUCTURES, nbcl5
from fastmace.config import NeighbourConfig
from fastmace.neighbors.skin import SkinNeighbourList, times_3x3

CUTOFF = 5.0
SHEAR = np.array([[1.0, 0.02, 0.0], [0.02, 0.98, 0.01], [0.0, 0.01, 1.015]])


def cases():
    out = []
    for name, build in STRUCTURES.items():
        out.append((f"{name}-rattled", build(rattle=0.05, seed=3)))
        strained = build(rattle=0.05, seed=4)
        strained.set_cell(strained.cell.array @ SHEAR, scale_atoms=True)
        out.append((f"{name}-strained", strained))
        out.append((f"{name}x3-rattled", build(repeat=3, rattle=0.05, seed=5)))
    out.append(("NbCl5x2-rattled", nbcl5(repeat=2, rattle=0.05, seed=5)))
    return out


CASES = cases()
IDS = [name for name, _ in CASES]
MODES = ["unsorted", "sorted", "groups"]


def make(atoms, device, mode, dtype=torch.float64, variable_cell=False, skin=0.4):
    groups = atoms.numbers % 3 if mode == "groups" else None
    return SkinNeighbourList(atoms.cell.array, CUTOFF, device, dtype, NeighbourConfig(skin=skin),
                             sort_receivers=mode != "unsorted", gpu_build=device == "cuda", sender_groups=groups,
                             variable_cell=variable_cell)


def candidates(nl):
    n = nl.num_candidates
    index = nl.candidate_index[:, :n].cpu()
    shifts = nl.candidate_unit_shifts[:n].cpu().round().to(torch.int64)
    return set(map(tuple, torch.cat([index.T, shifts], dim=1).tolist()))


def edges(nl):
    n = int(nl.count)
    index = nl.edge_index[:, :n].cpu()
    shifts = nl.unit_shifts[:n].cpu().round().to(torch.int64)
    return set(map(tuple, torch.cat([index.T, shifts], dim=1).tolist()))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("atoms", [a for _, a in CASES], ids=IDS)
def test_gpu_build_equals_cpu_build(atoms, mode):
    cpu, gpu = make(atoms, "cpu", mode), make(atoms, "cuda", mode)
    rng = np.random.default_rng(1)
    positions = atoms.positions
    for _ in range(3):
        assert cpu.build(positions) == gpu.build(torch.tensor(positions, device="cuda"))
        assert gpu.num_candidates == cpu.num_candidates and gpu.capacity == cpu.capacity
        assert candidates(gpu) == candidates(cpu)
        moved = positions + rng.uniform(-0.1, 0.1, positions.shape)
        cpu.compact(torch.tensor(moved))
        gpu.compact(torch.tensor(moved, device="cuda"))
        assert int(gpu.count) == int(cpu.count)
        assert edges(gpu) == edges(cpu)
        n = gpu.num_candidates
        if mode != "unsorted":
            receivers = gpu.candidate_index[1, :n]
            assert bool((receivers[1:] >= receivers[:-1]).all())
        assert bool((gpu.candidate_index[:, n:] == 0).all())
        positions = moved + rng.uniform(-0.3, 0.3, positions.shape)


@pytest.mark.parametrize("name", list(STRUCTURES))
def test_variable_cell_on_gpu(name):
    atoms = STRUCTURES[name](rattle=0.05, seed=6)
    cpu, gpu = make(atoms, "cpu", "sorted", variable_cell=True), make(atoms, "cuda", "sorted", variable_cell=True)
    cpu.build(atoms.positions)
    gpu.build(torch.tensor(atoms.positions, device="cuda"))
    rng = np.random.default_rng(2)
    for _ in range(4):
        strain = np.eye(3) + rng.normal(scale=0.01, size=(3, 3))
        cell = torch.tensor(atoms.cell.array @ strain)
        x = torch.tensor(atoms.positions @ strain + rng.uniform(-0.05, 0.05, atoms.positions.shape))
        cpu.set_cell(cell)
        gpu.set_cell(cell.cuda())
        assert bool(cpu.needs_rebuild(x)) == bool(gpu.needs_rebuild(x.cuda()))
        cpu.build(x)
        gpu.build(x.cuda())
        assert candidates(gpu) == candidates(cpu)
        cpu.compact(x)
        gpu.compact(x.cuda())
        assert edges(gpu) == edges(cpu)


@pytest.mark.parametrize("variable_cell", [False, True], ids=["fixed", "variable"])
def test_device_stages_replay_in_a_cuda_graph(variable_cell):
    """set_cell, needs_rebuild and compact captured once, replayed with new inputs: same as eager."""
    atoms = STRUCTURES["NaNbO3"](repeat=3, rattle=0.05, seed=7)
    nl = make(atoms, "cuda", "sorted", dtype=torch.float32, variable_cell=variable_cell)
    eager = make(atoms, "cuda", "sorted", dtype=torch.float32, variable_cell=variable_cell)
    x0 = torch.tensor(atoms.positions, device="cuda")
    nl.build(x0)
    eager.build(x0)
    x = x0.clone()
    cell = torch.tensor(atoms.cell.array, device="cuda")
    flag = torch.zeros((), dtype=torch.bool, device="cuda")

    def stage():
        if variable_cell:
            nl.set_cell(cell)
        flag.copy_(nl.needs_rebuild(x))
        nl.compact(x)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        stage()  # warm-up
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stage()
    rng = np.random.default_rng(3)
    for _ in range(5):
        strain = np.eye(3) + (rng.normal(scale=0.003, size=(3, 3)) if variable_cell else 0)
        new = torch.tensor(atoms.positions @ strain + rng.uniform(-0.08, 0.08, atoms.positions.shape),
                           device="cuda")
        x.copy_(new)
        cell.copy_(torch.tensor(atoms.cell.array @ strain))
        graph.replay()
        if variable_cell:
            eager.set_cell(cell)
        expected = eager.needs_rebuild(new)
        eager.compact(new)
        torch.cuda.synchronize()
        assert bool(flag) == bool(expected)
        assert int(nl.count) == int(eager.count)
        for name in ("edge_index", "unit_shifts", "shifts"):
            assert torch.equal(getattr(nl, name), getattr(eager, name)), name


BASELINE = os.environ.get("FASTMACE_BASELINE")


@pytest.fixture(scope="module")
def baseline():
    if not BASELINE:
        pytest.skip("FASTMACE_BASELINE is not set")
    root = Path(BASELINE) / "Experiments" / "Langevin_Thermostat"
    sys.path.insert(0, str(root))
    try:
        from langevin import gpu_langevin
    finally:
        sys.path.remove(str(root))
    return gpu_langevin


BUFFERS = ("candidate_unit_shifts", "edge_index", "unit_shifts", "shifts", "count", "reference_positions")


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("atoms", [a for _, a in CASES], ids=IDS)
def test_gpu_build_equals_baseline(baseline, atoms, mode, dtype):
    groups = atoms.numbers % 3 if mode == "groups" else None
    ours = make(atoms, "cuda", mode, dtype=dtype)
    theirs = baseline.SkinNeighbourList(atoms.cell.array, CUTOFF, 0.4, "cuda", dtype,
                                        sort_by_receiver=mode != "unsorted", sender_groups=groups)
    rng = np.random.default_rng(9)
    positions = atoms.positions
    for _ in range(3):
        x = torch.tensor(positions, device="cuda")
        assert ours.build(x) == theirs.build_gpu(x)
        moved = torch.tensor(positions + rng.uniform(-0.1, 0.1, positions.shape), device="cuda")
        ours.compact(moved)
        theirs.compact(moved)
        assert (ours.capacity, ours.num_candidates) == (theirs.capacity, theirs.num_candidates)
        for name in BUFFERS:
            assert torch.equal(getattr(ours, name), getattr(theirs, name)), name
        # Candidates (cap-nlist): int32 indices, shifts computed from the unit shifts and the build cell.
        assert torch.equal(ours.candidate_index.to(torch.int64), theirs.candidate_index)
        derived = times_3x3(ours.candidate_unit_shifts.to(torch.float64), ours.build_cell)
        assert torch.equal(derived, theirs.candidate_shifts64)
        assert torch.equal(derived.to(dtype), theirs.candidate_shifts)
        assert bool(ours.needs_rebuild(moved)) == bool(theirs.max_displacement_exceeded(moved))
        positions = moved.cpu().numpy() + rng.uniform(-0.3, 0.3, positions.shape)


@pytest.mark.parametrize("half", [False, True])
@pytest.mark.parametrize("max_grid_y", [1, 3, 7])
def test_pair_search_split_grid_finds_the_same_pairs(half, max_grid_y):
    """Receiver tiles split over the grid's y and z axes (as beyond 65,535 tiles, about 2.1M atoms) screen the
    same pairs as the 2-D grid: the same codes as a set, each once (the order is the atomic append's)."""
    from fastmace.neighbors.pair_search import screen_codes

    atoms = nbcl5(repeat=2, rattle=0.05)  # 288 atoms, 9 receiver tiles
    images = torch.tensor(np.stack(np.meshgrid(*[np.arange(-1, 2)] * 3, indexing="ij"), -1).reshape(-1, 3),
                          dtype=torch.float64)
    vectors = (images @ torch.tensor(atoms.cell.array)).cuda()
    wrapped = torch.tensor(atoms.get_positions(wrap=True)).cuda()
    assert -(-len(atoms) // 32) > max_grid_y  # the split is exercised
    reference = torch.sort(screen_codes(wrapped, vectors, 5.0, half=half).clone()).values
    split = torch.sort(screen_codes(wrapped, vectors, 5.0, half=half, max_grid_y=max_grid_y).clone()).values
    assert reference.numel() > 0 and torch.equal(split, reference)
    assert torch.unique(split).numel() == split.numel()


@pytest.mark.parametrize("variable_cell", [False, True])
def test_chunked_compact_on_gpu_is_bitwise_the_one_chunk_compact(variable_cell, monkeypatch):
    """As tests/cpu/test_neighbors.py's chunked-compact test, on the GPU build, and inside a CUDA graph."""
    from fastmace.neighbors import skin

    atoms = nbcl5(repeat=2, rattle=0.05)
    moved = torch.tensor(atoms.positions + np.random.default_rng(5).uniform(-0.1, 0.1, atoms.positions.shape),
                         device="cuda")
    lists = []
    for size in (skin.COMPACT_CHUNK, 777):
        monkeypatch.setattr(skin, "COMPACT_CHUNK", size)
        nl = make(atoms, "cuda", "groups", variable_cell=variable_cell)
        nl.build(torch.tensor(atoms.positions, device="cuda"))
        nl.compact(moved)
        lists.append((nl.edge_index.clone(), nl.unit_shifts.clone(), nl.shifts.clone(), int(nl.count)))
        if size == 777:  # the chunked compaction captured and replayed
            static = moved.clone()
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                nl.compact(static)
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(graph, stream=stream):
                nl.compact(static)
            nl.edge_index.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(nl.edge_index, lists[-1][0]) and int(nl.count) == lists[-1][3]
    (index, unit, shifts, count), (index_c, unit_c, shifts_c, count_c) = lists
    assert count == count_c > 0
    assert torch.equal(index, index_c) and torch.equal(unit, unit_c) and torch.equal(shifts, shifts_c)

