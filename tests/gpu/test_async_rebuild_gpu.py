"""The asynchronous early rebuild (``SkinNeighbourList.build_gpu_async``; fp32 profile, 28 October 2026) on CUDA.

It must leave exactly the buffers ``build_gpu`` leaves (the same pairs, order, shifts and padding), compact to the
same edges, synchronise nothing, and turn an overflow (screen buffer or capacity too small) into a fired rebuild
test, so that the engine's rebuild redo rebuilds synchronously before the step commits.
"""

import numpy as np
import pytest
import torch

from _support.structures import STRUCTURES, nbcl5
from fastmace.config import NeighbourConfig
from fastmace.neighbors.skin import SkinNeighbourList

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

CUTOFF = 5.0


def cases():
    out = [(f"{name}x3", build(repeat=3, rattle=0.05, seed=5)) for name, build in STRUCTURES.items()]
    out.append(("NbCl5x2", nbcl5(repeat=2, rattle=0.05, seed=5)))
    return out


CASES = cases()


def make(atoms, mode):
    groups = atoms.numbers % 3 if mode == "groups" else None
    nl = SkinNeighbourList(atoms.cell.array, CUTOFF, "cuda", torch.float32, NeighbourConfig(skin=0.4),
                           sort_receivers=mode != "unsorted", gpu_build=True, sender_groups=groups)
    nl.async_build = True
    return nl


def moved(atoms, seed, scale=0.15):
    rng = np.random.default_rng(seed)
    return torch.tensor(atoms.positions + rng.normal(scale=scale, size=atoms.positions.shape), dtype=torch.float64,
                        device="cuda")


def state(nl):
    names = ("candidate_index", "candidate_unit_shifts", "reference_positions", "build_cell")
    return {name: getattr(nl, name).clone() for name in names}


@pytest.mark.parametrize("mode", ["sorted", "groups", "unsorted"])
@pytest.mark.parametrize("name,atoms", CASES, ids=[name for name, _ in CASES])
def test_async_build_equals_sync_build(name, atoms, mode):
    first = torch.tensor(atoms.positions, dtype=torch.float64, device="cuda")
    sync, asynchronous = make(atoms, mode), make(atoms, mode)
    sync.build(first)
    asynchronous.build(first)  # the first build of a list is always synchronous
    assert asynchronous.async_builds == 0
    later = moved(atoms, seed=11)
    sync.build_gpu(later)
    asynchronous.build_gpu_async(later)
    assert asynchronous.async_builds == 1
    expected, got = state(sync), state(asynchronous)
    for key in expected:
        assert torch.equal(expected[key], got[key]), key
    assert asynchronous.num_candidates == sync.num_candidates
    sync.compact(later)
    asynchronous.compact(later)
    assert int(sync.count) == int(asynchronous.count)
    n = int(sync.count)
    assert torch.equal(sync.edge_index[:, :n], asynchronous.edge_index[:, :n])
    assert torch.equal(sync.shifts[:n], asynchronous.shifts[:n])
    assert not bool(asynchronous.needs_rebuild(later))


def test_async_build_takes_the_beside_path_only():
    atoms = STRUCTURES["NaCl"](repeat=3, rattle=0.05, seed=5)
    nl = make(atoms, "sorted")
    positions = torch.tensor(atoms.positions, dtype=torch.float64, device="cuda")
    nl.build(positions)
    nl.build(moved(atoms, seed=2))  # no reader stream: synchronous
    assert nl.async_builds == 0
    nl.reader_stream = torch.cuda.current_stream()
    try:
        assert nl.build(moved(atoms, seed=3)) is False
    finally:
        nl.reader_stream = None
    assert nl.async_builds == 1


def test_async_build_does_not_synchronise():
    atoms = STRUCTURES["NaCl"](repeat=3, rattle=0.05, seed=5)
    nl = make(atoms, "sorted")
    nl.build(torch.tensor(atoms.positions, dtype=torch.float64, device="cuda"))
    later = moved(atoms, seed=4)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        nl.build_gpu_async(later)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.parametrize("what", ["screen", "capacity"])
def test_async_overflow_fires_the_rebuild_test(what):
    atoms = STRUCTURES["NaCl"](repeat=3, rattle=0.05, seed=5)
    nl = make(atoms, "sorted")
    positions = torch.tensor(atoms.positions, dtype=torch.float64, device="cuda")
    nl.build(positions)
    if what == "screen":
        nl._pair_search_state["buffer"] = torch.empty(16, dtype=torch.int64, device="cuda")
    else:
        nl.capacity = 32  # the buffers are larger; the build must still see too small a capacity
    later = moved(atoms, seed=5, scale=0.01)
    nl.build_gpu_async(later)
    assert bool(nl.needs_rebuild(later))
    if what == "capacity":
        nl.capacity = nl.candidate_index.shape[1]
    nl._pair_search_state.pop("buffer")
    nl.build_gpu(later)  # the engine's redo: synchronous, grows what was too small
    assert not bool(nl.needs_rebuild(later))
