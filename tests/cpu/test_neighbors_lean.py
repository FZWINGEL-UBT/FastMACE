"""Lean neighbour buffers and the chunked build (cap-nlist): the same compacted edges as the layout before, bitwise.

``fastmace.neighbors.skin.LEAN_BUFFERS`` (read at construction) stores int32 candidate indices and no candidate
shifts; ``compact`` computes the shifts from the unit shifts and ``build_cell``. ``BUILD_CHUNK`` bounds the GPU
build's float64 decision; the result does not depend on it.
"""

import numpy as np
import pytest
import torch

from _support.structures import STRUCTURES, nbcl5
from fastmace.config import NeighbourConfig
from fastmace.neighbors import skin
from fastmace.neighbors.skin import SkinNeighbourList

CUTOFF = 5.0
SHEAR = np.array([[1.0, 0.02, 0.0], [0.02, 0.98, 0.01], [0.0, 0.01, 1.015]])
EDGES = ("edge_index", "unit_shifts", "shifts", "count", "reference_positions")


def cases():
    out = []
    for name, build in STRUCTURES.items():
        out.append((f"{name}-rattled", build(rattle=0.05, seed=3)))
        strained = build(rattle=0.05, seed=4)
        strained.set_cell(strained.cell.array @ SHEAR, scale_atoms=True)
        out.append((f"{name}-strained", strained))
    out.append(("NbCl5x2-rattled", nbcl5(repeat=2, rattle=0.05, seed=5)))
    return out


CASES = cases()
IDS = [name for name, _ in CASES]


def make(monkeypatch, atoms, lean, *, dtype=torch.float64, mode="sorted", variable_cell=False):
    monkeypatch.setattr(skin, "LEAN_BUFFERS", lean)
    groups = atoms.numbers % 3 if mode == "groups" else None
    return SkinNeighbourList(atoms.cell.array, CUTOFF, "cpu", dtype, NeighbourConfig(skin=0.4),
                             sort_receivers=mode != "unsorted", gpu_build=False, sender_groups=groups,
                             variable_cell=variable_cell)


def test_lean_layout(monkeypatch):
    atoms = STRUCTURES["NaCl"](rattle=0.05, seed=1)
    nl = make(monkeypatch, atoms, True, dtype=torch.float32)
    nl.build_gpu(torch.tensor(atoms.positions))
    assert nl.lean and nl.candidate_index.dtype == torch.int32
    assert nl.candidate_shifts is None and nl.candidate_shifts64 is None
    assert "candidate_shifts" not in nl.buffers() and "candidate_shifts64" not in nl.buffers()
    assert torch.equal(nl.buffers()["build_cell"], torch.tensor(atoms.cell.array))
    slots = [t for t in nl.buffers().values() if nl.capacity in t.shape]
    per_slot = sum(t.numel() * t.element_size() for t in slots) / nl.capacity
    assert per_slot == 60  # 104 before: int64 candidate indices, float32 and float64 candidate shifts
    legacy = make(monkeypatch, atoms, False, dtype=torch.float32)
    assert not legacy.lean and "build_cell" not in legacy.buffers()


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
@pytest.mark.parametrize("mode", ["unsorted", "sorted", "groups"])
@pytest.mark.parametrize("build", ["matscipy", "torch-build"])
@pytest.mark.parametrize("atoms", [a for _, a in CASES], ids=IDS)
def test_lean_equals_legacy(monkeypatch, atoms, build, mode, dtype):
    lean, legacy = (make(monkeypatch, atoms, flag, dtype=dtype, mode=mode) for flag in (True, False))
    rng = np.random.default_rng(9)
    positions = atoms.positions
    for _ in range(3):
        for nl in (lean, legacy):
            if build == "matscipy":
                nl.build_cpu(positions)
            else:
                nl.build_gpu(torch.tensor(positions))
        assert (lean.capacity, lean.num_candidates) == (legacy.capacity, legacy.num_candidates)
        assert torch.equal(lean.candidate_index.to(torch.int64), legacy.candidate_index)
        assert torch.equal(lean.candidate_unit_shifts, legacy.candidate_unit_shifts)
        moved = positions + rng.uniform(-0.1, 0.1, positions.shape)
        x = torch.tensor(moved)
        lean.compact(x)
        legacy.compact(x)
        for name in EDGES:
            a, b = getattr(lean, name), getattr(legacy, name)
            assert a.dtype == b.dtype and torch.equal(a, b), name
        positions = moved + rng.uniform(-0.3, 0.3, positions.shape)


@pytest.mark.parametrize("name", list(STRUCTURES))
def test_lean_variable_cell_equals_legacy(monkeypatch, name):
    atoms = STRUCTURES[name](rattle=0.05, seed=6)
    lean, legacy = (make(monkeypatch, atoms, flag, variable_cell=True) for flag in (True, False))
    cell = torch.tensor(atoms.cell.array @ SHEAR)
    for nl in (lean, legacy):
        nl.build_gpu(torch.tensor(atoms.positions))
        nl.set_cell(cell)
        nl.compact(torch.tensor(atoms.positions @ SHEAR))
    for key in EDGES:
        assert torch.equal(getattr(lean, key), getattr(legacy, key)), key


@pytest.mark.parametrize("mode", ["unsorted", "groups"])
def test_chunked_build_equals_one_chunk(monkeypatch, mode):
    atoms = nbcl5(repeat=2, rattle=0.05, seed=5)
    one = make(monkeypatch, atoms, True, mode=mode)
    monkeypatch.setattr(skin, "BUILD_CHUNK", 1 << 30)
    one.build_gpu(torch.tensor(atoms.positions))
    chunked = make(monkeypatch, atoms, True, mode=mode)
    monkeypatch.setattr(skin, "BUILD_CHUNK", 37)
    chunked.build_gpu(torch.tensor(atoms.positions))
    assert one.num_candidates > 37 * 10
    for name, tensor in one.buffers().items():
        assert torch.equal(tensor, chunked.buffers()[name]), name


def test_lean_trial_list_round_trip(monkeypatch):
    atoms = STRUCTURES["NaCl"](rattle=0.05, seed=2)
    nl = make(monkeypatch, atoms, True)
    nl.build_gpu(torch.tensor(atoms.positions))
    nl.compact(torch.tensor(atoms.positions))
    saved = nl.save_list()
    assert "build_cell" in saved
    before = {k: v.clone() for k, v in nl.buffers().items()}
    nl.build_gpu(torch.tensor(atoms.positions + 0.2))
    nl.restore_list(saved)
    for name, tensor in nl.buffers().items():
        assert torch.equal(tensor, before[name]), name
