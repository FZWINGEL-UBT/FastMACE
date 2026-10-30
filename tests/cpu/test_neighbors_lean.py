"""Lean neighbour buffers and the chunked build (cap-nlist): int32 candidate indices, no stored candidate shifts.

``compact`` computes the candidates' shifts from the unit shifts and ``build_cell`` with the build's times_3x3.
Before cap-nlist the list stored them (``times_3x3`` on the GPU build, numpy's ``unit_shifts @ cell`` on the
matscipy build); the comparison against both layouts, and MD trajectories bitwise equal with the old layout,
were recorded at commit 787e44df (docs/validation/capacity-nlist.md). ``BUILD_CHUNK`` bounds the GPU build's
float64 decision; the result does not depend on it.
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


def make(atoms, *, dtype=torch.float64, mode="sorted"):
    groups = atoms.numbers % 3 if mode == "groups" else None
    return SkinNeighbourList(atoms.cell.array, CUTOFF, "cpu", dtype, NeighbourConfig(skin=0.4),
                             sort_receivers=mode != "unsorted", gpu_build=False, sender_groups=groups)


def test_lean_layout():
    atoms = STRUCTURES["NaCl"](rattle=0.05, seed=1)
    nl = make(atoms, dtype=torch.float32)
    nl.build_gpu(torch.tensor(atoms.positions))
    assert nl.candidate_index.dtype == torch.int32
    assert not hasattr(nl, "candidate_shifts") and not hasattr(nl, "candidate_shifts64")
    assert torch.equal(nl.buffers()["build_cell"], torch.tensor(atoms.cell.array))
    slots = [t for t in nl.buffers().values() if nl.capacity in t.shape]
    per_slot = sum(t.numel() * t.element_size() for t in slots) / nl.capacity
    assert per_slot == 60  # 104 before: int64 candidate indices, float32 and float64 candidate shifts


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
@pytest.mark.parametrize("build", ["matscipy", "torch-build"])
@pytest.mark.parametrize("atoms", [a for _, a in CASES], ids=IDS)
def test_compacted_shifts_equal_the_stored_ones_before(atoms, build, dtype):
    """The shifts compact() writes equal numpy's ``unit_shifts @ cell`` (what the matscipy build stored before; an
    independent product), cast to the edge precision, padding slots included. The GPU build stored times_3x3's
    values, which the baseline tests in test_neighbors.py compare."""
    nl = make(atoms, dtype=dtype)
    rng = np.random.default_rng(9)
    positions = atoms.positions
    for _ in range(3):
        if build == "matscipy":
            nl.build_cpu(positions)
        else:
            nl.build_gpu(torch.tensor(positions))
        x = torch.tensor(positions + rng.uniform(-0.1, 0.1, positions.shape))
        nl.compact(x)
        stored = torch.from_numpy(nl.unit_shifts.to(torch.float64).numpy() @ atoms.cell.array)
        assert torch.equal(nl.shifts, stored.to(dtype))
        positions = x.numpy() + rng.uniform(-0.3, 0.3, positions.shape)


@pytest.mark.parametrize("mode", ["unsorted", "groups"])
def test_chunked_build_equals_one_chunk(monkeypatch, mode):
    atoms = nbcl5(repeat=2, rattle=0.05, seed=5)
    one = make(atoms, mode=mode)
    monkeypatch.setattr(skin, "BUILD_CHUNK", 1 << 30)
    one.build_gpu(torch.tensor(atoms.positions))
    chunked = make(atoms, mode=mode)
    monkeypatch.setattr(skin, "BUILD_CHUNK", 37)
    chunked.build_gpu(torch.tensor(atoms.positions))
    assert one.num_candidates > 37 * 10
    for name, tensor in one.buffers().items():
        assert torch.equal(tensor, chunked.buffers()[name]), name


def test_lean_trial_list_round_trip():
    atoms = STRUCTURES["NaCl"](rattle=0.05, seed=2)
    nl = make(atoms)
    nl.build_gpu(torch.tensor(atoms.positions))
    nl.compact(torch.tensor(atoms.positions))
    saved = nl.save_list()
    assert "build_cell" in saved
    before = {k: v.clone() for k, v in nl.buffers().items()}
    nl.build_gpu(torch.tensor(atoms.positions + 0.2))
    nl.restore_list(saved)
    for name, tensor in nl.buffers().items():
        assert torch.equal(tensor, before[name]), name
