"""Lean neighbour buffers on CUDA (cap-nlist): the chunked Triton build, and its lower peak.

1. The Triton build in chunks of ``BUILD_CHUNK`` equals the build in one chunk, buffer for buffer, after several
   builds and compactions on the test structures.
2. The build's peak allocation above the list's own buffers is below 64 B per pair.

The comparison against the layout before cap-nlist (stored candidate shifts, int64 candidate indices) and MD
trajectories bitwise equal with it were recorded at commit 787e44df (docs/validation/capacity-nlist.md).
"""

import numpy as np
import pytest
import torch

from _support.structures import STRUCTURES, nbcl5
from fastmace.config import NeighbourConfig
from fastmace.neighbors import skin
from fastmace.neighbors.skin import SkinNeighbourList

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]
CUTOFF = 5.0
CASES = [(f"{name}x3", build(repeat=3, rattle=0.05, seed=5)) for name, build in STRUCTURES.items()]
CASES.append(("NbCl5x2", nbcl5(repeat=2, rattle=0.05, seed=5)))


def make(atoms, mode, dtype):
    groups = atoms.numbers % 3 if mode == "groups" else None
    return SkinNeighbourList(atoms.cell.array, CUTOFF, "cuda", dtype, NeighbourConfig(skin=0.4),
                             sort_receivers=mode != "unsorted", gpu_build=True, sender_groups=groups)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
@pytest.mark.parametrize("mode", ["unsorted", "sorted", "groups"])
@pytest.mark.parametrize("name,atoms", CASES, ids=[name for name, _ in CASES])
def test_chunked_build_equals_one_chunk_on_cuda(monkeypatch, name, atoms, mode, dtype):
    chunked, whole = make(atoms, mode, dtype), make(atoms, mode, dtype)
    rng = np.random.default_rng(9)
    positions = atoms.positions
    for _ in range(3):
        x = torch.tensor(positions, device="cuda")
        monkeypatch.setattr(skin, "BUILD_CHUNK", 1000)
        grew = chunked.build(x)
        monkeypatch.setattr(skin, "BUILD_CHUNK", 1 << 30)
        assert whole.build(x) == grew
        moved = torch.tensor(positions + rng.uniform(-0.1, 0.1, positions.shape), device="cuda")
        chunked.compact(moved)
        whole.compact(moved)
        assert (chunked.capacity, chunked.num_candidates) == (whole.capacity, whole.num_candidates)
        for key, tensor in whole.buffers().items():
            assert torch.equal(chunked.buffers()[key], tensor), key
        positions = moved.cpu().numpy() + rng.uniform(-0.3, 0.3, positions.shape)


def test_lean_build_peak(monkeypatch):
    atoms = STRUCTURES["NaCl"](repeat=8, rattle=0.05, seed=2)  # 4,096 atoms, about 0.2M screened pairs
    monkeypatch.setattr(skin, "BUILD_CHUNK", 1 << 14)
    nl = make(atoms, "groups", torch.float32)
    x = torch.tensor(atoms.positions, device="cuda")
    nl.build(x)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    nl.build(x + 0.01)
    torch.cuda.synchronize()
    assert (torch.cuda.max_memory_allocated() - before) / nl.num_candidates < 64
