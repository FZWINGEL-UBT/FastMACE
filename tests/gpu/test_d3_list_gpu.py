"""The D3 half list on CUDA (plan task 3b-T02): the Triton build equals the CPU build bit for bit, the
device stages capture, and NbCl5 3x3x3 has rule 12's pair count (SPEC §11.2)."""

import numpy as np
import pytest
import torch

from _support.structures import STRUCTURES, nbcl5
from fastmace.neighbors import adaptive
from fastmace.neighbors.d3_list import D3PairList
from fastmace.neighbors.pair_search import candidate_pairs

D3_CUTOFF = 40 * 0.52917726
SHEAR = np.array([[1.0, 0.07, -0.03], [0.05, 0.93, 0.04], [-0.02, 0.06, 1.08]])


def cases():
    out = []
    for name, build in STRUCTURES.items():
        out.append((f"{name}-rattled", build(repeat=1, rattle=0.05, seed=3)))
        strained = build(repeat=2, rattle=0.05, seed=4)
        strained.set_cell(strained.cell.array @ SHEAR, scale_atoms=True)
        out.append((f"{name}x2-strained", strained))
    unwrapped = nbcl5(repeat=2, rattle=0.05, seed=5)
    rng = np.random.default_rng(1)
    unwrapped.positions += rng.integers(-3, 4, size=(len(unwrapped), 3)) @ unwrapped.cell.array
    out.append(("NbCl5x2-unwrapped", unwrapped))
    return out


CASES = cases()
IDS = [name for name, _ in CASES]


@pytest.mark.parametrize("atoms", [a for _, a in CASES], ids=IDS)
def test_gpu_build_equals_cpu_build(atoms):
    """Same pairs in the same order (the unique sort key), every buffer bitwise."""
    cpu = D3PairList(atoms.cell.array, D3_CUTOFF, 1.0, "cpu")
    gpu = D3PairList(atoms.cell.array, D3_CUTOFF, 1.0, "cuda")
    cpu.build(atoms.positions)
    gpu.build(torch.tensor(atoms.positions, device="cuda"))
    assert gpu.num_pairs == cpu.num_pairs and gpu.capacity == cpu.capacity
    for name, tensor in cpu.buffers().items():
        assert torch.equal(gpu.buffers()[name].cpu(), tensor), name
    assert int(gpu.count) == gpu.num_pairs


def test_half_screen_equals_the_full_screen():
    """pair_search half=True on CUDA: the sender <= receiver part of the full screen."""
    atoms = nbcl5(repeat=2, rattle=0.05, seed=6)
    x = torch.tensor(atoms.positions, device="cuda")
    images = torch.tensor(np.array([[0, 0, 0], [1, 0, 0], [0, -1, 1]]) @ atoms.cell.array, device="cuda")
    full = torch.stack(candidate_pairs(x, images, 8.0), dim=1).cpu()
    half = torch.stack(candidate_pairs(x, images, 8.0, half=True), dim=1).cpu()
    expected = {tuple(row) for row in full.tolist() if row[1] <= row[2]}
    assert {tuple(row) for row in half.tolist()} == expected and len(half) == len(expected)


@pytest.mark.parametrize("variable_cell", [False, True], ids=["fixed", "variable"])
def test_device_stages_replay_in_a_cuda_graph(variable_cell):
    """set_cell, needs_rebuild and prepare captured once, replayed with new inputs: same as eager."""
    atoms = nbcl5(repeat=2, rattle=0.05, seed=7)
    nl = D3PairList(atoms.cell.array, D3_CUTOFF, 1.0, "cuda", variable_cell=variable_cell)
    eager = D3PairList(atoms.cell.array, D3_CUTOFF, 1.0, "cuda", variable_cell=variable_cell)
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
        nl.prepare(x)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        stage()  # warm-up
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stage()
    rng = np.random.default_rng(3)
    for trial in range(5):
        strain = np.eye(3) + (rng.normal(scale=0.003, size=(3, 3)) if variable_cell else 0)
        new = torch.tensor(atoms.positions @ strain + rng.uniform(-0.6, 0.6, atoms.positions.shape) * trial / 5,
                           device="cuda")
        x.copy_(new)
        cell.copy_(torch.tensor(atoms.cell.array @ strain))
        graph.replay()
        if variable_cell:
            eager.set_cell(cell)
        expected = eager.needs_rebuild(new)
        eager.prepare(new)
        torch.cuda.synchronize()
        assert bool(flag) == bool(expected)
        for name in ("shifted_positions", "image_vectors"):
            assert torch.equal(getattr(nl, name), getattr(eager, name)), name
        assert torch.equal(nl.inside(), eager.inside())


def test_nbcl5_3x3x3_pairs_and_build_time():
    """SPEC §11.2: NbCl5 3x3x3 stores about 1.4 M pairs at 21.5 Å and 1.75 M at 23.2 Å; builds are counted."""
    # The test structure is the 36-atom primitive cell: 3x3x3 is 972 atoms, half the production supercell
    # (1944 atoms) at the same density, so half its pairs.
    atoms = nbcl5(repeat=3, rattle=0.05, seed=8)
    assert len(atoms) == 972
    x = torch.tensor(atoms.positions, device="cuda")
    for skin in (0.33, 2.0):
        nl = D3PairList(atoms.cell.array, D3_CUTOFF, skin, "cuda")
        nl.build(x)
        nl.build(x)
        estimate = adaptive.rule_12_d3(natoms=len(atoms), volume_A3=atoms.get_volume(), radius_A=D3_CUTOFF + skin)
        assert nl.num_pairs == pytest.approx(estimate.pairs, rel=0.03)
        counters = nl.counters()
        assert counters["d3_builds"] == 2
        print(f"NbCl5 3x3x3 ({len(atoms)} atoms, {atoms.get_volume() / len(atoms):.1f} A^3/atom), skin {skin} A: "
              f"{nl.num_pairs} pairs (estimate {estimate.pairs}), {counters['d3_images']} images, capacity "
              f"{nl.capacity}, build {counters['d3_last_build_time_s'] * 1e3:.1f} ms")
