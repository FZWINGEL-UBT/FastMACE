"""The D3 kernel's Triton passes on CUDA (plan task 3b-T03, SPEC §11.2).

On rattled and strained NaCl, NbCl5 and NaNbCl6, with a brute-force half list
(``_support.d3_cases``; 3b-T02's GPU list replaces it in 3b-T04):
- float64 energy, forces and stress equal ``torch-dftd`` 0.5.1 to ``TORCH_DFTD`` (the CPU file's
  tolerances), and simple-dftd3 to the two references' agreement with each other;
- without pass 3 the forces fail that comparison;
- float32 pair terms stay within the gate's floors of the float64 kernel (SPEC §12.3);
- deterministic, with no atomics: repeated eager calls are bitwise equal, and so is a CUDA graph
  replay to the eager call, also after new positions and a new cell are copied into the graph's
  inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("torch_dftd")
pytest.importorskip("triton")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpu"))

from test_d3_kernel import (  # noqa: E402
    CHAIN_CASES,
    GATE,
    RECORD,
    SKIN,
    TORCH_DFTD,
    deviations,
    evaluate,
    references,  # noqa: F401  (fixture)
    simple_dftd3,
)

from _support.d3_cases import CASES, half_list  # noqa: E402
from fastmace.longrange.d3_kernel import D3Kernel  # noqa: E402

DTYPES = (torch.float64, torch.float32)


@pytest.mark.parametrize("name", CASES)
def test_float64_matches_torch_dftd(name, references):  # noqa: F811
    assert all(d <= tol for d, tol in zip(deviations(evaluate(CASES[name](), device="cuda"), references[name]),
                                          TORCH_DFTD))


@pytest.mark.parametrize("name", CASES)
def test_float64_matches_simple_dftd3_to_the_references_agreement(name, references):  # noqa: F811
    pytest.importorskip("dftd3.interface")
    atoms = CASES[name]()
    other = simple_dftd3(atoms)
    agreement = deviations(other, references[name])
    ours = deviations(evaluate(atoms, device="cuda"), other)
    assert all(d <= max(2.0 * a, floor) for d, a, floor in zip(ours, agreement, TORCH_DFTD)), (ours, agreement)


@pytest.mark.parametrize("name", CHAIN_CASES)
def test_without_the_chain_term_the_forces_fail(name, references):  # noqa: F811
    energy, forces, _ = deviations(evaluate(CASES[name](), device="cuda", chain=False), references[name])
    assert energy <= TORCH_DFTD[0]
    assert forces > 100 * TORCH_DFTD[1]


@pytest.mark.parametrize("name", CASES)
def test_float32_within_the_gate(name):
    atoms = CASES[name]()
    energy, forces, stress = deviations(evaluate(atoms, device="cuda", dtype=torch.float32),
                                        evaluate(atoms, device="cuda"))
    assert energy / len(atoms) <= GATE[0] and forces <= GATE[1] and stress <= GATE[2]


@pytest.mark.parametrize("name", CASES)
def test_gpu_equals_the_cpu_passes(name):
    atoms = CASES[name]()
    assert all(d <= 1e-13 for d in deviations(evaluate(atoms, device="cuda"), evaluate(atoms)))


def _kernel(atoms, dtype):
    i, j, images = half_list(atoms, RECORD.pair_cutoff_bohr * RECORD.autoang + SKIN)
    kernel = D3Kernel(RECORD, atoms.numbers, device="cuda", dtype=dtype)
    kernel.prepare(torch.as_tensor(i), torch.as_tensor(j), torch.as_tensor(images))
    return kernel


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("name", ["NbCl5-strained", "NaNbCl6-rattled"])
def test_capture_and_replay_bitwise(name, dtype):
    atoms = CASES[name]()
    kernel = _kernel(atoms, dtype)
    positions = torch.tensor(atoms.positions, device="cuda")
    cell = torch.tensor(atoms.cell.array, device="cuda")
    eager = [t.clone() for t in kernel(positions, cell)]
    again = kernel(positions, cell)
    assert all(torch.equal(a, b) for a, b in zip(eager, again))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        kernel(positions, cell)  # warm-up on the side stream, as before every capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = kernel(positions, cell)
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(eager, captured))

    rng = np.random.default_rng(1)
    moved = atoms.copy()
    moved.positions += rng.normal(0.0, 0.02, moved.positions.shape)
    moved.set_cell(moved.cell.array * 1.003, scale_atoms=False)
    positions.copy_(torch.tensor(moved.positions, device="cuda"))
    cell.copy_(torch.tensor(moved.cell.array, device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    fresh = kernel(positions.clone(), cell.clone())
    assert all(torch.equal(a, b) for a, b in zip(fresh, captured))
    assert not torch.equal(fresh[1], eager[1])


def test_species_changed_in_place_replays_like_a_fresh_kernel():
    atoms = CASES["NaNbCl6-rattled"]()
    i, j, images = (torch.as_tensor(a) for a in half_list(atoms, RECORD.pair_cutoff_bohr * RECORD.autoang + SKIN))
    positions = torch.tensor(atoms.positions, device="cuda")
    cell = torch.tensor(atoms.cell.array, device="cuda")
    numbers = torch.tensor(atoms.numbers, device="cuda")
    kernel = D3Kernel(RECORD, numbers, device="cuda")
    kernel.prepare(i, j, images)
    kernel(positions, cell)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = kernel(positions, cell)
    numbers[0], numbers[2] = 17, 11
    graph.replay()
    fresh = D3Kernel(RECORD, numbers.clone(), device="cuda")
    fresh.prepare(i, j, images)
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(fresh(positions, cell), captured))
