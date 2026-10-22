"""The D3 kernel's Triton passes on CUDA (plan task 3b-T03, SPEC §11.2).

On rattled and strained NaCl, NbCl5 and NaNbCl6, with a brute-force half list
(``_support.d3_cases``; 3b-T02's GPU list replaces it in 3b-T04):
- float64 energy, forces and stress against ``torch-dftd`` 0.5.1 and simple-dftd3 by the runbook's
  item 1 rule (tests/cpu/test_d3_kernel.py), and against the torch tier to 1e-13;
- the negative control: without pass 3 the forces fail that comparison;
- float32 pair terms stay within the gate's floors of the float64 kernel (SPEC §12.3);
- ``evaluate_energy`` equals the full energy bit for bit in float64, and to rounding in float32;
- deterministic, with no atomics: repeated eager calls are bitwise equal, and so is a CUDA graph
  replay to the eager call, also after new positions, a new cell or a species change are written
  into the graph's inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("triton")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpu"))

from test_d3_kernel import (  # noqa: E402
    CHAIN_CASES,
    evaluate,
    inputs,
    make_kernel,
    references,  # noqa: F401  (fixture)
)

from _support.d3_cases import CASES, GATE, deviations, within  # noqa: E402

DTYPES = (torch.float64, torch.float32)


@pytest.mark.parametrize("which", ["torch-dftd", "simple-dftd3"])
@pytest.mark.parametrize("name", CASES)
def test_float64_against_both_references(name, which, references):  # noqa: F811
    atoms = CASES[name]()
    first, second, agreement = references[name]
    reference = first if which == "torch-dftd" else second
    deviation = deviations(evaluate(atoms, device="cuda"), reference, len(atoms))
    assert within(deviation, agreement), (deviation, agreement)


@pytest.mark.parametrize("name", CASES)
def test_float64_equals_the_torch_tier(name):
    atoms = CASES[name]()
    assert all(d <= 1e-13 for d in deviations(evaluate(atoms, device="cuda"), evaluate(atoms), 1))


@pytest.mark.parametrize("name", CHAIN_CASES)
def test_without_the_chain_term_the_forces_fail(name, references):  # noqa: F811
    atoms = CASES[name]()
    first, _, agreement = references[name]
    energy, forces, _ = deviations(evaluate(atoms, device="cuda", chain=False), first, len(atoms))
    assert energy <= 2 * agreement[0] + 1e-10
    assert forces > 10 * (2 * agreement[1] + 1e-8)


@pytest.mark.parametrize("name", CASES)
def test_float32_within_the_gate(name):
    atoms = CASES[name]()
    deviation = deviations(evaluate(atoms, device="cuda", dtype=torch.float32), evaluate(atoms, device="cuda"),
                           len(atoms))
    assert all(d <= g for d, g in zip(deviation, GATE)), deviation


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("name", ["NbCl5-strained", "NaNbCl6-rattled"])
def test_energy_only_equals_the_full_energy(name, dtype):
    atoms = CASES[name]()
    kernel = make_kernel(atoms, device="cuda", dtype=dtype)
    positions, cell = inputs(atoms, "cuda")
    energy, full = kernel.evaluate_energy(positions, cell), kernel.evaluate(positions, cell).energy
    if dtype == torch.float64:
        assert torch.equal(energy, full)
    else:
        assert abs(energy.item() - full.item()) <= 1e-12 * abs(full.item())


def _capture(function):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        function()  # warm-up on the side stream, as before every capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = function()
    return graph, out


def _fields(result):
    return (result,) if isinstance(result, torch.Tensor) else (result.energy, result.forces, result.virial)


@pytest.mark.parametrize("entry", ["evaluate", "evaluate_energy"])
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("name", ["NbCl5-strained", "NaNbCl6-rattled"])
def test_capture_and_replay_bitwise(name, dtype, entry):
    atoms = CASES[name]()
    kernel = make_kernel(atoms, device="cuda", dtype=dtype)
    positions, cell = inputs(atoms, "cuda")
    call = getattr(kernel, entry)
    eager = [t.clone() for t in _fields(call(positions, cell))]
    assert all(torch.equal(a, b) for a, b in zip(eager, _fields(call(positions, cell))))

    graph, captured = _capture(lambda: call(positions, cell))
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(eager, _fields(captured)))

    moved = atoms.copy()
    moved.positions += np.random.default_rng(1).normal(0.0, 0.02, moved.positions.shape)
    moved.set_cell(moved.cell.array * 1.003, scale_atoms=False)
    new_positions, new_cell = inputs(moved, "cuda")
    positions.copy_(new_positions)
    cell.copy_(new_cell)
    graph.replay()
    torch.cuda.synchronize()
    fresh = _fields(call(new_positions, new_cell))
    assert all(torch.equal(a, b) for a, b in zip(fresh, _fields(captured)))
    assert not torch.equal(fresh[0], eager[0])


def test_species_changed_in_place_replays_like_a_fresh_kernel():
    atoms = CASES["NaNbCl6-rattled"]()
    numbers = torch.tensor(atoms.numbers, device="cuda")
    kernel = make_kernel(atoms, device="cuda", numbers=numbers)
    positions, cell = inputs(atoms, "cuda")
    graph, captured = _capture(lambda: kernel.evaluate(positions, cell))
    sodium, chlorine = int(np.flatnonzero(atoms.numbers == 11)[0]), int(np.flatnonzero(atoms.numbers == 17)[0])
    numbers[sodium], numbers[chlorine] = 17, 11
    graph.replay()
    fresh = make_kernel(atoms, device="cuda", numbers=numbers.clone()).evaluate(positions, cell)
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(_fields(fresh), _fields(captured)))
