"""Idea 11 on the whole-system path (``forces.WHOLE_SYSTEM_FLOAT64_EDGES``, ``engine.float64_edge_vectors``).

MACE forms the edge vectors from the model's float32 positions, so far from the origin every vector carries the
float32 spacing of its coordinates. With the switch the force field keeps its float64 positions in the dict and the
bound forward adds a detached correction: the vectors are the float64 vectors' rounding, the gradient MACE's.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from fastmace.config import EngineConfig, NeighbourConfig
from fastmace.model import forces as forces_module
from fastmace.model.forces import MACEForceField
from fastmace.rewrites import engine

pytest.importorskip("mace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NEIGHBOURS = NeighbourConfig(skin=0.5, sort_receivers="on", gpu_neighbour_list="off")
EAGER = EngineConfig(compile_mode="none")
Z_VALUES = [8, 11, 17, 26]
OFFSET = np.array([1500.0, -700.0, 333.0])


def structure(seed=8):
    from _support.structures import nacl

    atoms = nacl(repeat=2, rattle=0.1, seed=seed)
    atoms.numbers[3] = 8
    return atoms


def strained(atoms, seed=3, amplitude=0.02):
    e = np.random.default_rng(seed).uniform(-amplitude, amplitude, (3, 3))
    out = atoms.copy()
    out.set_cell(atoms.cell.array @ (np.eye(3) + 0.5 * (e + e.T)), scale_atoms=True)
    return out


def relative(got, expected):
    got, expected = np.asarray(got, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    return float(np.abs(got - expected).max()) / float(np.abs(expected).max())


@pytest.fixture
def calculators():
    """A float64 and a float32 calculator of the same tiny model (torch's default dtype restored afterwards)."""
    from mace.calculators import MACECalculator

    from _support.tiny_models import FAMILIES, tiny_mace_model

    default = torch.get_default_dtype()
    model = tiny_mace_model(Z_VALUES, **FAMILIES["mh1"])
    try:
        f64 = MACECalculator(models=[copy.deepcopy(model)], device="cpu", default_dtype="float64")
        f32 = MACECalculator(models=[copy.deepcopy(model)], device="cpu", default_dtype="float32")
        yield f64, f32
    finally:
        torch.set_default_dtype(default)


def test_vectors_are_the_float64_vectors_rounding_far_from_the_origin():
    rng = np.random.default_rng(5)
    cell64 = torch.tensor(np.diag([23.4, 25.1, 22.7]) + rng.uniform(-1, 1, (3, 3)), dtype=torch.float64)
    n, e = 400, 3000
    positions64 = torch.tensor(rng.uniform(0, 20, (n, 3)) + OFFSET, dtype=torch.float64)
    edge_index = torch.tensor(rng.integers(0, n, (2, e)))
    unit = torch.tensor(rng.integers(-1, 2, (e, 3)), dtype=torch.float32)
    sender, receiver = edge_index
    positions32 = positions64.to(torch.float32).requires_grad_(True)
    shifts32 = (unit.double() @ cell64).to(torch.float32)
    plain = positions32[receiver] - positions32[sender] + shifts32  # MACE's
    exact = positions64[receiver] - positions64[sender] + unit.double() @ cell64
    vectors, lengths = engine.float64_edge_vectors(plain, positions64, edge_index, unit, cell64)
    assert vectors.dtype == torch.float32
    plain_error = float((plain.detach().double() - exact).abs().max())
    error = float((vectors.detach().double() - exact).abs().max())
    assert plain_error > 1e-5  # the float32 spacing at 1,500 A
    assert error <= 4e-6  # the vectors' own rounding (|v| up to ~60 A here)
    assert torch.allclose(lengths, torch.linalg.vector_norm(vectors, dim=-1, keepdim=True))
    # The gradient is MACE's: the correction is a constant.
    weights = torch.randn(e, 3, dtype=torch.float32)
    (got,) = torch.autograd.grad((vectors * weights).sum(), positions32)
    positions32.grad = None
    (want,) = torch.autograd.grad((plain * weights).sum(), positions32)
    assert torch.equal(got, want)


def test_the_correction_strains_with_its_edge():
    rng = np.random.default_rng(7)
    vectors = torch.tensor(rng.uniform(-3, 3, (50, 3)), dtype=torch.float64)
    positions64 = torch.tensor(rng.uniform(0, 5, (10, 3)) + OFFSET, dtype=torch.float64)
    edge_index = torch.tensor(rng.integers(0, 10, (2, 50)))
    unit = torch.zeros(50, 3)
    cell64 = torch.eye(3, dtype=torch.float64) * 30
    strain = torch.tensor(rng.uniform(-0.01, 0.01, (1, 3, 3)), dtype=torch.float64)
    plain, _ = engine.float64_edge_vectors(vectors, positions64, edge_index, unit, cell64)
    got, _ = engine.float64_edge_vectors(vectors, positions64, edge_index, unit, cell64, strain)
    symmetric = 0.5 * (strain[0] + strain[0].T)
    correction = plain - vectors
    assert torch.allclose(got, vectors + correction @ (torch.eye(3, dtype=torch.float64) + symmetric))


@pytest.mark.parametrize("stress", [False, True], ids=["forces", "stress"])
def test_float32_call_far_from_the_origin_is_the_origin_call(calculators, stress, monkeypatch):
    """A float32 force field 1,500 A from the origin: with the switch its energy, forces and stress are the float64
    model's as closely as at the origin; without it the forces carry the coordinates' float32 spacing."""
    f64, f32 = calculators
    atoms = structure()
    moved = atoms.copy()
    moved.positions = moved.positions + OFFSET
    reference = MACEForceField(f64, atoms, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=stress)
    cases = [atoms] + ([strained(atoms)] if stress else [])
    expected = [reference.evaluate(current, stress=stress) for current in cases]
    errors = {}
    for switch in (False, True):
        monkeypatch.setattr(forces_module, "WHOLE_SYSTEM_FLOAT64_EDGES", switch)
        for label, frame in (("origin", atoms), ("moved", moved)):
            field = MACEForceField(f32, frame, neighbours=NEIGHBOURS, engine=EAGER, compute_stress=stress)
            assert field.float64_edges is switch
            assert (engine.POSITIONS64_KEY in field.data) is switch
            for k, (current, want) in enumerate(zip(cases, expected)):
                current = current.copy()
                if label == "moved":
                    current.positions = current.positions + OFFSET
                got = field.evaluate(current, stress=stress)
                errors[switch, label, k] = (abs(got[0] - want[0]) / abs(want[0]), relative(got[1], want[1]),
                                            relative(got[2], want[2]) if stress else 0.0)
    for k in range(len(cases)):
        origin, moved_on, moved_off = errors[True, "origin", k], errors[True, "moved", k], errors[False, "moved", k]
        assert moved_off[1] > 1e-5  # the float32 spacing at 1,500 A in the forces
        assert moved_on[1] <= max(4 * origin[1], 2e-6)
        assert moved_on[0] <= 1e-6 and moved_on[2] <= max(4 * origin[2], 2e-6)
        assert moved_on[1] < moved_off[1] / 5


def test_released_whole_system_drops_the_float64_positions(calculators, monkeypatch):
    _, f32 = calculators
    monkeypatch.setattr(forces_module, "WHOLE_SYSTEM_FLOAT64_EDGES", True)
    field = MACEForceField(f32, structure(), neighbours=NEIGHBOURS, engine=EAGER)
    assert field.float64_edges
    field.release_node_attrs()
    field.release_whole_system()
    assert engine.POSITIONS64_KEY not in field.data and engine.CELL64_KEY not in field.data
    assert not field.float64_edges


def test_float64_models_and_the_default_keep_mace_vectors(calculators):
    f64, f32 = calculators
    assert forces_module.WHOLE_SYSTEM_FLOAT64_EDGES is False
    for calc in (f64, f32):
        field = MACEForceField(calc, structure(), neighbours=NEIGHBOURS, engine=EAGER)
        assert not field.float64_edges and engine.POSITIONS64_KEY not in field.data
