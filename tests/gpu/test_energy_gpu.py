"""Float64 energy accumulation on CUDA (plan task 1b-T03; SPEC §7.3).

A tiny MH-1-class cuEquivariance model with a synthetic all-electron E0 table, in float32 and
float64, eager and compiled:

- the float64 energy replays in a CUDA graph exactly as the eager call computes it;
- forces and stress equal those of 1a's energy sum (``float64_energy=False``) to rounding;
- in float32 the energy is closer to the float64 model's than the stock float32 sum.

The measurements on released models (POLAR-1-S and OMOL-0 XL on NbCl5, MH-1's change, the step
time) are ``tools/energy64_gpu.py``'s; results in docs/validation/1b-T03-energy.md.
"""

import copy

import numpy as np
import pytest
import torch

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace.config import EngineConfig, NeighbourConfig
from fastmace.model import compile as fc
from fastmace.model.forces import MACEForceField
from fastmace.neighbors.buckets import edge_bucket, round_up_edges

pytest.importorskip("cuequivariance_torch")
torch.serialization.add_safe_globals([slice])

Z_VALUES = [8, 11, 17, 26]
E0 = [-2048.1234567891, -4412.9876543219, -12522.6512345678, -34510.4321987654]
NEIGHBOURS = NeighbourConfig(skin=0.5, sort_receivers="off", gpu_neighbour_list="on")


def structure():
    atoms = nacl(repeat=2, rattle=0.1, seed=8)
    atoms.numbers[3] = 8
    return atoms


def source_model():
    model = tiny_mace_model(Z_VALUES, **FAMILIES["mh1"])
    with torch.no_grad():
        model.atomic_energies_fn.atomic_energies.copy_(torch.tensor(E0, dtype=torch.float64))
    return model


def cueq_calculator(model, dtype):
    """MACE's calculator of a cuEq conversion of ``model`` on the GPU in ``dtype``."""
    from mace.calculators import MACECalculator
    from mace.cli.convert_e3nn_cueq import run

    calc = MACECalculator(models=[copy.deepcopy(model)], device="cuda", default_dtype="float64")
    calc.models = [run(calc.models[0].to("cuda", dtype), device="cuda", return_model=True)]
    return calc


def evaluate(field, positions):
    field.compact(positions)
    count = int(field.edge_count)
    field.use_edges(round_up_edges(count, field.edge_capacity, edge_bucket(count, 0.25, 256)))
    out = field(positions)
    return [value.detach().clone() for value in out]


def captured(field, positions):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            field(positions)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = field(positions)
    graph.replay()
    torch.cuda.synchronize()
    return [value.detach().clone() for value in out]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("compile_mode", ["none", "default"])
def test_float64_energy_replays_and_keeps_forces(dtype, compile_mode):
    model, atoms = source_model(), structure()
    calc = cueq_calculator(model, dtype)
    exact = torch.tensor(E0, dtype=torch.float64)
    fc.install_engine_patches()  # capture-safe readout mask (the engine's job; undone by conftest)
    fields = {flag: MACEForceField(calc, atoms, neighbours=NEIGHBOURS, compute_stress=True,
                                   engine=EngineConfig(compile_mode=compile_mode),
                                   compile_cache=fc.CompiledModelCache(), float64_energy=flag,
                                   atomic_energies=exact if flag else None)
              for flag in (True, False)}
    positions = torch.tensor(atoms.positions + 0.03, device="cuda")
    eager = {flag: evaluate(field, positions) for flag, field in fields.items()}
    replayed = {flag: captured(field, positions) for flag, field in fields.items()}
    assert eager[True][0].dtype == torch.float64
    assert torch.equal(replayed[True][0], eager[True][0])
    # Forces and stress: MACE's autograd of its own scalar either way (design A). Bitwise on the CPU
    # (tests/cpu/test_energy.py); here two graphs with atomic scatters, so to rounding.
    tolerance = 1e-12 if dtype == torch.float64 else 1e-5
    for index in (1, 2):
        a, b = replayed[True][index], replayed[False][index]
        assert float((a - b).abs().max()) <= tolerance * float(b.abs().max())
    reference = MACEForceField(cueq_calculator(model, torch.float64), atoms, neighbours=NEIGHBOURS)
    expected = float(evaluate(reference, positions)[0][0])
    error, stock_error = abs(float(eager[True][0][0]) - expected), abs(float(eager[False][0][0]) - expected)
    print({"dtype": str(dtype), "compile_mode": compile_mode, "error": error, "stock_error": stock_error})
    if dtype == torch.float32:
        assert error < stock_error
    else:
        np.testing.assert_allclose(float(eager[True][0][0]), float(eager[False][0][0]), rtol=1e-12)
