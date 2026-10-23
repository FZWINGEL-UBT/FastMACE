"""Float64 energy accumulation on CUDA (plan task 1b-T03; SPEC §7.3).

A tiny MH-1-class cuEquivariance model with a synthetic all-electron E0 table, in float32 and
float64, eager and compiled:

- the float64 energy replays in a CUDA graph as the eager call computes it (to the order of the
  GPU's atomic float64 adds);
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
    return [value.detach().clone() for value in out if value is not None]


def captured(field, positions, replays=1):
    """The outputs of ``replays`` replays of one captured graph of ``field`` (a list of output lists)."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            field(positions)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = field(positions)
    results = []
    for _ in range(replays):
        graph.replay()
        torch.cuda.synchronize()
        results.append([value.detach().clone() for value in out])
    return results


# Samples per force field for the noise of its atomic scatters: replays of its graph and eager calls.
REPLAYS, EAGER_CALLS = 8, 4
# The largest difference between the two fields may exceed the largest spread within one field by
# this factor (both are maxima over samples of the same atomic-order noise). 2-T14c on an RTX 4000
# Ada, float32 compiled, 20 runs: the ratio was at most 1.00 for forces and 1.07 for stress; the
# noise was 1.7e-7 to 3.4e-7 eV/Å, so the 1e-6 shift below exceeded the bound at least 1.5 times.
NOISE_FACTOR = 2.0


def spread(samples, index):
    """The largest max-abs difference between any two of ``samples`` (output lists) in output ``index``."""
    values = [sample[index] for sample in samples]
    return max(float((x - y).abs().max()) for i, x in enumerate(values) for y in values[i + 1:])


# The relative floor under the noise bound, by force-field precision. The two fields are different graphs, so they
# also differ by rounding that no spread within one field shows: on an A100 (sm_80; 4-T13's pilot, 23 October 2026)
# the float32 scatters were almost repeatable (noise 7.5e-10 eV/A) and the fields differed by 1.6e-9 eV/A. 1e-8 of
# the largest force is still well under float32's resolution, and far under the 1e-6 eV/A shift that must fail.
FLOOR = {torch.float32: 1e-8, torch.float64: 1e-12}


def within_noise(cross, noise, scale, dtype=torch.float64):
    """True when every cross-field difference is within the noise bound (plus ``FLOOR[dtype]`` relative)."""
    return max(cross) <= NOISE_FACTOR * noise + FLOOR[dtype] * scale


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
    eager_calls = {flag: [evaluate(field, positions) for _ in range(EAGER_CALLS)] for flag, field in fields.items()}
    replays = {flag: captured(field, positions, REPLAYS) for flag, field in fields.items()}
    eager = {flag: calls[0] for flag, calls in eager_calls.items()}
    replayed = {flag: runs[0] for flag, runs in replays.items()}
    assert eager[True][0].dtype == torch.float64
    # The float64 sums use atomic adds on the GPU: the same value up to their summation order. In
    # float32 the per-atom terms themselves come from atomic float32 scatters (4e-7 eV on 64 atoms).
    np.testing.assert_allclose(float(replayed[True][0][0]), float(eager[True][0][0]),
                               rtol=1e-13 if dtype == torch.float64 else 0,
                               atol=0 if dtype == torch.float64 else 1e-5)
    # Forces and stress: MACE's autograd of its own scalar either way (design A). Bitwise on the CPU
    # (tests/cpu/test_energy.py); here two graphs with atomic scatters, so to rounding.
    # The noise of the atomic scatters bounds the difference: the largest spread within one field
    # (its replays and eager calls) over both fields. A single replay against a single eager call
    # was too small an estimate: in float32 on an RTX 4000 Ada about 1 run in 9 failed at 4x (2-T14c).
    for index in (1, 2):
        noise = max(spread(replays[flag] + eager_calls[flag], index) for flag in (True, False))
        cross = [float((a[index] - b[index]).abs().max()) for a, b in zip(replays[True], replays[False])]
        scale = float(replayed[False][index].abs().max())
        print({"output": index, "noise": noise, "cross": max(cross)})
        assert within_noise(cross, noise, scale, dtype), (max(cross), noise)
    # The bound still catches a real change: 1e-6 eV/Å on the force component where the two fields
    # agree best (so the shift is not helped by the noise).
    difference = (replays[True][0][1] - replays[False][0][1]).abs()
    shift = torch.zeros_like(difference)
    shift.view(-1)[int(difference.argmin())] = 1e-6
    shifted = [float((a[1] + shift - b[1]).abs().max()) for a, b in zip(replays[True], replays[False])]
    noise = max(spread(replays[flag] + eager_calls[flag], 1) for flag in (True, False))
    print({"shifted": max(shifted), "bound": NOISE_FACTOR * noise})
    assert not within_noise(shifted, noise, float(replayed[False][1].abs().max()), dtype), (max(shifted), noise)
    reference = MACEForceField(cueq_calculator(model, torch.float64), atoms, neighbours=NEIGHBOURS)
    expected = float(evaluate(reference, positions)[0][0])
    error, stock_error = abs(float(eager[True][0][0]) - expected), abs(float(eager[False][0][0]) - expected)
    print({"dtype": str(dtype), "compile_mode": compile_mode, "error": error, "stock_error": stock_error})
    if dtype == torch.float32:
        assert error < stock_error
    else:
        np.testing.assert_allclose(float(eager[True][0][0]), float(eager[False][0][0]), rtol=1e-12)
