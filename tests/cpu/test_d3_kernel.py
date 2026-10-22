"""The D3 kernel's three passes on the CPU (plan task 3b-T03, SPEC §11.2).

On rattled and strained NaCl, NbCl5 and NaNbCl6 (``_support.d3_cases``), with a brute-force half list:
- float64 energy, forces and stress equal ``torch-dftd`` 0.5.1 (Becke–Johnson, the record's
  parameters) to ``TORCH_DFTD``. Measured: 4e-16 eV, 2e-16 eV/Å, 1e-17 eV/Å³;
- and simple-dftd3 (``dftd3``) to twice its own distance from ``torch-dftd``, i.e. to the two
  references' agreement with each other;
- forces and stress are the derivatives of the energy (central differences), independent of
  either reference;
- without pass 3 (the CN chain term) the forces fail the ``torch-dftd`` comparison;
- float32 pair terms stay within the gate's floors of float64 (SPEC §12.3);
- the list's order and padding slots do not matter.

The Triton kernels run here through Triton's interpreter, in a fresh process, when Triton is
installed (the GPU environments; the CPU environment skips). On the GPU: tests/gpu/test_d3_kernel_gpu.py.
The references and the tables need torch-dftd, so the tests skip without it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("torch_dftd")

from _support.d3_cases import CASES, half_list, torch_dftd  # noqa: E402
from fastmace.longrange.d3 import D3  # noqa: E402
from fastmace.longrange.d3_kernel import D3Kernel  # noqa: E402

RECORD = D3(functional="r2scan")
SKIN = 1.0  # Å added to the list cutoff, so the kernel's own r² mask does the cutting

# float64 against torch-dftd: energy eV, max force component eV/Å, max stress component eV/Å³
TORCH_DFTD = (1e-10, 1e-10, 1e-12)
# The gate's absolute floors (SPEC §12.3): energy per atom eV, force eV/Å, stress eV/Å³
GATE = (1e-5, 1e-5, 1e-7)
# Without the chain term the NbCl5 and NaNbCl6 forces move by 3e-5 and 1.6e-4 eV/Å (NaCl: its
# C6 weights sit on one reference, so dE/dCN is ~0 and the chain term vanishes)
CHAIN_CASES = ("NbCl5-rattled", "NbCl5-strained", "NaNbCl6-rattled", "NaNbCl6-strained")


def evaluate(atoms, *, dtype=torch.float64, chain=True, device="cpu", shuffle=False, padding=0):
    i, j, images = half_list(atoms, RECORD.pair_cutoff_bohr * RECORD.autoang + SKIN)
    if shuffle:
        order = np.random.default_rng(0).permutation(len(i))
        i, j, images = i[order], j[order], images[order]
    if padding:
        n = len(atoms)
        i = np.concatenate([i, np.full(padding, n)])
        j = np.concatenate([j, np.full(padding, n)])
        images = np.concatenate([images, np.zeros((padding, 3), dtype=np.int64)])
    kernel = D3Kernel(RECORD, atoms.numbers, device=device, dtype=dtype, chain=chain)
    kernel.prepare(torch.as_tensor(i), torch.as_tensor(j), torch.as_tensor(images))
    positions = torch.tensor(atoms.positions, device=device)
    cell = torch.tensor(atoms.cell.array, device=device)
    energy, forces, virial = kernel(positions, cell)
    stress = -virial.cpu().numpy() / atoms.get_volume()
    return energy.item(), forces.cpu().double().numpy(), stress


def deviations(ours, reference):
    return (abs(ours[0] - reference[0]), np.abs(ours[1] - reference[1]).max(), np.abs(ours[2] - reference[2]).max())


@pytest.fixture(scope="module")
def references():
    return {name: torch_dftd(make(), RECORD) for name, make in CASES.items()}


@pytest.mark.parametrize("name", CASES)
def test_float64_matches_torch_dftd(name, references):
    assert all(d <= tol for d, tol in zip(deviations(evaluate(CASES[name]()), references[name]), TORCH_DFTD))


def simple_dftd3(atoms):
    """Energy, forces and stress from simple-dftd3 with the record's parameters and cutoffs, ATM off."""
    from dftd3.interface import DispersionModel, RationalDampingParam

    from fastmace.longrange.d3_kernel import bj_parameters

    bohr = RECORD.autoang
    model = DispersionModel(atoms.numbers, atoms.positions / bohr, atoms.cell.array / bohr, atoms.pbc)
    model.set_realspace_cutoff(RECORD.pair_cutoff_bohr, RECORD.pair_cutoff_bohr, RECORD.cn_cutoff_bohr)
    result = model.get_dispersion(RationalDampingParam(**bj_parameters(RECORD), s9=0.0), grad=True)
    energy = result["energy"] * RECORD.autoev
    forces = -result["gradient"] * RECORD.autoev / bohr
    stress = result["virial"] * RECORD.autoev / atoms.get_volume()
    return energy, forces, stress


@pytest.mark.parametrize("name", CASES)
def test_float64_matches_simple_dftd3_to_the_references_agreement(name, references):
    pytest.importorskip("dftd3.interface")
    atoms = CASES[name]()
    other = simple_dftd3(atoms)
    agreement = deviations(other, references[name])
    ours = deviations(evaluate(atoms), other)
    assert all(d <= max(2.0 * a, floor) for d, a, floor in zip(ours, agreement, TORCH_DFTD)), (ours, agreement)


@pytest.mark.parametrize("name", ["NbCl5-strained", "NaNbCl6-rattled"])
def test_forces_and_stress_are_energy_derivatives(name):
    atoms = CASES[name]()
    _, forces, stress = evaluate(atoms)
    h = 1e-4
    for atom, axis in [(0, 0), (1, 2), (len(atoms) - 1, 1)]:
        energies = []
        for sign in (1, -1):
            moved = atoms.copy()
            moved.positions[atom, axis] += sign * h
            energies.append(evaluate(moved)[0])
        assert abs(-(energies[0] - energies[1]) / (2 * h) - forces[atom, axis]) < 1e-8
    # A strain moves every pair, and the 40-bohr cutoff has no smoothing (as in torch-dftd): at 1e-4
    # pairs crossing it shift the difference by ~1e-5 eV/Å³, at 1e-7 almost never.
    h = 1e-7
    for a, b in [(0, 0), (1, 2), (2, 1)]:
        energies = []
        for sign in (1, -1):
            strain = np.eye(3)
            strain[a, b] += sign * h
            moved = atoms.copy()
            moved.set_cell(atoms.cell.array @ strain, scale_atoms=True)
            energies.append(evaluate(moved)[0])
        assert abs((energies[0] - energies[1]) / (2 * h) / atoms.get_volume() - stress[a, b]) < 1e-10


@pytest.mark.parametrize("name", CHAIN_CASES)
def test_without_the_chain_term_the_forces_fail(name, references):
    energy, forces, stress = deviations(evaluate(CASES[name](), chain=False), references[name])
    assert energy <= TORCH_DFTD[0]  # the chain term changes no energy
    assert forces > 100 * TORCH_DFTD[1]


@pytest.mark.parametrize("name", CASES)
def test_float32_within_the_gate(name):
    atoms = CASES[name]()
    single, double = evaluate(atoms, dtype=torch.float32), evaluate(atoms)
    energy, forces, stress = deviations(single, double)
    assert energy / len(atoms) <= GATE[0] and forces <= GATE[1] and stress <= GATE[2]


def test_order_and_padding_do_not_matter():
    atoms = CASES["NbCl5-rattled"]()
    reference = evaluate(atoms)
    shuffled = evaluate(atoms, shuffle=True, padding=37)
    assert all(d <= 1e-13 for d in deviations(shuffled, reference))


DRIVER = textwrap.dedent("""
    import json, sys
    sys.path[:0] = [{tests!r}]
    import fastmace  # noqa: F401  (before torch)
    import numpy as np, torch
    from _support.d3_cases import CASES, half_list
    from fastmace.longrange.d3 import D3
    from fastmace.longrange.d3_kernel import D3Kernel

    record = D3(functional="r2scan")
    out = {{}}
    for name in ("NaNbCl6-strained", "NaCl-rattled"):
        atoms = CASES[name]()
        i, j, images = half_list(atoms, record.pair_cutoff_bohr * record.autoang + 1.0)
        for dtype in (torch.float64, torch.float32):
            for chain in (True, False):
                kernel = D3Kernel(record, atoms.numbers, device="cpu", dtype=dtype, chain=chain)
                kernel.prepare(torch.as_tensor(i), torch.as_tensor(j), torch.as_tensor(images))
                positions, cell = torch.tensor(atoms.positions), torch.tensor(atoms.cell.array)
                ours, theirs = kernel._triton(positions, cell), kernel._torch(positions, cell)
                out[f"{{name}} {{dtype}} {{chain}}"] = [float((a.double() - b.double()).abs().max())
                                                       for a, b in zip(ours, theirs)]
    print(json.dumps(out))
""")


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="Triton is not installed (CPU environment)")
def test_triton_kernels_through_the_interpreter():
    tests = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    run = subprocess.run([sys.executable, "-c", DRIVER.format(tests=tests)], env=env, capture_output=True,
                         text=True, timeout=1200)
    assert run.returncode == 0, run.stderr[-3000:]
    result = json.loads(run.stdout.strip().splitlines()[-1])
    for key, (energy, forces, virial) in result.items():
        tolerance = 1e-12 if "float64" in key else 1e-5  # float32: the two paths round differently
        assert energy <= tolerance and forces <= tolerance and virial <= tolerance, (key, result[key])


def test_species_changed_in_place_equals_a_fresh_kernel():
    """A species event writes ``numbers`` in place; the kernel reads Z through it (conductor's choice 5)."""
    atoms = CASES["NaNbCl6-rattled"]()
    i, j, images = (torch.as_tensor(a) for a in half_list(atoms, RECORD.pair_cutoff_bohr * RECORD.autoang + SKIN))
    positions, cell = torch.tensor(atoms.positions), torch.tensor(atoms.cell.array)
    numbers = torch.tensor(atoms.numbers)
    kernel = D3Kernel(RECORD, numbers, device="cpu")
    kernel.prepare(i, j, images)
    before = kernel(positions, cell)
    numbers[0], numbers[2] = 17, 11  # Na <-> one Cl
    after = kernel(positions, cell)
    fresh = D3Kernel(RECORD, numbers.clone(), device="cpu")
    fresh.prepare(i, j, images)
    assert all(torch.equal(a, b) for a, b in zip(after, fresh(positions, cell)))
    assert not torch.equal(before[0], after[0])
