"""The D3 kernel's three passes on the CPU: the torch tier (plan task 3b-T03, SPEC §11.2).

On rattled and strained NaCl, NbCl5 and NaNbCl6 (``_support.d3_cases``), with a brute-force half list:
- float64 energy, forces and stress against ``torch-dftd`` 0.5.1 **and** simple-dftd3, by the
  runbook's item 1 rule: each deviation at most twice the references' deviation from each other,
  plus the rounding floors (1e-10 eV/atom, 1e-8 eV/Å, 1e-9 eV/Å³);
- forces and stress are the derivatives of the energy (central differences), independent of
  either reference;
- the negative control: without pass 3 (the CN chain term) the forces fail the comparison;
- float32 pair terms stay within the gate's floors of float64 (SPEC §12.3);
- ``evaluate_energy`` equals the full evaluation's energy bit for bit, in float64 and float32;
- the list's order and padding slots, and ``virial=False``, change nothing;
- a species change written into ``numbers`` equals a fresh kernel (conductor's choice 5);
- tables that are not ``d3.REFERENCE_TABLES``' are refused.

The Triton kernels run here through Triton's interpreter, in a fresh process, when Triton is
installed (the GPU environments; the CPU environment skips). On the GPU: tests/gpu/test_d3_kernel_gpu.py.
The reference comparisons need torch-dftd and simple-dftd3 (``fastmace[d3-reference]``); the kernel
itself needs neither, as its tables ship with FastMACE.
"""

from __future__ import annotations

import hashlib
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

from _support.d3_cases import CASES, GATE, deviations, half_list, within
from fastmace.longrange import d3
from fastmace.longrange import d3_kernel as kernel_module
from fastmace.longrange.d3_kernel import D3Kernel

RECORD = d3.D3(functional="r2scan")
SKIN = 1.0  # Å added to the list cutoff, so the kernel's own r² mask does the cutting
# Without the chain term the NbCl5 and NaNbCl6 forces move by 1e-5 to 1e-4 eV/Å. NaCl's C6 weights
# sit on one reference, so dE/dCN is ~0 there and the chain term vanishes.
CHAIN_CASES = ("NbCl5-rattled", "NbCl5-strained", "NaNbCl6-rattled", "NaNbCl6-strained")


def make_kernel(atoms, *, dtype=torch.float64, chain=True, device="cpu", shuffle=False, padding=0, numbers=None):
    i, j, index, table = half_list(atoms, d3.pair_cutoff_A(RECORD) + SKIN)
    if shuffle:
        order = np.random.default_rng(0).permutation(len(i))
        i, j, index = i[order], j[order], index[order]
    if padding:
        n = len(atoms)
        i = np.concatenate([i, np.full(padding, n)])
        j = np.concatenate([j, np.full(padding, n)])
        index = np.concatenate([index, np.zeros(padding, dtype=np.int32)])
    numbers = torch.tensor(atoms.numbers, device=device) if numbers is None else numbers
    kernel = D3Kernel(RECORD, numbers, device=device, dtype=dtype, chain=chain)
    kernel.prepare(torch.as_tensor(i), torch.as_tensor(j), torch.as_tensor(index), torch.as_tensor(table))
    return kernel


def inputs(atoms, device="cpu"):
    return torch.tensor(atoms.positions, device=device), torch.tensor(atoms.cell.array, device=device)


def evaluate(atoms, **options):
    device = options.get("device", "cpu")
    result = make_kernel(atoms, **options).evaluate(*inputs(atoms, device))
    return (result.energy.item(), result.forces.cpu().double().numpy(),
            result.stress(atoms.get_volume()).cpu().numpy())


@pytest.fixture(scope="module")
def references():
    """Per case: torch-dftd's result, simple-dftd3's, and their deviation from each other."""
    pytest.importorskip("torch_dftd")
    pytest.importorskip("dftd3.interface")
    from fastmace.reference.d3 import simple_dftd3, torch_dftd3

    out = {}
    for name, make in CASES.items():
        atoms = make()
        first, second = torch_dftd3(atoms, RECORD), simple_dftd3(atoms, RECORD)
        out[name] = (first, second, deviations(first, second, len(atoms)))
    return out


@pytest.mark.parametrize("which", ["torch-dftd", "simple-dftd3"])
@pytest.mark.parametrize("name", CASES)
def test_float64_against_both_references(name, which, references):
    atoms = CASES[name]()
    first, second, agreement = references[name]
    reference = first if which == "torch-dftd" else second
    deviation = deviations(evaluate(atoms), reference, len(atoms))
    assert within(deviation, agreement), (deviation, agreement)


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
    atoms = CASES[name]()
    first, _, agreement = references[name]
    energy, forces, _ = deviations(evaluate(atoms, chain=False), first, len(atoms))
    assert energy <= 2 * agreement[0] + 1e-10  # the chain term changes no energy
    assert forces > 10 * (2 * agreement[1] + 1e-8)


@pytest.mark.parametrize("name", CASES)
def test_float32_within_the_gate(name):
    atoms = CASES[name]()
    deviation = deviations(evaluate(atoms, dtype=torch.float32), evaluate(atoms), len(atoms))
    assert all(d <= g for d, g in zip(deviation, GATE)), deviation


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=str)
def test_energy_only_equals_the_full_energy(dtype):
    atoms = CASES["NbCl5-strained"]()
    kernel = make_kernel(atoms, dtype=dtype)
    positions, cell = inputs(atoms)
    energy = kernel.evaluate_energy(positions, cell)
    assert energy.dtype == torch.float64 and energy.shape == ()
    assert torch.equal(energy, kernel.evaluate(positions, cell).energy)


def test_order_padding_and_virial_flag_change_nothing():
    atoms = CASES["NbCl5-rattled"]()
    reference = evaluate(atoms)
    shuffled = evaluate(atoms, shuffle=True, padding=37)
    assert all(d <= 1e-13 for d in deviations(shuffled, reference, 1))
    kernel = make_kernel(atoms)
    full, light = kernel.evaluate(*inputs(atoms)), kernel.evaluate(*inputs(atoms), virial=False)
    assert light.virial is None
    assert torch.equal(full.energy, light.energy) and torch.equal(full.forces, light.forces)


def test_species_changed_in_place_equals_a_fresh_kernel():
    atoms = CASES["NaNbCl6-rattled"]()
    numbers = torch.tensor(atoms.numbers)
    kernel = make_kernel(atoms, numbers=numbers)
    positions, cell = inputs(atoms)
    before = kernel.evaluate(positions, cell)
    sodium, chlorine = int(np.flatnonzero(atoms.numbers == 11)[0]), int(np.flatnonzero(atoms.numbers == 17)[0])
    numbers[sodium], numbers[chlorine] = 17, 11
    after = kernel.evaluate(positions, cell)
    fresh = make_kernel(atoms, numbers=numbers.clone()).evaluate(positions, cell)
    assert all(torch.equal(a, b) for a, b in zip((after.energy, after.forces, after.virial),
                                                 (fresh.energy, fresh.forces, fresh.virial)))
    assert not torch.equal(before.energy, after.energy)


def test_the_shipped_tables_are_torch_dftds():
    data = kernel_module.TABLES.read_bytes()
    assert len(data) == 690_158
    assert d3.REFERENCE_TABLES.endswith("sha256=" + hashlib.sha256(data).hexdigest())
    assert (kernel_module.TABLES.parent / "LICENSE").read_text().startswith("MIT License")
    if importlib.util.find_spec("torch_dftd") is not None:
        theirs = Path(importlib.util.find_spec("torch_dftd").origin).parent / "nn" / "params" / "dftd3_params.npz"
        assert theirs.read_bytes() == data


def test_the_tables_are_package_data():
    """pyproject.toml ships the tables, their licence and notice with ``fastmace.longrange`` (no [d3] needed)."""
    pyproject = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    line = next(line for line in pyproject.splitlines() if line.startswith('"fastmace.longrange" = '))
    for name in ("data/dftd3_params.npz", "data/LICENSE", "data/NOTICE.md"):
        assert f'"{name}"' in line
        assert (kernel_module.TABLES.parent.parent / name).is_file()


def test_other_tables_are_refused(monkeypatch):
    monkeypatch.setattr(d3, "REFERENCE_TABLES", "torch-dftd-0.5.1:dftd3_params.npz:sha256=" + "0" * 64)
    with pytest.raises(RuntimeError, match="reinstall FastMACE"):
        kernel_module.reference_tables.__wrapped__()


DRIVER = textwrap.dedent("""
    import json, sys
    sys.path[:0] = [{tests!r}]
    import fastmace  # noqa: F401  (before torch)
    import numpy as np, torch
    from _support.d3_cases import CASES, half_list
    from fastmace.longrange import d3
    from fastmace.longrange.d3_kernel import D3Kernel

    record = d3.D3(functional="r2scan")
    out = {{}}
    for name in ("NaNbCl6-strained", "NaCl-rattled"):
        atoms = CASES[name]()
        lists = [torch.as_tensor(a) for a in half_list(atoms, d3.pair_cutoff_A(record) + 1.0)]
        for dtype in (torch.float64, torch.float32):
            for chain in (True, False):
                kernel = D3Kernel(record, atoms.numbers, device="cpu", dtype=dtype, chain=chain)
                kernel.prepare(*lists)
                positions, cell = torch.tensor(atoms.positions), torch.tensor(atoms.cell.array)
                ours = kernel._triton(positions, cell, False, True)
                theirs = kernel._torch(positions, cell, False, True)
                energy = kernel._triton(positions, cell, True, False)[0]
                out[f"{{name}} {{dtype}} {{chain}}"] = [float((a.double() - b.double()).abs().max())
                                                       for a, b in zip(ours, theirs)] + [
                    bool(torch.equal(energy, ours[0]))]
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
    for key, (energy, forces, virial, energy_only_equal) in result.items():
        tolerance = 1e-12 if "float64" in key else 1e-5  # float32: the two paths round differently
        assert energy <= tolerance and forces <= tolerance and virial <= tolerance, (key, result[key])
        assert energy_only_equal, key
