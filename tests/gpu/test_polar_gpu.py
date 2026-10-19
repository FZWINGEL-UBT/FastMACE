"""The GPU-only PolarMACE path on CUDA (plan task 1a-T10, SPEC §11.1).

- the fixed k-vector list against graph_longrange on the GPU, in float32 and float64;
- a tiny PolarMACE: the enabled force and stress call runs without host synchronisation and is
  captured in a CUDA graph whose replay equals the eager call;
- POLAR-1-S (``FASTMACE_MODELS``), plain cuEquivariance float64, uncompiled: MACE's energy and
  forces, the stress against finite differences of the energy (MACE issue 1642), the uvu rewrite's
  self-check;
- POLAR-1-S hybrid float32 in a fresh interpreter (``FASTMACE_HYBRID_PYTHON``, as the loading
  tests): the enabled, uvu-rewritten model against the released one.

Workstations: run with the NAS ``mace_cueq_latest`` (graph-longrange 0.4.0) and the hybrid child
in ``mace_cueq_oeq``.
"""

import copy
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

from _support.structures import nacl
from _support.tiny_models import polar_available, tiny_family_calculator
from fastmace.longrange import polar
from fastmace.model import labels as table
from fastmace.rewrites.polar import uvu

torch.serialization.add_safe_globals([slice])  # e3nn 0.4.4's constants under torch's weights_only load

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODELS = os.environ.get("FASTMACE_MODELS")
HYBRID_PYTHON = os.environ.get("FASTMACE_HYBRID_PYTHON") or sys.executable
DEVICE = "cuda:0"

pytestmark = pytest.mark.skipif(not polar_available(), reason="PolarMACE needs graph_longrange")


def _hybrid_has(module):
    if HYBRID_PYTHON == sys.executable:
        return importlib.util.find_spec(module) is not None
    code = f"import importlib.util, sys; sys.exit(importlib.util.find_spec({module!r}) is None)"
    return subprocess.run([HYBRID_PYTHON, "-c", code], capture_output=True, timeout=120, check=False).returncode == 0


def has_checkpoint(label):
    return bool(MODELS) and (Path(MODELS) / table.LABELS[label].checkpoint.file_name).is_file()


needs_polar1_s = pytest.mark.skipif(not has_checkpoint("polar1-s"), reason="POLAR-1-S not in FASTMACE_MODELS")


def _structure(repeat=2, seed=17):
    """Rattled, sheared NaCl (as polar_stress_probe.py)."""
    atoms = nacl(repeat=repeat, rattle=0.05, seed=seed)
    e = np.random.default_rng(20261016).uniform(-0.02, 0.02, (3, 3))
    atoms.set_cell(atoms.cell.array @ (np.identity(3) + 0.5 * (e + e.T)), scale_atoms=True)
    return atoms


# --------------------------------------------------------------------------- k-vectors


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fixed_k_vectors_on_cuda(dtype):
    import types

    from graph_longrange.kspace import compute_k_vectors_flat
    for cell in (np.identity(3) * 16.76, np.array([[11.6, 0, 0], [0.4, 10.9, 0], [-1.1, 0.7, 12.8]])):
        cutoff = torch.tensor(1.85, dtype=dtype, device=DEVICE)
        state = polar.PolarState(types.SimpleNamespace(kspace_cutoff=cutoff), cell, headroom=0.0)
        cell64 = torch.tensor(cell)
        cell_t = cell64.to(dtype).to(DEVICE).view(1, 3, 3)
        rcell_t = (2 * torch.pi * torch.linalg.inv(cell64.mT)).to(dtype).to(DEVICE).view(1, 3, 3)
        k_ref, n2_ref, _, _ = compute_k_vectors_flat(cutoff, cell_t, rcell_t)
        k, n2, batch, k0 = polar.compute_k_vectors_fixed(polar._Call(state), cutoff, cell_t, rcell_t)
        included = batch == 0
        assert int(included.sum()) == len(k_ref) and float(k0[0]) == 1.0
        # float64 bit for bit; float32 to rounding (the GPU's einsum sums in another order, 17 October 2026)
        exact = dict(rtol=0, atol=0) if dtype == torch.float64 else dict(rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(k[included], k_ref, **exact)
        torch.testing.assert_close(n2[included], n2_ref, **exact)
        assert not bool(state.overflow)


# --------------------------------------------------------------------------- tiny model: no sync, CUDA graph


def _tiny():
    calc = tiny_family_calculator("polar", [11, 17], device=DEVICE)
    return calc, calc.models[0]


def _data(calc, atoms):
    return calc._atoms_to_batch(atoms).to_dict()


def test_tiny_polar_runs_without_host_synchronisation():
    calc, model = _tiny()
    atoms = _structure(repeat=1)
    enabled = copy.deepcopy(model)
    polar.enable(enabled, atoms.cell.array, variable_cell=True)
    data = _data(calc, atoms)
    enabled(dict(data), compute_stress=True, training=False)  # warm-up outside the check
    torch.cuda.synchronize()
    with polar.no_sync_check():
        out = enabled(dict(data), compute_stress=True, training=False)
    reference = model(_data(calc, atoms), compute_stress=False, training=False)
    torch.testing.assert_close(out["forces"], reference["forces"], rtol=1e-10, atol=1e-10)
    assert not polar.overflowed(enabled)


def test_tiny_polar_force_and_stress_call_is_captured_in_a_cuda_graph():
    calc, model = _tiny()
    atoms = _structure(repeat=1)
    enabled = copy.deepcopy(model)
    polar.enable(enabled, atoms.cell.array, variable_cell=True)
    data = _data(calc, atoms)
    positions = data["positions"].detach().clone()
    static = dict(data, positions=positions)

    def call():
        out = enabled(dict(static), compute_stress=True, training=False)
        return out["energy"].detach(), out["forces"].detach(), out["stress"].detach()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    generator = torch.Generator(device=DEVICE).manual_seed(1)
    for _ in range(3):
        moved = (data["positions"] + 1e-3 * torch.randn(positions.shape, generator=generator, device=DEVICE,
                                                       dtype=positions.dtype)).detach()
        with torch.no_grad():
            positions.copy_(moved)
        graph.replay()
        torch.cuda.synchronize()
        eager = enabled(dict(data, positions=moved.clone()), compute_stress=True, training=False)
        for got, key in zip(captured, ("energy", "forces", "stress")):
            torch.testing.assert_close(got, eager[key].detach(), rtol=1e-10, atol=1e-10)
    assert not polar.overflowed(enabled)


# --------------------------------------------------------------------------- POLAR-1-S, plain cuEq float64


def _ase(calc, atoms, stress=True):
    atoms = atoms.copy()
    calc.reset()  # ASE would return cached results for the same positions and cell
    atoms.calc = calc
    energy, forces = atoms.get_potential_energy(), atoms.get_forces()
    return energy, forces, (atoms.get_stress(voigt=False) if stress else None)


def _fd_stress(calc, atoms, h=1e-4):
    volume = abs(np.linalg.det(atoms.cell.array))
    fd = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            eps = np.zeros((3, 3))
            if i == j:
                eps[i, i] = h
            else:
                eps[i, j] = eps[j, i] = h / 2
            energies = []
            for sign in (1, -1):
                deformed = atoms.copy()
                deformation = np.identity(3) + sign * eps
                deformed.set_cell(atoms.cell.array @ deformation, scale_atoms=False)
                deformed.positions = atoms.positions @ deformation
                energies.append(_ase(calc, deformed, stress=False)[0])
            fd[i, j] = fd[j, i] = (energies[0] - energies[1]) / (2 * h * volume)
    return fd


@pytest.fixture(scope="module")
def polar1_s_float64():
    from fastmace.model import loading
    return loading.load("polar1-s", backend="cueq", dtype="float64", device=DEVICE)


@pytest.mark.slow
@needs_polar1_s
def test_polar1_s_float64_forces_and_stress(polar1_s_float64):
    calc = polar1_s_float64.calculator
    model = calc.models[0]
    atoms = _structure(repeat=2)
    released = _ase(calc, atoms)
    polar.enable(model, atoms.cell.array, variable_cell=True)
    try:
        enabled = _ase(calc, atoms)
        fd = _fd_stress(calc, atoms)
        overflow = polar.overflowed(model)
    finally:
        polar.disable(model)
    np.testing.assert_allclose(enabled[0], released[0], rtol=1e-12)
    np.testing.assert_allclose(enabled[1], released[1], rtol=0, atol=1e-9)
    gpu_error, released_error = np.abs(enabled[2] - fd).max(), np.abs(released[2] - fd).max()
    # Research engine (polar_stress_probe.py, NaCl 3x3x3): 7.5e-9 against 2.0e-5 eV/A^3 for MACE 0.3.16.
    assert gpu_error < 1e-7, gpu_error
    assert released_error > 10 * gpu_error, (released_error, gpu_error)
    assert not overflow


@pytest.mark.slow
@needs_polar1_s
def test_polar1_s_uvu_self_check(polar1_s_float64):
    model = polar1_s_float64.model
    rewritten = copy.deepcopy(model)
    assert uvu.applies_to(model) == (True, "ok")
    names = uvu.apply(rewritten)
    result = uvu.selfcheck(model, rewritten)
    assert names and result["passed"] and result["modules"] == len(names), result


# --------------------------------------------------------------------------- POLAR-1-S hybrid float32


@pytest.mark.slow
@needs_polar1_s
@pytest.mark.skipif(not _hybrid_has("openequivariance"), reason="OpenEquivariance is not installed")
def test_polar1_s_hybrid_float32_in_a_fresh_process(polar1_s_float64):
    calc = polar1_s_float64.calculator
    atoms = _structure(repeat=2)
    polar.enable(calc.models[0], atoms.cell.array, variable_cell=True)
    try:
        _, f64, s64 = _ase(calc, atoms)
    finally:
        polar.disable(calc.models[0])
    code = """
        import json
        import numpy as np
        import fastmace
        from fastmace.longrange import polar
        from fastmace.model import loading
        from fastmace.rewrites.polar import uvu
        loaded = loading.load("polar1-s", backend="hybrid", dtype="float32", device="cuda:0")
        from _support.structures import nacl

        def _ase(calc, atoms):
            atoms = atoms.copy()
            calc.reset()  # ASE would return cached results for the same positions and cell
            atoms.calc = calc
            return atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress(voigt=False)

        calc = loaded.calculator
        atoms = nacl(repeat=2, rattle=0.05, seed=17)
        e = np.random.default_rng(20261016).uniform(-0.02, 0.02, (3, 3))
        atoms.set_cell(atoms.cell.array @ (np.identity(3) + 0.5 * (e + e.T)), scale_atoms=True)
        released = _ase(calc, atoms)
        # In place: a deep copy of a hybrid model loses MACE's OEq adapters (the force field's
        # copy restores them, 1a-T11), which is not this test's subject.
        rewritten = calc.models[0]
        uvu.apply(rewritten)
        polar.enable(rewritten, atoms.cell.array, variable_cell=True)
        enabled = _ase(calc, atoms)
        print(json.dumps({"released_f": released[1].tolist(), "released_s": released[2].tolist(),
                          "f": enabled[1].tolist(), "s": enabled[2].tolist(), "e": enabled[0],
                          "released_e": released[0], "overflow": polar.overflowed(rewritten),
                          "patches": [r.name for r in loaded.patches]}))
    """
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "tests")]),
               PYTHONDONTWRITEBYTECODE="1")
    if HYBRID_PYTHON != sys.executable:
        env.pop("LD_LIBRARY_PATH", None)  # the in-process OEq runtime needs none
    result = subprocess.run([HYBRID_PYTHON, "-c", textwrap.dedent(code)], cwd=REPOSITORY_ROOT, env=env,
                            capture_output=True, text=True, timeout=1800, check=False)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-6000:]
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["patches"] == ["oeq_runtime"] and not out["overflow"]
    f, released_f = np.array(out["f"]), np.array(out["released_f"])
    # Forces: the GPU-only path and uvu change only float32 rounding; the long-range part does not
    # enter the forces differently. Stress: the hybrid float32 stress follows the corrected float64 one.
    assert np.abs(f - released_f).max() < 1e-3
    assert np.abs(f - f64).max() < 1e-3
    assert np.abs(np.array(out["s"]) - s64).max() < 1e-5
