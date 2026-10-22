"""The Langevin ``fixcm`` projection (plan task 3-T22; docs/design/fixcm-projection.md).

``IntegratorConfig.fixcm_projection``:

- ``"ase"`` (the default): ASE's projection, unchanged bit for bit (its noise-stream equivalence). It
  subtracts the plain mean from the random displacements and P/(N m_i) from the random velocities, then
  scales both by sqrt(N / (N - 1)). Right only for equal masses.
- ``"mass_weighted"``: the conditional Gaussian at fixed centre of mass. It subtracts the mass-weighted
  mean from both, with no scale factor.

The exact property checked here: with ``"mass_weighted"``, a run from a state at rest equals the run
without ``fixcm`` minus its centre-of-mass motion, step by step, with the same random numbers and a
translation-invariant force field. So it samples the fixcm-off canonical distribution of the internal
motion: 3N - 3 degrees of freedom at the target, per species too. The bias of ``"ase"`` is measured in
``docs/validation/3-T22-fixcm.md``.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.units import fs

from fastmace.config import IntegratorConfig
from fastmace.integrators.langevin import Langevin
from fastmace.state import SimulationState

FRICTION = 0.05 / fs


def harmonic_forces(positions, bonds, lengths, k=1.0):
    """Springs on fixed pairs: translation invariant, so the forces sum to zero."""
    i, j, shifts = bonds
    d = positions[j] + shifts - positions[i]
    r = torch.linalg.norm(d, dim=1, keepdim=True)
    f = k * (r - lengths) * d / r
    out = torch.zeros_like(positions)
    out.index_add_(0, i, f)
    out.index_add_(0, j, -f)
    return out


def bond_network(atoms, cutoff=3.2):
    from ase.neighborlist import neighbor_list

    i, j, shift = neighbor_list("ijS", atoms, cutoff)
    keep = i < j
    i, j, shift = i[keep], j[keep], shift[keep]
    cell = atoms.cell.array
    shifts = torch.tensor(shift @ cell)
    positions = torch.tensor(atoms.get_positions())
    ii, jj = torch.tensor(i), torch.tensor(j)
    lengths = torch.linalg.norm(positions[jj] + shifts - positions[ii], dim=1, keepdim=True)
    return (ii, jj, shifts), lengths


def licscl(repeat=(2, 1, 1)):
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat(repeat)
    numbers = atoms.numbers.copy()
    cations = np.flatnonzero(numbers == 11)
    numbers[cations[0::2]], numbers[cations[1::2]] = 3, 55
    atoms.set_atomic_numbers(numbers)
    atoms.rattle(0.02, seed=1)
    return atoms


def integrator(atoms, *, temperature_K=600.0, seed=5, **config):
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=np.random.default_rng(seed))
    Stationary(atoms)
    state = SimulationState.from_atoms(atoms, "cpu", seed)
    config = IntegratorConfig(timestep_fs=1.0, friction=FRICTION, noise_batch=16, **config)
    return state, Langevin(state, config, temperature_K=temperature_K)


def run(state, dyn, network, steps):
    bonds, lengths = network
    state.forces.copy_(harmonic_forces(state.positions, bonds, lengths))
    for _ in range(steps):
        dyn.prepare_step(state)
        dyn.first_half(state)
        state.forces.copy_(harmonic_forces(state.positions, bonds, lengths))
        dyn.second_half(state)


def test_the_option_and_its_default():
    assert IntegratorConfig().fixcm_projection == "ase"
    assert IntegratorConfig(fixcm_projection="mass_weighted").fixcm_projection == "mass_weighted"
    with pytest.raises(ValueError, match="fixcm_projection"):
        IntegratorConfig(fixcm_projection="plain")


def test_ase_projection_is_unchanged_bitwise():
    """The explicit "ase" choice is the default code path: same numbers, same bits."""
    atoms = licscl()
    network = bond_network(atoms)
    results = []
    for options in ({}, {"fixcm_projection": "ase"}):
        state, dyn = integrator(atoms.copy(), **options)
        run(state, dyn, network, 25)
        results.append((state.positions.clone(), state.velocities.clone()))
    for a, b in zip(*results, strict=True):
        assert torch.equal(a, b)


def test_mass_weighted_noise_has_no_centre_of_mass_part():
    atoms = licscl()
    state, dyn = integrator(atoms, fixcm_projection="mass_weighted")
    run(state, dyn, bond_network(atoms), 1)
    masses = state.masses
    for name in ("rnd_pos", "rnd_vel"):
        total = (getattr(dyn, name) * masses).sum(dim=0)
        assert torch.allclose(total, torch.zeros(3, dtype=torch.float64), atol=1e-14), name


@pytest.mark.parametrize("structure", ["licscl", "nacl"])
def test_mass_weighted_equals_fixcm_off_minus_the_centre_of_mass(structure):
    """Same random numbers, translation-invariant forces, start at rest: the corrected run is the run
    without fixcm seen from its centre of mass, so it samples the same internal distribution."""
    atoms = licscl() if structure == "licscl" else bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 1, 1))
    network = bond_network(atoms)
    free_state, free = integrator(atoms.copy(), fixcm=False)
    fixed_state, fixed = integrator(atoms.copy(), fixcm_projection="mass_weighted")
    masses = free_state.masses
    total = masses.sum()
    start = (free_state.positions * masses).sum(dim=0) / total
    for _ in range(4):
        run(free_state, free, network, 10)
        run(fixed_state, fixed, network, 10)
        drift = (free_state.velocities * masses).sum(dim=0) / total
        shift = (free_state.positions * masses).sum(dim=0) / total - start
        torch.testing.assert_close(fixed_state.velocities, free_state.velocities - drift, rtol=0, atol=1e-11)
        torch.testing.assert_close(fixed_state.positions, free_state.positions - shift, rtol=0, atol=1e-11)
        momentum = (fixed_state.velocities * masses).sum(dim=0)
        torch.testing.assert_close(momentum, torch.zeros(3, dtype=torch.float64), rtol=0, atol=1e-11)


def test_equal_masses_differ_from_ase_by_its_scale_factor_only():
    """With equal masses both projections remove the same mean; ASE then scales by sqrt(N / (N - 1))."""
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat((2, 1, 1))
    network = bond_network(atoms, cutoff=4.0)
    noise = {}
    for projection in ("ase", "mass_weighted"):
        state, dyn = integrator(atoms.copy(), fixcm_projection=projection)
        run(state, dyn, network, 1)
        noise[projection] = (dyn.rnd_pos.clone(), dyn.rnd_vel.clone())
    n = len(atoms)
    for a, b in zip(noise["ase"], noise["mass_weighted"], strict=True):
        torch.testing.assert_close(a, b * (n / (n - 1.0)) ** 0.5, rtol=1e-13, atol=1e-18)


def test_the_fused_stage_keeps_the_ase_projection_only():
    from fastmace.kernels.triton import fused_integrator

    atoms = licscl()
    state, dyn = integrator(atoms, fixcm_projection="mass_weighted")
    engine = SimpleNamespace(cuda=True, variable_cell=False, integrator=dyn, state=state,
                             force_field=SimpleNamespace(neighbours=None), _mids=(state.velocities,))
    reason = fused_integrator.applicable(engine)
    assert reason is not None and "fixcm_projection" in reason
    without = dataclasses.replace(dyn.config, fixcm=False, fixcm_projection="mass_weighted")
    dyn.config = without
    dyn.fix_com = False
    assert "fixcm_projection" not in (fused_integrator.applicable(engine) or "")
