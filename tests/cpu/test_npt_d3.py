"""NPT with D3: the dispersion virial in BAOAB's barostat (plan task 3b-T05; SPEC §11.2, §8.5).

A tiny MH-1-class model on rattled NaCl, the minimal planner on the CPU, D3 on its torch tier (as
``test_d3_integration.py``); the reference is MACE alone plus ``torch-dftd`` on the same record.

- the cell force ``force_eps`` is BAOAB's formula with the total stress, MACE's plus D3's, at the start
  and after steps, in both cell modes; with MACE's stress alone it would differ;
- an accepted MC species swap refreshes ``force_eps`` with the new total stress (§8.5 step 6);
- a hydrostatic NPT trajectory with D3 follows ASE's ``LangevinBAOAB`` driving MACE's own calculator plus
  ``torch-dftd`` (``SumCalculator``) with the same seed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from test_d3_integration import NA_CL, RECORD, _planner

from _support.structures import nacl
from fastmace import integrators
from fastmace.planner.workload import Workload

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    pytest.importorskip("cuequivariance_torch")
    from _support.mc_fixtures import tiny_loaded

    return tiny_loaded(tmp_path_factory.mktemp("models"), NA_CL)


def npt(cell_mode="hydrostatic", pressure_GPa=0.0):
    return integrators.LangevinBAOAB(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01,
                                     pressure_GPa=pressure_GPa, cell_mode=cell_mode).to_config(seed=1)


def engine_for(loaded, atoms, cell_mode="hydrostatic", pressure_GPa=0.0):
    from fastmace.model.forcefield import ForceField

    planner = _planner()
    field = ForceField(loaded, atoms, precision="float64", species=list(NA_CL), dispersion=RECORD)
    plan = planner.plan(field, atoms, npt(cell_mode, pressure_GPa), Workload(max_temperature_K=300.0))
    return planner.build(plan, field, atoms)


def state_atoms(engine, like):
    atoms = like.copy()
    state = engine.state
    atoms.numbers = state.species.detach().numpy()
    atoms.set_cell(state.cell.detach().numpy().copy())
    atoms.set_positions(state.positions.detach().numpy().copy())
    return atoms


def stresses(loaded, atoms, monkeypatch):
    """MACE's stress alone and MACE's plus torch-dftd's on RECORD (eV/Å³, ASE sign)."""
    pytest.importorskip("torch_dftd")
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.reference.d3 import torch_dftd3

    planner = _planner()
    monkeypatch.setattr(minimal, "default_planner", lambda force_field: planner)
    _, _, mace = ForceField(loaded, atoms, precision="float64", species=list(NA_CL)).evaluate(atoms, stress=True)
    return mace, mace + torch_dftd3(atoms, RECORD)[2]


def expected_force_eps(engine, stress, before_kick=True):
    """LangevinBAOAB._update_force_eps (ASE's ``_update_force_eps``) with ``stress``, from the state.

    Stage B computes it before its velocity half kick (``before_kick``: the kinetic part takes
    v - dt/2 F/m); the registry's refresh between steps (§8.5 step 6) takes the state's velocities."""
    state, dyn = engine.state, engine.integrator
    cell = state.cell.detach().numpy()
    volume = abs(np.linalg.det(cell))
    m = state.masses.detach().numpy().reshape(-1, 1)
    v = state.velocities.detach().numpy()
    if before_kick:
        v = v - 0.5 * dyn.dt * state.forces.detach().numpy() / m
    external = dyn.externalstress_step.detach().numpy()
    if dyn.hydrostatic:
        ke2 = float((m * v * v).sum())
        x = (ke2 - volume * np.trace(stress)) / (3.0 * volume)
        return 3.0 * volume * (x + external) + 3.0 / dyn.Ndof * ke2
    kinetic = (m * v).T @ v
    x = (kinetic - volume * stress) / volume
    return volume * (x + external) + np.trace(kinetic) / dyn.Ndof * np.identity(3)


@pytest.mark.parametrize("cell_mode", ["hydrostatic", "full"])
def test_force_eps_has_the_d3_virial(loaded, monkeypatch, cell_mode):
    atoms = nacl(1, rattle=0.05, seed=3)
    engine = engine_for(loaded, atoms, cell_mode, pressure_GPa=0.3)
    for steps in (0, 15):
        engine.integrator.set_temperature(300.0)
        engine.run(steps)
        engine.sync()
        mace, total = stresses(loaded, state_atoms(engine, atoms), monkeypatch)
        stress = engine.state.stress.detach().numpy()
        # the state's stress is MACE's plus D3's (to the reference's agreement, as test_d3_integration.py) ...
        np.testing.assert_allclose(stress, total, rtol=0, atol=1e-8)
        # ... and the cell force is BAOAB's formula with it
        force_eps = engine.integrator.force_eps.detach().numpy()
        expected = expected_force_eps(engine, stress, before_kick=steps > 0)  # step 0: start(), no kick yet
        np.testing.assert_allclose(force_eps, expected, rtol=1e-12, atol=1e-12)
        # D3's part is not negligible here: MACE's stress alone gives another cell force
        assert np.abs(expected_force_eps(engine, mace) - force_eps).max() > 1e-4


@pytest.mark.parametrize("cell_mode", ["hydrostatic", "full"])
def test_an_accepted_species_swap_refreshes_force_eps_with_d3(loaded, monkeypatch, cell_mode):
    from fastmace.events import Event, Transaction

    atoms = nacl(1, rattle=0.05, seed=7)
    engine = engine_for(loaded, atoms, cell_mode)
    engine.integrator.set_temperature(300.0)
    engine.run(5)
    engine.sync()
    numbers = engine.state.species.detach().numpy().copy()
    numbers[[0, 1]] = numbers[[1, 0]]
    assert numbers[0] != numbers[1]
    assert Transaction(engine, [Event("species", values=numbers)]).run() == "accepted"
    _, total = stresses(loaded, state_atoms(engine, atoms), monkeypatch)
    stress = engine.state.stress.detach().numpy()
    np.testing.assert_allclose(stress, total, rtol=0, atol=1e-8)
    np.testing.assert_allclose(engine.integrator.force_eps.detach().numpy(),
                               expected_force_eps(engine, stress, before_kick=False), rtol=1e-12, atol=1e-12)


def test_npt_trajectory_with_d3_follows_ase(loaded):
    pytest.importorskip("torch_dftd")
    from ase import units
    from ase.calculators.mixing import SumCalculator
    from ase.md.langevinbaoab import LangevinBAOAB as ASELangevinBAOAB

    from fastmace.reference.d3 import torch_dftd3_calculator

    atoms = nacl(1, rattle=0.05, seed=5)
    engine = engine_for(loaded, atoms, pressure_GPa=0.5)
    dyn = engine.integrator
    reference = atoms.copy()
    reference.calc = SumCalculator([loaded.calculator, torch_dftd3_calculator(RECORD)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ase_dyn = ASELangevinBAOAB(reference, timestep=1.0 * units.fs, temperature_K=300.0,
                                   externalstress=-0.5 * units.GPa, hydrostatic=True, T_tau=dyn.T_tau,
                                   P_tau=dyn.P_tau, rng=np.random.default_rng(1))
    assert dyn.barostat_mass == pytest.approx(ase_dyn.barostat_mass, rel=1e-12)
    dyn.set_temperature(300.0)
    engine.run(40)
    ase_dyn.run(40)
    values = engine.sync()
    assert abs(reference.get_volume() / atoms.get_volume() - 1) > 1e-4  # the barostat moved the cell
    np.testing.assert_allclose(values["cell"], reference.cell.array, rtol=0, atol=1e-8)
    np.testing.assert_allclose(values["positions"], reference.positions, rtol=0, atol=1e-8)
    np.testing.assert_allclose(values["momenta"], reference.get_momenta(), rtol=0, atol=1e-7)
