"""Integrator qualification (SPEC §12.1 "Integrator corrections", §12.2 "ASE equivalence"; plan 1b-T09).

The full-cell BAOAB has two variants. ``ase_trace_drag_all_elements=True`` must reproduce ASE's
``LangevinBAOAB`` (it tests the port); the production variant, whose drag on the atom velocities is
(p_eps + Tr(p_eps)/N_dof · I)/W (Quigley and Probert 2004, Eq. 5b), must reproduce an independent
reference with the same correction: ``fastmace.reference.integrators``, ASE with only the drag line
replaced (same noise stream, so trajectories compare exactly) and a float64 single step written from
the equation. Forces come from a tiny random MACE model through its ASE calculator on both sides, so
only the integrators differ. The tests read only the simulation state, never integrator internals.
"""
import functools
import warnings

import numpy as np
import pytest
import torch
from ase import units
from ase.md.langevinbaoab import LangevinBAOAB as ASELangevinBAOAB
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from _support.structures import nacl
from _support.tiny_models import tiny_mace_calculator
from fastmace.config import IntegratorConfig
from fastmace.errors import FastMACEError
from fastmace.integrators.baoab import LangevinBAOAB
from fastmace.reference import integrators as reference
from fastmace.state import SimulationState

SEED = 11
STEPS = 300
TIMESTEP_FS = 1.0
T_TAU_FS = 100.0
P_TAU_FS = 400.0
PEAK_K = 300.0
STRESS = -0.5 * units.GPa  # ASE's sign: a pressure of 0.5 GPa
TOLERANCE = 1e-9


@functools.lru_cache(maxsize=None)
def calculator():
    """One tiny random MACE calculator (float64) shared by both sides of every comparison."""
    return tiny_mace_calculator([11, 17])


def crystal():
    return nacl(2, rattle=0.05, seed=3)


def external(hydrostatic):
    """Hydrostatic load, or an anisotropic one that changes the cell shape (Voigt)."""
    return STRESS if hydrostatic else [STRESS, 1.5 * STRESS, 0.7 * STRESS, 0.0, 0.0, 0.0]


def config(hydrostatic, trace_all, noise_batch=7):
    return IntegratorConfig(ensemble="npt", timestep_fs=TIMESTEP_FS, friction=1.0 / (T_TAU_FS * units.fs),
                            barostat_tau_fs=P_TAU_FS, cell_mode="hydrostatic" if hydrostatic else "full",
                            noise_batch=noise_batch, ase_trace_drag_all_elements=trace_all)


class CalculatorForces:
    """The engine's force stage through an ASE calculator on a host copy of the state."""

    def __init__(self, atoms):
        self.atoms = atoms.copy()
        self.atoms.calc = calculator()

    def __call__(self, state):
        self.atoms.set_cell(state.cell.numpy().copy())
        self.atoms.set_positions(state.positions.numpy().copy())
        state.forces.copy_(torch.from_numpy(self.atoms.get_forces()))
        state.energy.copy_(torch.tensor([self.atoms.get_potential_energy()], dtype=torch.float64))
        state.stress.copy_(torch.from_numpy(self.atoms.get_stress(voigt=False)))


def target(step, steps):
    return PEAK_K * step / steps


def run_ase(cls, hydrostatic, steps=STEPS, atoms=None, temperature_K=0.0):
    """ASE's integrator (or the corrected reference) from the same state and seed, heating 0 -> PEAK_K."""
    atoms = crystal() if atoms is None else atoms.copy()
    atoms.calc = calculator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ASE's heuristic-mass notice
        dyn = cls(atoms, timestep=TIMESTEP_FS * units.fs, T_tau=T_TAU_FS * units.fs, P_tau=P_TAU_FS * units.fs,
                  hydrostatic=hydrostatic, externalstress=external(hydrostatic), temperature_K=temperature_K,
                  rng=np.random.default_rng(SEED))
    for step in range(1, steps + 1):
        if temperature_K == 0.0:
            dyn.set_temperature(temperature_K=target(step, steps))
        dyn.run(1)
    return atoms


def run_fastmace(hydrostatic, trace_all, steps=STEPS, atoms=None, temperature_K=0.0, noise_batch=7):
    """FastMACE's LangevinBAOAB, eager steps in the engine's order (docs/design/interfaces.md)."""
    atoms = crystal() if atoms is None else atoms
    state = SimulationState.from_atoms(atoms, "cpu", SEED, compute_stress=True)
    field = CalculatorForces(atoms)
    dyn = LangevinBAOAB(state, config(hydrostatic, trace_all, noise_batch), externalstress=external(hydrostatic),
                        temperature_K=temperature_K)
    field(state)
    dyn.start(state)
    for step in range(1, steps + 1):
        if temperature_K == 0.0:
            dyn.set_temperature(target(step, steps))
        dyn.prepare_step(state)
        dyn.first_half(state)
        field(state)
        dyn.second_half(state)
        state.step += 1
    return state


def deviation(state, atoms):
    """Largest absolute difference of cell (Å), positions (Å) and velocities (Å/ASE time)."""
    return max(np.abs(state.cell.numpy() - atoms.cell.array).max(),
               np.abs(state.positions.numpy() - atoms.positions).max(),
               np.abs(state.velocities.numpy() - atoms.get_velocities()).max())


@functools.lru_cache(maxsize=None)
def trajectories(hydrostatic):
    """The 300-step runs one cell mode needs (cached: each costs about 10 s on one CPU thread)."""
    runs = dict(ase=run_ase(ASELangevinBAOAB, hydrostatic), fastmace_ase=run_fastmace(hydrostatic, True))
    if not hydrostatic:
        runs.update(corrected=run_ase(reference.CorrectedLangevinBAOAB, hydrostatic),
                    fastmace=run_fastmace(hydrostatic, False))
    return runs


def assert_the_run_moved(atoms):
    assert atoms.get_temperature() > 50                                  # the thermostat heated it
    assert abs(atoms.get_volume() / crystal().get_volume() - 1) > 2e-3   # the barostat moved the cell


# ---------------------------------------------------------------------------- trajectories

@pytest.mark.parametrize("hydrostatic", [True, False], ids=["hydrostatic", "full"])
def test_ase_variant_reproduces_ase(hydrostatic):
    runs = trajectories(hydrostatic)
    assert deviation(runs["fastmace_ase"], runs["ase"]) < TOLERANCE
    assert float(runs["fastmace_ase"].energy[0]) == pytest.approx(runs["ase"].get_potential_energy(), abs=1e-8)
    assert_the_run_moved(runs["ase"])
    if not hydrostatic:  # the anisotropic load changed the cell shape
        lengths = runs["ase"].cell.lengths()
        assert abs(lengths[1] - lengths[0]) > 1e-3


def test_production_variant_reproduces_the_corrected_reference():
    runs = trajectories(False)
    assert deviation(runs["fastmace"], runs["corrected"]) < TOLERANCE
    assert float(runs["fastmace"].energy[0]) == pytest.approx(runs["corrected"].get_potential_energy(), abs=1e-8)
    assert_the_run_moved(runs["corrected"])


def test_the_variants_differ_by_the_correction():
    """Negative control: the trace term on all nine elements against on the diagonal, O(1/N_dof) of the drag."""
    runs = trajectories(False)
    variants = deviation(runs["fastmace"], runs["ase"])
    references = deviation(runs["fastmace_ase"], runs["corrected"])
    assert 1e3 * TOLERANCE < variants < 1e-2
    assert 1e3 * TOLERANCE < references < 1e-2
    # the correction is small against the motion itself
    motion = np.abs(runs["ase"].positions - crystal().positions).max()
    assert variants < 0.1 * motion


# ---------------------------------------------------------------------------- the reference itself

def test_corrected_reference_changes_only_the_drag():
    """Hydrostatic cells have no off-diagonal drag: the corrected reference equals ASE there."""
    corrected = run_ase(reference.CorrectedLangevinBAOAB, True, steps=20)
    stock = run_ase(ASELangevinBAOAB, True, steps=20)
    np.testing.assert_array_equal(corrected.positions, stock.positions)
    np.testing.assert_array_equal(corrected.cell.array, stock.cell.array)


def test_corrected_drag_is_the_published_formula():
    rng = np.random.default_rng(0)
    p_eps = rng.normal(size=(3, 3))
    p_eps = p_eps + p_eps.T
    n_dof, mass = 192, 7.5
    expected = (p_eps + np.trace(p_eps) / n_dof * np.identity(3)) / mass
    np.testing.assert_allclose(reference.corrected_drag(p_eps, n_dof, mass), expected, rtol=0, atol=1e-15)
    ase_drag = reference.corrected_drag(p_eps, n_dof, mass, trace_all_elements=True)
    np.testing.assert_allclose(ase_drag - expected, np.trace(p_eps) / n_dof / mass * (1 - np.identity(3)),
                               rtol=0, atol=1e-15)


def test_ase_source_guard_accepts_the_installed_ase():
    reference.check_ase_source()  # raises when ASE's BAOAB changed


def test_ase_source_guard_refuses_a_changed_ase(monkeypatch):
    monkeypatch.setitem(reference.ASE_SOURCE_SHA256, "_BAOAB_OU", "0" * 64)
    with pytest.raises(FastMACEError, match="_BAOAB_OU"):
        reference.check_ase_source()
    with pytest.raises(FastMACEError):
        run_ase(reference.CorrectedLangevinBAOAB, False, steps=1)


# ---------------------------------------------------------------------------- single steps from the paper

def warm_crystal(seed=5):
    atoms = crystal()
    MaxwellBoltzmannDistribution(atoms, temperature_K=PEAK_K, rng=np.random.default_rng(seed))
    return atoms


def paper_step(atoms, p_eps, trace_all_elements=False, steps=1):
    """``reference.baoab_step`` from ``atoms`` with ASE's settings and noise stream."""
    atoms = atoms.copy()
    atoms.calc = calculator()
    rng = np.random.default_rng(SEED)
    settings = reference.BAOABSettings.from_ase(atoms, timestep=TIMESTEP_FS * units.fs, T_tau=T_TAU_FS * units.fs,
                                                P_tau=P_TAU_FS * units.fs, externalstress=external(False),
                                                temperature_K=PEAK_K)
    step = reference.BAOABStep(atoms.positions, atoms.get_velocities(), atoms.get_masses(), atoms.cell.array,
                               p_eps)
    for _ in range(steps):
        xi, zeta = rng.normal(size=(len(atoms), 3)), rng.normal(size=(3, 3))
        step = reference.baoab_step(step, settings, forces_and_stress(atoms), xi, zeta,
                                    trace_all_elements=trace_all_elements)
    return step


def forces_and_stress(atoms):
    def evaluate(positions, cell):
        atoms.set_cell(cell)
        atoms.set_positions(positions)
        return atoms.get_forces(), atoms.get_stress(voigt=False)
    return evaluate


def ase_step(cls, atoms, p_eps, steps=1):
    atoms = atoms.copy()
    atoms.calc = calculator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dyn = cls(atoms, timestep=TIMESTEP_FS * units.fs, T_tau=T_TAU_FS * units.fs, P_tau=P_TAU_FS * units.fs,
                  hydrostatic=False, externalstress=external(False), temperature_K=PEAK_K,
                  rng=np.random.default_rng(SEED))
    dyn.p_eps = np.array(p_eps, dtype=float)
    dyn.run(steps)
    return atoms, dyn.p_eps


def assert_same_step(step, atoms, p_eps=None, atol=1e-12):
    np.testing.assert_allclose(step.positions, atoms.positions, rtol=0, atol=atol)
    np.testing.assert_allclose(step.velocities, atoms.get_velocities(), rtol=0, atol=atol)
    np.testing.assert_allclose(step.cell, atoms.cell.array, rtol=0, atol=atol)
    if p_eps is not None:
        np.testing.assert_allclose(step.p_eps, p_eps, rtol=0, atol=atol * 1e3)


def moving_barostat():
    """A barostat momentum with a large trace and off-diagonal elements, so the drag variants differ."""
    rng = np.random.default_rng(1)
    p_eps = rng.normal(size=(3, 3))
    return 50.0 * (p_eps + p_eps.T) + 200.0 * np.identity(3)


@pytest.mark.parametrize("steps", [1, 3])
def test_paper_step_with_the_ase_drag_equals_ase(steps):
    """The single step written from the equations reproduces ASE when given ASE's drag (checks the step)."""
    atoms, p_eps = warm_crystal(), moving_barostat()
    expected, expected_p_eps = ase_step(ASELangevinBAOAB, atoms, p_eps, steps)
    assert_same_step(paper_step(atoms, p_eps, trace_all_elements=True, steps=steps), expected, expected_p_eps)


@pytest.mark.parametrize("steps", [1, 3])
def test_paper_step_equals_the_corrected_ase(steps):
    atoms, p_eps = warm_crystal(), moving_barostat()
    expected, expected_p_eps = ase_step(reference.CorrectedLangevinBAOAB, atoms, p_eps, steps)
    assert_same_step(paper_step(atoms, p_eps, steps=steps), expected, expected_p_eps)
    stock, _ = ase_step(ASELangevinBAOAB, atoms, p_eps, steps)
    assert np.abs(stock.get_velocities() - expected.get_velocities()).max() > 1e-9  # the variants differ


@pytest.mark.parametrize("noise_batch", [1, 7])
def test_production_step_equals_the_paper_step(noise_batch):
    """FastMACE's production BAOAB, three steps from rest of the barostat, against the step from the paper."""
    atoms = warm_crystal()
    state = run_fastmace(False, False, steps=3, atoms=atoms.copy(), temperature_K=PEAK_K, noise_batch=noise_batch)
    expected = paper_step(atoms, np.zeros((3, 3)), steps=3)
    assert_same_step(expected, state_as_atoms(state, atoms))
    ase_variant = run_fastmace(False, True, steps=3, atoms=atoms.copy(), temperature_K=PEAK_K)
    assert np.abs(ase_variant.velocities.numpy() - state.velocities.numpy()).max() > 1e-12


def state_as_atoms(state, like):
    atoms = like.copy()
    atoms.set_cell(state.cell.numpy())
    atoms.set_positions(state.positions.numpy())
    atoms.set_velocities(state.velocities.numpy())
    return atoms


# ---------------------------------------------------------------------------- the record for 1b-T13

def test_qualification_record_names_both_parts():
    record = reference.QUALIFICATION
    assert record["id"] == reference.QUALIFICATION_ID
    assert record["correction"]["drag"] == "(p_eps + Tr(p_eps)/N_dof * I) / W"
    assert record["correction"]["source"].startswith("Quigley and Probert 2004")
    for part in ("ase_variant", "production"):
        entry = record[part]
        assert entry["tolerance"] == TOLERANCE and entry["steps"] == STEPS
        assert entry["reference"] and entry["tests"]
    assert record["ase_source_sha256"] == reference.ASE_SOURCE_SHA256
