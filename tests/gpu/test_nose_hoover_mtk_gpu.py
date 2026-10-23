"""Nose-Hoover chains and MTK on CUDA (plan task 4-T03): captured half steps, and ASE equivalence on the device.

The chain loops are unrolled into the captured graphs; replays must equal eager steps bitwise, with
temperature and pressure targets filled between replays (no host sync inside a half step). The ASE
comparison runs the CUDA arithmetic (exp, expm1, the block Taylor series) against ASE 3.27's classes with
ASE's Lennard-Jones on the host, in float64. The equivalence with MACE (conductor's choice 2) needs the
planner and Simulation wiring of 4-T01 and runs after it.
"""

import numpy as np
import pytest
import torch
from ase import units
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.md.nose_hoover_chain import MTKNPT, IsotropicMTKNPT, NoseHooverChainNVT
from ase.md.velocitydistribution import Stationary

from fastmace import integrators as records
from fastmace.integrators.base import CONSERVED_KEY
from fastmace.integrators.registry import build_integrator
from fastmace.neighbors.skin import det_3x3
from fastmace.state import SimulationState

STEPS = 20


def atoms():
    structure = bulk("Ar", "fcc", a=5.26, cubic=True).repeat((3, 3, 3))
    structure.rattle(0.1, seed=6)
    structure.set_velocities(np.random.default_rng(1).standard_normal((len(structure), 3)) * 0.005)
    return structure


class HarmonicForces:
    """A tether and a volume spring, written into the state; graph-capturable."""

    def __init__(self, state, k=0.5, bulk_modulus=0.5):
        self.x0 = state.positions.clone()
        self.v0 = det_3x3(state.cell).clone()
        self.k, self.bulk_modulus = k, bulk_modulus
        self.eye = torch.eye(3, dtype=torch.float64, device=state.positions.device)

    def __call__(self, state):
        d = state.positions - self.x0
        state.forces.copy_(-self.k * d)
        state.energy.copy_((0.5 * self.k * (d * d).sum()).reshape(1))
        if state.stress is not None:
            volume = det_3x3(state.cell)
            state.stress.copy_(self.k * (d.T @ d) / volume + self.bulk_modulus * (volume / self.v0 - 1.0) * self.eye)


def config(name, fixcm):
    common = dict(timestep_fs=2.0, temperature_K=50.0, tdamp_fs=40.0, fixcm=fixcm, dof="auto" if fixcm else "ase")
    if name == "nhc":
        return records.NoseHooverChain(**common).to_config(seed=1)
    return records.MTK(pressure_GPa=0.5, pdamp_fs=100.0, cell_mode=name, **common).to_config(seed=1)


def build(name, fixcm):
    state = SimulationState.from_atoms(atoms(), "cuda", 20261001, compute_stress=name != "nhc")
    dyn = build_integrator(state, config(name, fixcm))
    field = HarmonicForces(state)
    field(state)
    dyn.start(state)
    return state, dyn, field


def written(state, dyn):
    tensors = dict(dyn.buffers(state))
    tensors.update(forces=state.forces, energy=state.energy)
    if state.stress is not None:
        tensors["stress"] = state.stress
    return tensors


def capture(state, dyn, field):
    """Graphs A (first_half) and B (force call, second_half), warmed up without changing the state."""
    tensors = written(state, dyn)
    saved = {key: value.clone() for key, value in tensors.items()}
    stage_a = lambda: dyn.first_half(state)  # noqa: E731

    def stage_b():
        field(state)
        dyn.second_half(state)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            stage_a()
            stage_b()
    torch.cuda.current_stream().wait_stream(side)
    graphs = {}
    for key, stage in (("A", stage_a), ("B", stage_b)):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stage()
        graphs[key] = graph
    for key, value in tensors.items():
        value.copy_(saved[key])
    torch.cuda.synchronize()
    return graphs


@pytest.mark.parametrize("fixcm", [True, False], ids=["fixcm", "ase"])
@pytest.mark.parametrize("name", ["nhc", "hydrostatic", "full"])
def test_captured_half_steps_replay_the_eager_steps(name, fixcm):
    eager_state, eager_dyn, eager_field = build(name, fixcm)
    state, dyn, field = build(name, fixcm)
    graphs = capture(state, dyn, field)
    pointers = {key: value.data_ptr() for key, value in written(state, dyn).items()}
    for step in range(1, STEPS + 1):
        for target in (eager_dyn, dyn):
            target.set_temperature(50.0 + 15.0 * step)
            if name != "nhc":
                target.set_pressure(0.5 - 0.05 * step)
        eager_dyn.prepare_step(eager_state)
        eager_dyn.first_half(eager_state)
        eager_field(eager_state)
        eager_dyn.second_half(eager_state)
        dyn.prepare_step(state)
        graphs["A"].replay()
        graphs["B"].replay()
    torch.cuda.synchronize()
    assert {key: value.data_ptr() for key, value in written(state, dyn).items()} == pointers
    reference = written(eager_state, eager_dyn)
    for key, value in written(state, dyn).items():
        assert torch.equal(value, reference[key]), f"{key}: max |difference| {(value - reference[key]).abs().max()}"
    assert not torch.equal(state.positions, eager_field.x0)
    assert float(dyn.thermostat.eta.abs().max()) > 0
    if dyn.variable_cell:
        assert not torch.equal(state.cell, torch.tensor(atoms().cell.array, device="cuda"))


class ASEForces:
    """ASE's Lennard-Jones on a host copy of the state, written to the device: energy, forces, stress."""

    def __init__(self, structure):
        self.atoms = structure.copy()
        self.atoms.calc = LennardJones(sigma=3.4, epsilon=0.0104, rc=7.0, smooth=False)

    def __call__(self, state):
        self.atoms.set_cell(state.cell.cpu().numpy().copy())
        self.atoms.set_positions(state.positions.cpu().numpy().copy())
        state.forces.copy_(torch.from_numpy(self.atoms.get_forces()))
        state.energy.copy_(torch.tensor([self.atoms.get_potential_energy()], dtype=torch.float64))
        if state.stress is not None:
            state.stress.copy_(torch.from_numpy(self.atoms.get_stress(voigt=False)))


def warm_argon():
    structure = atoms()
    structure.set_velocities(np.random.default_rng(3).standard_normal((len(structure), 3)) * 0.004)
    Stationary(structure)
    return structure


@pytest.mark.parametrize("name", ["nhc", "hydrostatic", "full"])
def test_cuda_trajectory_matches_ase(name):
    """The CUDA arithmetic (exp, expm1, the block Taylor series), float64, against ASE 3.27 for 300 steps."""
    reference, test = warm_argon(), warm_argon()
    reference.calc = LennardJones(sigma=3.4, epsilon=0.0104, rc=7.0, smooth=False)
    common = dict(timestep=2.0 * units.fs, temperature_K=50.0, tdamp=40.0 * units.fs)
    if name == "nhc":
        ase_dyn = NoseHooverChainNVT(reference, **common)
    else:
        cls = IsotropicMTKNPT if name == "hydrostatic" else MTKNPT
        ase_dyn = cls(reference, pressure_au=0.5 * units.GPa, pdamp=100.0 * units.fs, **common)
    state = SimulationState.from_atoms(test, "cuda", 7, compute_stress=name != "nhc")
    dyn = build_integrator(state, config(name, fixcm=False))
    field = ASEForces(test)
    field(state)
    dyn.start(state)
    for _ in range(300):
        ase_dyn.run(1)
        dyn.prepare_step(state)
        dyn.first_half(state)
        field(state)
        dyn.second_half(state)
    np.testing.assert_allclose(state.positions.cpu().numpy(), reference.positions, atol=1e-9)
    np.testing.assert_allclose(state.velocities.cpu().numpy(), reference.get_velocities(), atol=1e-9)
    np.testing.assert_allclose(state.cell.cpu().numpy(), reference.cell.array, atol=1e-9)
    np.testing.assert_allclose(dyn.thermostat.p_eta.cpu().numpy(), ase_dyn._thermostat._p_eta, rtol=1e-8, atol=1e-12)
    conserved = float(dyn.output_values(state)[CONSERVED_KEY][0])
    assert conserved == pytest.approx(ase_dyn.get_conserved_energy(), abs=1e-8)
    if name != "nhc":
        assert abs(reference.get_volume() / atoms().get_volume() - 1) > 1e-3
