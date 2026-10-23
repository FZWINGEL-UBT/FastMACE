"""Plan task 4-T02: NVE (ASE ``VelocityVerlet``), CSVR (ASE 3.27 ``Bussi``) and Berendsen (``NVTBerendsen``,
``NPTBerendsen``, ``Inhomogeneous_NPTBerendsen``) as graph-capturable half steps.

The integrators run eagerly in the engine's order (``md_step``) against their ASE classes on the same tiny
float64 MACE model: positions within 1e-12 Angstrom after 200 steps, ``dof="ase"``, and a zero-momentum
start, so the ``fixcm`` projection removes only float drift. NPT Berendsen makes one force call per step,
so it is compared to ASE's class reordered the same way (``Reordered``) and, loosely, to ASE itself.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest
import torch
from ase import units
from ase.build import bulk
from ase.md.bussi import Bussi
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet as ASEVelocityVerlet

from _support.tiny_models import FAMILIES, tiny_mace_calculator
from fastmace import integrators as records
from fastmace.engine.registry import policy_for
from fastmace.integrators import registry as scheme_registry
from fastmace.integrators.base import CONSERVED_KEY
from fastmace.integrators.berendsen import Berendsen
from fastmace.integrators.csvr import CSVR
from fastmace.integrators.nve import VelocityVerlet
from fastmace.io import restart as restart_io
from fastmace.state import SimulationState

SEED = 11
STEPS = 200
DT_FS = 1.0
TOLERANCE = 1e-12  # Angstrom (positions) and Angstrom/(ASE time) (velocities), after STEPS steps


# -------------------------------------------------------------------------- helpers

def nacl(seed=0):
    """NaCl 2 x 2 x 1 conventional cells (32 atoms, unequal masses), rattled, 300 K, zero momentum."""
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 1))
    atoms.rattle(0.03, seed=seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(seed))
    Stationary(atoms)
    return atoms


_CALCULATORS = {}


def calculator(slot):
    """One tiny MH-1-like MACE calculator per side, warmed up (TorchScript's first call rounds differently)."""
    if slot not in _CALCULATORS:
        calc = tiny_mace_calculator([11, 17], **FAMILIES["mh1"])
        warm = nacl(seed=99)
        warm.calc = calc
        warm.get_forces()
        warm.get_stress()
        _CALCULATORS[slot] = calc
    return _CALCULATORS[slot]


class Forces:
    """The engine's force stage on a host copy of the state: energy, forces and, if allocated, stress."""

    def __init__(self, atoms, slot="fastmace"):
        self.atoms = atoms.copy()
        self.atoms.calc = calculator(slot)
        self.calls = 0

    def __call__(self, state):
        self.calls += 1
        self.atoms.set_cell(state.cell.numpy().copy())
        self.atoms.set_positions(state.positions.numpy().copy())
        state.forces.copy_(torch.from_numpy(self.atoms.get_forces()))
        state.energy.copy_(torch.tensor([self.atoms.get_potential_energy()], dtype=torch.float64))
        if state.stress is not None:
            state.stress.copy_(torch.from_numpy(self.atoms.get_stress(voigt=False)))


def new_state(atoms, compute_stress=False, seed=SEED):
    return SimulationState.from_atoms(atoms, "cpu", seed, compute_stress=compute_stress)


def start(state, integrator, field):
    field(state)
    integrator.start(state)


def md_step(state, integrator, field):
    integrator.prepare_step(state)
    integrator.first_half(state)
    field(state)
    integrator.second_half(state)
    state.step += 1


def run_md(state, integrator, field, steps):
    for _ in range(steps):
        md_step(state, integrator, field)


def build(record, state, **config_options):
    """The integrator of a public record, through the registry (as the planner builds it)."""
    config = record.to_config(seed=SEED)
    if config_options:
        config = dataclasses.replace(config, **config_options)
    return scheme_registry.build_integrator(state, config, temperature_K=config.temperature_K)


def reference_atoms(atoms):
    reference = atoms.copy()
    reference.calc = calculator("ase")
    return reference


def assert_close(state, reference, tolerance=TOLERANCE):
    np.testing.assert_allclose(state.positions.numpy(), reference.positions, rtol=0, atol=tolerance)
    np.testing.assert_allclose(state.velocities.numpy(), reference.get_velocities(), rtol=0, atol=tolerance)
    np.testing.assert_allclose(state.cell.numpy(), reference.cell.array, rtol=0, atol=tolerance)


def assert_same_run(a, b):
    for name in ("positions", "velocities", "cell", "forces", "energy"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    if a.stress is not None:
        assert torch.equal(a.stress, b.stress)
    assert a.step == b.step


NVE = records.NVE(timestep_fs=DT_FS, dof="ase")
CSVR_RECORD = records.CSVR(timestep_fs=DT_FS, temperature_K=600.0, tau_fs=20.0, dof="ase")
NVT_BERENDSEN = records.Berendsen(timestep_fs=DT_FS, temperature_K=600.0, taut_fs=20.0, dof="ase")
NPT_BERENDSEN = records.Berendsen(timestep_fs=DT_FS, temperature_K=600.0, taut_fs=20.0, pressure_GPa=1.0,
                                  taup_fs=50.0, compressibility_per_GPa=0.05, dof="ase")
INHOMOGENEOUS = dataclasses.replace(NPT_BERENDSEN, mask=(1, 0, 1))


# -------------------------------------------------------------------------- the scheme records

def test_records_are_registered_with_their_support():
    reg = scheme_registry.registry()
    nve, csvr, berendsen = (reg.integrator(name) for name in ("nve", "csvr", "berendsen"))
    assert nve.load() is VelocityVerlet and csvr.load() is CSVR and berendsen.load() is Berendsen
    assert nve.ensembles == ("nve",) and csvr.ensembles == ("nvt",) and set(berendsen.ensembles) == {"nvt", "npt"}
    assert berendsen.non_canonical and not nve.non_canonical and not csvr.non_canonical
    assert berendsen.report == "non-canonical: equilibration only"
    for record in (nve, csvr, berendsen):
        assert "fixcm" in record.constraints and record.conserved is not None or record is berendsen
        assert record.ase_dof == "constraints"
    assert csvr.state_keys == ("transferred_energy",) and nve.state_keys == () and berendsen.state_keys == ()
    for public in (NVE, CSVR_RECORD, NVT_BERENDSEN, NPT_BERENDSEN, INHOMOGENEOUS):
        assert scheme_registry.integrator_refusals(public.to_config(seed=0),
                                                   variable_cell=public.to_config(seed=0).variable_cell) == []


@pytest.mark.parametrize("record", [NVE, CSVR_RECORD, NVT_BERENDSEN, NPT_BERENDSEN, INHOMOGENEOUS],
                         ids=["nve", "csvr", "nvt-berendsen", "npt-berendsen", "inhomogeneous"])
def test_buffers_have_policies_and_addresses_stay_fixed(record):
    atoms = nacl(seed=2)
    variable = record.to_config(seed=0).variable_cell
    state = new_state(atoms, compute_stress=variable)
    dyn = build(record, state, noise_batch=7)
    assert type(dyn).__name__ == {"nve": "VelocityVerlet", "csvr": "CSVR"}.get(dyn.config.scheme, "Berendsen")
    assert dyn.variable_cell == variable
    field = Forces(atoms)
    start(state, dyn, field)
    buffers = dyn.buffers(state)
    for name in buffers:
        if name not in ("positions", "velocities", "cell"):
            assert policy_for(f"integrator.{name}").owner == "integrator", name
    assert buffers["positions"] is state.positions and buffers["velocities"] is state.velocities
    if variable:
        assert buffers["cell"] is state.cell
    pointers = {name: tensor.data_ptr() for name, tensor in buffers.items()}
    run_md(state, dyn, field, 10)
    if dyn.config.scheme != "nve":
        dyn.set_temperature(450.0)
    assert {name: tensor.data_ptr() for name, tensor in dyn.buffers(state).items()} == pointers
    assert dyn.mid_step_tensors(state)[0] is state.velocities
    assert set(dyn.state_dict()) == set(dyn.record.state_keys)


def test_output_values_and_the_conserved_quantity():
    atoms = nacl(seed=3)
    for record in (NVE, CSVR_RECORD, NVT_BERENDSEN, NPT_BERENDSEN):
        variable = record.to_config(seed=0).variable_cell
        state = new_state(atoms, compute_stress=variable)
        dyn = build(record, state)
        field = Forces(atoms)
        start(state, dyn, field)
        run_md(state, dyn, field, 3)
        values = dyn.output_values(state)
        expected = {"positions", "momenta", "forces", "energy", "kinetic"}
        expected |= {"cell", "stress"} if variable else set()
        expected |= {CONSERVED_KEY} if dyn.record.conserved else set()
        assert set(values) == expected, record
        kinetic = (0.5 * (state.masses * state.velocities ** 2).sum()).reshape(1)
        assert torch.equal(values["kinetic"], kinetic)
        if dyn.config.scheme == "nve":
            assert torch.equal(values[CONSERVED_KEY], kinetic + state.energy)
        if dyn.config.scheme == "csvr":
            assert torch.equal(values[CONSERVED_KEY], kinetic + state.energy - dyn.transferred_energy)


def test_degrees_of_freedom_follow_d2():
    atoms = nacl()
    n = len(atoms)
    for record, auto in ((NVE, 3 * n - 3), (CSVR_RECORD, 3 * n - 3), (NVT_BERENDSEN, 3 * n - 3)):
        for dof, expected in (("ase", 3 * n), ("auto", auto)):
            state = new_state(atoms)
            dyn = build(dataclasses.replace(record, dof=dof), state)
            assert dyn.degrees_of_freedom(state) == expected
    state = new_state(atoms)
    dyn = build(dataclasses.replace(CSVR_RECORD, dof="auto", fixcm=False), state)
    assert dyn.degrees_of_freedom(state) == 3 * n
    assert dyn.ndof == 3 * n
    assert float(dyn.target_kinetic) == pytest.approx(0.5 * units.kB * 600.0 * 3 * n, rel=1e-15)


# -------------------------------------------------------------------------- refusals

def test_refusals():
    atoms = nacl()
    with pytest.raises(ValueError, match="nve"):
        VelocityVerlet(new_state(atoms), CSVR_RECORD.to_config(seed=0))
    with pytest.raises(ValueError, match="nvt"):
        CSVR(new_state(atoms), NVE.to_config(seed=0))
    with pytest.raises(ValueError, match="stress"):
        build(NPT_BERENDSEN, new_state(atoms))
    still = atoms.copy()
    still.set_velocities(np.zeros((len(atoms), 3)))
    with pytest.raises(ValueError, match="kinetic energy is zero"):
        build(CSVR_RECORD, new_state(still))
    dyn = build(NVE, new_state(atoms))
    with pytest.raises(NotImplementedError, match="temperature"):
        dyn.set_temperature(300.0)
    with pytest.raises(NotImplementedError):
        dyn.set_pressure(1.0)
    with pytest.raises(NotImplementedError):
        build(NVT_BERENDSEN, new_state(atoms)).set_pressure(1.0)


# -------------------------------------------------------------------------- NVE

def test_nve_matches_ase_velocity_verlet():
    atoms = nacl(seed=4)
    reference = reference_atoms(atoms)
    ase_dyn = ASEVelocityVerlet(reference, timestep=DT_FS * units.fs)
    state = new_state(atoms)
    field = Forces(atoms)
    dyn = build(NVE, state)
    start(state, dyn, field)
    energies = []
    for _ in range(STEPS):
        ase_dyn.run(1)
        md_step(state, dyn, field)
        energies.append(float(dyn.output_values(state)[CONSERVED_KEY][0]))
    assert field.calls == STEPS + 1
    assert_close(state, reference)
    assert energies[-1] == pytest.approx(reference.get_total_energy(), abs=1e-10)
    assert np.ptp(energies) < 1e-2 * abs(energies[0]) + 1e-3  # recorded, not judged: velocity Verlet conserves
    total = (state.velocities * state.masses).sum(dim=0)
    assert float(total.abs().max()) < 1e-12  # fixcm keeps the total momentum at zero


def test_nve_fixcm_removes_the_starting_momentum():
    atoms = nacl(seed=5)
    atoms.set_velocities(atoms.get_velocities() + np.array([0.01, -0.02, 0.005]))
    state = new_state(atoms)
    field = Forces(atoms)
    dyn = build(NVE, state)
    start(state, dyn, field)
    assert float((state.velocities * state.masses).sum(dim=0).abs().max()) < 1e-13
    run_md(state, dyn, field, 5)
    assert float((state.velocities * state.masses).sum(dim=0).abs().max()) < 1e-13
    free = new_state(atoms)
    free_dyn = build(dataclasses.replace(NVE, fixcm=False), free)
    start(free, free_dyn, Forces(atoms))
    np.testing.assert_array_equal(free.velocities.numpy(), atoms.get_velocities())


# -------------------------------------------------------------------------- CSVR

@pytest.mark.parametrize("noise_batch", [1, 7, 1000])
def test_csvr_noise_is_ases_stream(noise_batch):
    """Per step one standard_normal, then 2 standard_gamma(0.5 (ndof - 1)), as ASE's Bussi draws them."""
    atoms = nacl()
    state = new_state(atoms)
    dyn = build(CSVR_RECORD, state, noise_batch=noise_batch)
    assert tuple(state.rng.batch.shape) == (noise_batch, 2) and state.rng.cursor == noise_batch
    reference = np.random.default_rng(SEED)
    ndof = 3 * len(atoms)
    for _ in range(20):
        dyn.prepare_step(state)
        normal = reference.standard_normal(size=(1,))[0]  # ASE 3.27's draw
        gamma = 2.0 * reference.standard_gamma(0.5 * (ndof - 1))
        assert float(dyn.noise[0]) == normal and float(dyn.noise[1]) == gamma
    assert state.rng.cursor == (20 - 1) % noise_batch + 1


def run_csvr_pair(noise_batch=256, steps=STEPS):
    atoms = nacl(seed=6)
    reference = reference_atoms(atoms)
    ase_dyn = Bussi(reference, timestep=DT_FS * units.fs, temperature_K=600.0, taut=20.0 * units.fs,
                    rng=np.random.default_rng(SEED))
    state = new_state(atoms)
    field = Forces(atoms)
    dyn = build(CSVR_RECORD, state, noise_batch=noise_batch)
    start(state, dyn, field)
    for _ in range(steps):
        ase_dyn.run(1)
        md_step(state, dyn, field)
    return reference, ase_dyn, state, dyn, field


@pytest.mark.parametrize("noise_batch", [1, 7, 1000])
def test_csvr_matches_ase_bussi(noise_batch):
    reference, ase_dyn, state, dyn, field = run_csvr_pair(noise_batch)
    assert field.calls == STEPS + 1
    assert_close(state, reference)
    assert float(dyn.transferred_energy) == pytest.approx(ase_dyn.transferred_energy, abs=1e-11)
    assert abs(ase_dyn.transferred_energy) > 1e-3  # the thermostat really worked
    conserved = reference.get_total_energy() - ase_dyn.transferred_energy
    assert float(dyn.output_values(state)[CONSERVED_KEY][0]) == pytest.approx(conserved, abs=1e-10)


def test_csvr_ndof_change_redraws_the_rows_from_the_cursor():
    """4-T04's FixAtoms events change ndof: the rows not yet used are drawn again with the new gamma shape."""
    atoms = nacl()
    state = new_state(atoms)
    dyn = build(CSVR_RECORD, state, noise_batch=7)
    for _ in range(3):
        dyn.prepare_step(state)
    used = state.rng.batch[:3].clone()
    reference = np.random.default_rng(SEED)
    for _ in range(7):  # the first batch, drawn already
        reference.standard_normal(), reference.standard_gamma(0.5 * (3 * len(atoms) - 1))
    state.fixed_indices = (0, 1)  # what 4-T04's event changes; the count follows
    dyn.refresh_degrees_of_freedom(state)
    ndof = 3 * (len(atoms) - 2)
    assert dyn.ndof == ndof
    assert float(dyn.target_kinetic) == pytest.approx(0.5 * units.kB * 600.0 * ndof, rel=1e-15)
    assert torch.equal(state.rng.batch[:3], used) and state.rng.cursor == 3
    for row in range(3, 7):
        assert float(state.rng.batch[row, 0]) == reference.standard_normal()
        assert float(state.rng.batch[row, 1]) == 2.0 * reference.standard_gamma(0.5 * (ndof - 1))
    unchanged = state.rng.batch.clone()
    dyn.refresh_degrees_of_freedom(state)  # no change: nothing drawn
    assert torch.equal(state.rng.batch, unchanged)


# -------------------------------------------------------------------------- Berendsen

def test_nvt_berendsen_matches_ase():
    atoms = nacl(seed=7)
    reference = reference_atoms(atoms)
    ase_dyn = NVTBerendsen(reference, timestep=DT_FS * units.fs, temperature_K=600.0, taut=20.0 * units.fs,
                           fixcm=True)
    state = new_state(atoms)
    field = Forces(atoms)
    dyn = build(NVT_BERENDSEN, state)
    start(state, dyn, field)
    for _ in range(STEPS):
        ase_dyn.run(1)
        md_step(state, dyn, field)
    assert field.calls == STEPS + 1
    assert_close(state, reference)
    assert reference.get_temperature() > 400  # the thermostat heated it


def test_berendsen_clamps_the_scaling_as_ase():
    atoms = nacl(seed=8)
    for target in (5000.0, 1.0):  # lambda 1.1 and 0.9
        reference = reference_atoms(atoms)
        ase_dyn = NVTBerendsen(reference, timestep=DT_FS * units.fs, temperature_K=target, taut=DT_FS * units.fs)
        state = new_state(atoms)
        field = Forces(atoms)
        dyn = build(dataclasses.replace(NVT_BERENDSEN, temperature_K=target, taut_fs=DT_FS), state)
        start(state, dyn, field)
        for _ in range(5):
            ase_dyn.run(1)
            md_step(state, dyn, field)
        assert_close(state, reference)


class Reordered:
    """ASE's NPT Berendsen step with FastMACE's order: the first kick uses the forces of the previous step
    (at the unscaled cell), so a step makes one force call. Otherwise ASE 3.27's step, line for line."""

    def step(self, forces=None):
        if forces is None:
            forces = self.atoms.get_forces(md=True)  # cached: the end of the previous step
        NVTBerendsen.scale_velocities(self)
        self.scale_positions_and_cell()
        p = self.atoms.get_momenta()
        p += 0.5 * self.dt * forces
        if self.fix_com:
            psum = p.sum(axis=0) / float(len(p))
            p = p - psum
        self.atoms.set_positions(self.atoms.get_positions() + self.dt * p / self.atoms.get_masses()[:, np.newaxis])
        self.atoms.set_momenta(p)
        forces = self.atoms.get_forces(md=True)
        self.atoms.set_momenta(self.atoms.get_momenta() + 0.5 * self.dt * forces)
        return forces


class ReorderedNPTBerendsen(Reordered, NPTBerendsen):
    pass


class ReorderedInhomogeneous(Reordered, Inhomogeneous_NPTBerendsen):
    pass


def ase_npt(cls, atoms, mask=None):
    options = dict(timestep=DT_FS * units.fs, temperature_K=600.0, taut=20.0 * units.fs,
                   pressure_au=1.0 * units.GPa, taup=50.0 * units.fs, compressibility_au=0.05 / units.GPa,
                   fixcm=True)
    if mask is not None:
        options["mask"] = mask
    return cls(atoms, **options)


def run_npt_pair(record, cls, mask=None, steps=STEPS):
    atoms = nacl(seed=9)
    reference = reference_atoms(atoms)
    ase_dyn = ase_npt(cls, reference, mask)
    state = new_state(atoms, compute_stress=True)
    field = Forces(atoms)
    dyn = build(record, state)
    start(state, dyn, field)
    for _ in range(steps):
        ase_dyn.run(1)
        md_step(state, dyn, field)
    return atoms, reference, state, field


@pytest.mark.parametrize("record,cls,mask", [(NPT_BERENDSEN, ReorderedNPTBerendsen, None),
                                             (INHOMOGENEOUS, ReorderedInhomogeneous, (1, 0, 1))],
                         ids=["isotropic", "inhomogeneous"])
def test_npt_berendsen_matches_the_reordered_reference(record, cls, mask):
    atoms, reference, state, field = run_npt_pair(record, cls, mask)
    assert field.calls == STEPS + 1  # one force call per step
    assert_close(state, reference)
    start_lengths, lengths = atoms.cell.lengths(), reference.cell.lengths()
    assert abs(reference.get_volume() / atoms.get_volume() - 1) > 1e-3  # the barostat moved the cell
    if mask is not None:
        assert lengths[1] == pytest.approx(start_lengths[1], rel=1e-14)  # the masked axis stays
        assert abs(lengths[0] / start_lengths[0] - 1) > 1e-4


# Measured on this test (4-T02, 23 October 2026): after 200 steps the one-call order departs from ASE's
# two-call step by 9.6e-6 (isotropic) and 2.4e-6 (inhomogeneous) in the relative volume and by 2.4e-4 and
# 2.8e-4 Angstrom in the positions, while the volume moved by 7.6 % and 5.1 %. The bounds keep a margin.
ASE_VOLUME_BOUND = 1e-4
ASE_POSITION_BOUND = 2e-3


@pytest.mark.parametrize("record,cls,mask", [(NPT_BERENDSEN, NPTBerendsen, None),
                                             (INHOMOGENEOUS, Inhomogeneous_NPTBerendsen, (1, 0, 1))],
                         ids=["isotropic", "inhomogeneous"])
def test_npt_berendsen_stays_near_ase(record, cls, mask):
    """ASE's own step evaluates the forces twice (after the cell scaling, and after the drift)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atoms, reference, state, field = run_npt_pair(record, cls, mask)
    volume = float(torch.linalg.det(state.cell))
    assert abs(volume / reference.get_volume() - 1) < ASE_VOLUME_BOUND
    assert np.abs(state.positions.numpy() - reference.positions).max() < ASE_POSITION_BOUND


def test_berendsen_pressure_target_is_a_device_fill():
    atoms = nacl(seed=10)
    state = new_state(atoms, compute_stress=True)
    dyn = build(NPT_BERENDSEN, state)
    pointer = dyn.pressure.data_ptr()
    dyn.set_pressure(2.5)
    assert dyn.pressure.data_ptr() == pointer
    assert float(dyn.pressure) == pytest.approx(2.5 * units.GPa, rel=1e-15)
    np.testing.assert_array_equal(dyn.externalstress_host, -2.5 * units.GPa)
    dyn.restore_externalstress(-1.0 * units.GPa)
    assert float(dyn.pressure) == -(-1.0 * units.GPa)


# -------------------------------------------------------------------------- restarts

@pytest.mark.parametrize("record", [NVE, CSVR_RECORD, NVT_BERENDSEN, NPT_BERENDSEN, INHOMOGENEOUS],
                         ids=["nve", "csvr", "nvt-berendsen", "npt-berendsen", "inhomogeneous"])
def test_restart_round_trip_and_interrupted_equals_uninterrupted(tmp_path, record):
    """Run, write a restart mid-batch, resume in a fresh state and integrator: bitwise the uninterrupted run."""
    atoms = nacl(seed=12)
    variable = record.to_config(seed=0).variable_cell
    state = new_state(atoms, compute_stress=variable)
    dyn = build(record, state, noise_batch=7)
    field = Forces(atoms)
    start(state, dyn, field)
    run_md(state, dyn, field, 10)
    if dyn.config.scheme == "csvr":
        assert 0 < state.rng.cursor < 7
    path = tmp_path / restart_io.RESTART_NAME
    restart_io.write_restart(path, state, engine="gpu", integrator_state=dyn.state_dict(),
                             variable_cell=dyn.variable_cell)

    resumed = new_state(atoms, compute_stress=variable, seed=123)
    resumed_dyn = build(record, resumed, noise_batch=7)
    resumed_field = Forces(atoms)
    start(resumed, resumed_dyn, resumed_field)  # the engine's construction, then the restart replaces it
    restart_io.load_state(restart_io.read_restart(path, engine="gpu"), resumed, resumed_dyn)
    assert_same_run(resumed, state)
    if dyn.config.scheme == "csvr":
        assert torch.equal(resumed_dyn.transferred_energy, dyn.transferred_energy)

    run_md(state, dyn, field, 20)
    run_md(resumed, resumed_dyn, resumed_field, 20)
    assert_same_run(resumed, state)
    for name, value in dyn.state_dict().items():
        np.testing.assert_array_equal(resumed_dyn.state_dict()[name], value)


def test_csvr_state_dict_round_trip():
    atoms = nacl()
    state = new_state(atoms)
    dyn = build(CSVR_RECORD, state)
    dyn.transferred_energy.fill_(0.125)
    saved = dyn.state_dict()
    assert set(saved) == {"transferred_energy"}
    pointer = dyn.transferred_energy.data_ptr()
    dyn.transferred_energy.zero_()
    dyn.load_state_dict(saved)
    assert dyn.transferred_energy.data_ptr() == pointer and float(dyn.transferred_energy) == 0.125
    with pytest.raises(ValueError):
        dyn.load_state_dict({})
    with pytest.raises(ValueError):
        build(NVE, new_state(atoms)).load_state_dict({"transferred_energy": 0.0})
