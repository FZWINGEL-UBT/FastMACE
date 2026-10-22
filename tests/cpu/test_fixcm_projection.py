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
    assert IntegratorConfig().fixcm_projection == "auto"  # the planner's rule (3-T22b)
    assert IntegratorConfig(fixcm_projection="mass_weighted").fixcm_projection == "mass_weighted"
    with pytest.raises(ValueError, match="fixcm_projection"):
        IntegratorConfig(fixcm_projection="plain")


def test_ase_projection_is_unchanged_bitwise():
    """The explicit "ase" choice is the default code path: same numbers, same bits."""
    atoms = licscl()
    network = bond_network(atoms)
    results = []
    for options in ({}, {"fixcm_projection": "ase"}, {"fixcm_projection": "auto"}):  # "auto" reads as "ase"
        state, dyn = integrator(atoms.copy(), **options)
        run(state, dyn, network, 25)
        results.append((state.positions.clone(), state.velocities.clone()))
    for other in results[1:]:
        for a, b in zip(results[0], other, strict=True):
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


def test_the_fused_stage_takes_both_projections():
    """3-T22b: the fused Triton stage implements the mass-weighted means too (bitwise against the PyTorch
    stage: tests/gpu/test_fused_projection_gpu.py, and on the CPU through Triton's interpreter,
    tests/cpu/test_fused_projection_interpreter.py)."""
    from fastmace.kernels.triton import fused_integrator

    atoms = licscl()
    state, dyn = integrator(atoms, fixcm_projection="mass_weighted")
    engine = SimpleNamespace(cuda=True, variable_cell=False, integrator=dyn, state=state,
                             force_field=SimpleNamespace(neighbours=None), _mids=(state.velocities,))
    assert fused_integrator.applicable(engine) == "no skin neighbour list"  # not the projection


def test_redraw_all_and_the_mass_weighted_thermostat_share_one_convention():
    """``redraw_all`` (3-T09) draws Σ m v² = (3N − 3) kT at zero momentum, which is what ``"mass_weighted"``
    holds (above: it is the run without fixcm minus its centre of mass). ASE's projection holds 3N kT, so
    with it every accepted ``redraw_all`` block lowers the kinetic energy by (N − 1)/N (3-T12's addendum;
    measured in docs/validation/3-T22-fixcm.md)."""
    from fastmace.mc import momenta

    atoms = licscl()
    masses = torch.tensor(atoms.get_masses())[:, None]
    n, kT, draws = len(atoms), 0.05, 20_000
    generator = torch.Generator().manual_seed(3)
    normals = torch.randn((draws, n, 3), generator=generator, dtype=torch.float64)
    twice_kinetic = torch.stack([(masses * momenta.redraw_all(masses, z, kT) ** 2).sum() for z in normals]) / kT
    assert abs(float(twice_kinetic.mean()) - (3 * n - 3)) < 4 * (2 * (3 * n - 3) / draws) ** 0.5
    # The same projector as the thermostat's mass-weighted noise.
    velocities = torch.sqrt(kT / masses) * normals[0]
    projected = velocities - (velocities * masses).sum(dim=0) / masses.sum()
    torch.testing.assert_close(momenta.redraw_all(masses, normals[0], kT), projected, rtol=0, atol=1e-15)


# --------------------------------------------------------------------------- the planner's rule (3-T22b)


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    pytest.importorskip("cuequivariance_torch")
    from _support.mc_fixtures import tiny_loaded

    return tiny_loaded(tmp_path_factory.mktemp("models"), (11, 17, 19))


@pytest.fixture
def planner(monkeypatch):
    from _support.mc_fixtures import cpu_planner

    return cpu_planner(monkeypatch)


def plan_simulation(loaded, *, mc=True, ensemble="nvt", **record):
    from _support.mc_fixtures import MC_WORKLOAD, mc_simulation
    from _support.structures import nacl
    from fastmace import integrators
    from fastmace.planner.workload import Workload

    if ensemble == "nvt":
        integrator = integrators.Langevin(timestep_fs=1.0, temperature_K=600.0, friction_per_fs=0.02, **record)
    else:
        integrator = integrators.LangevinBAOAB(timestep_fs=1.0, temperature_K=600.0, friction_per_fs=0.02)
    workload = Workload(**MC_WORKLOAD) if mc else Workload(max_temperature_K=600.0)
    return mc_simulation(loaded, nacl(repeat=1, rattle=0.05, seed=3), (11, 17, 19), integrator=integrator,
                         workload=workload)


def decision(sim):
    found = [entry for entry in sim.plan.decisions if entry[0] == "IntegratorConfig.fixcm_projection"]
    assert len(found) == 1, sim.plan.decisions
    return found[0][1], getattr(found[0][2], "code", None)


@pytest.mark.parametrize(("options", "expected"), [
    (dict(mc=True), ("mass_weighted", "fixcm-projection-mc")),
    (dict(mc=False), ("mass_weighted", "fixcm-projection-md")),  # decision 37 (3-T22c)
    (dict(mc=True, fixcm=False), ("ase", "fixcm-projection-ase")),
    (dict(mc=True, ensemble="npt"), ("ase", "fixcm-projection-ase")),
    (dict(mc=True, fixcm_projection="ase"), ("ase", "fixcm-projection-given")),
    (dict(mc=False, fixcm_projection="mass_weighted"), ("mass_weighted", "fixcm-projection-given")),
])
def test_the_plan_resolves_auto_with_a_decision(loaded, planner, options, expected):
    sim = plan_simulation(loaded, **options)
    assert decision(sim) == expected
    assert sim.plan.integrator.fixcm_projection == expected[0]
    assert sim._engine.integrator.config.fixcm_projection == expected[0]
    assert getattr(sim._engine.integrator, "mass_weighted_com", False) == (expected[0] == "mass_weighted")


def test_md_auto_is_the_mass_weighted_projection_and_ase_stays_available(loaded, planner):
    """Decision 37: "auto" for MD under fixcm is "mass_weighted", the same trajectory as asking for it;
    an explicit "ase" runs ASE's projection, a different trajectory."""
    runs = {}
    for name, record in (("auto", {}), ("mass_weighted", {"fixcm_projection": "mass_weighted"}),
                         ("ase", {"fixcm_projection": "ase"})):
        sim = plan_simulation(loaded, mc=False, **record)
        sim.run(steps=5)
        runs[name] = sim.committed()
    np.testing.assert_array_equal(runs["auto"].positions, runs["mass_weighted"].positions)
    np.testing.assert_array_equal(runs["auto"].momenta, runs["mass_weighted"].momenta)
    assert not np.array_equal(runs["auto"].momenta, runs["ase"].momenta)
    momentum = (runs["auto"].momenta).sum(axis=0)
    np.testing.assert_allclose(momentum, 0.0, atol=1e-12)


# --------------------------------------------------------------------------- reported temperatures (3-T22c)


def test_degrees_of_freedom_follow_the_projection():
    from fastmace.io.trajectory import degrees_of_freedom, kinetic_temperature

    nvt = IntegratorConfig()
    assert degrees_of_freedom(10, IntegratorConfig(fixcm_projection="mass_weighted")) == 27
    assert degrees_of_freedom(10, IntegratorConfig(fixcm_projection="ase")) == 30
    assert degrees_of_freedom(10, nvt) == 30  # "auto" outside a plan reads as "ase"
    assert degrees_of_freedom(10, IntegratorConfig(fixcm=False, fixcm_projection="mass_weighted")) == 30
    assert degrees_of_freedom(10, IntegratorConfig(ensemble="npt", fixcm_projection="mass_weighted")) == 30
    assert degrees_of_freedom(10, None) == 30
    from ase import units

    assert kinetic_temperature(0.5 * 27 * units.kB * 600.0, 27) == pytest.approx(600.0, rel=1e-15)


@pytest.mark.parametrize(("record", "degrees"), [({}, "3N-3"), ({"fixcm_projection": "ase"}, "3N")])
def test_thermo_rows_count_the_projections_degrees_of_freedom(loaded, planner, tmp_path, record, degrees):
    from ase import units

    from fastmace.io import output as output_io

    sim = plan_simulation_with_output(loaded, tmp_path / "out", record)
    natoms = len(sim._atoms)
    expected = 3 * natoms - 3 if degrees == "3N-3" else 3 * natoms
    assert sim.degrees_of_freedom == expected
    sim.run(steps=4)
    sim.flush_output()
    header, rows = output_io.read_thermo(tmp_path / "out")
    temperature = float(rows[-1][header.index("temperature_K")])
    kinetic = float(rows[-1][header.index("kinetic_eV_atom")]) * natoms
    assert temperature == pytest.approx(2 * kinetic / (expected * units.kB), rel=1e-12)
    sim.close()


def plan_simulation_with_output(loaded, directory, record):
    from _support.structures import nacl
    from fastmace import integrators
    from fastmace.io.output import Output
    from fastmace.model.forcefield import ForceField
    from fastmace.planner.workload import Workload
    from fastmace.simulation import Simulation

    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    integrator = integrators.Langevin(timestep_fs=1.0, temperature_K=600.0, friction_per_fs=0.02, **record)
    return Simulation(atoms, ForceField(loaded, atoms, precision="float64", species=[11, 17, 19]),
                      integrator=integrator, workload=Workload(max_temperature_K=600.0), seed=1,
                      output=Output(directory, thermo_every=1, trajectory_every=2, restart_every=None))


# --------------------------------------------------------------------------- resume (3-T22c)


@pytest.mark.parametrize("stored", ["missing", "auto"])
def test_a_resume_keeps_the_projection_its_record_holds(loaded, planner, tmp_path, stored):
    """A run that ran ASE's projection resumes with it: a record from before the option (no field), and an
    "auto" record that 3-T22b resolved to "ase" for MD (its stored plan says so)."""
    import json

    from fastmace.simulation import SIMULATION_NAME, Simulation

    sim = plan_simulation_with_output(loaded, tmp_path / "run", {"fixcm_projection": "ase"})
    sim.run(steps=3)
    sim.save_restart(tmp_path / "run")
    want = sim.committed()
    sim.run(steps=2)
    after = sim.committed()
    sim.close()
    path = tmp_path / "run" / SIMULATION_NAME
    record = json.loads(path.read_text())
    fields = record["integrator"]["fields"]
    if stored == "missing":
        del fields["fixcm_projection"]
    else:
        fields["fixcm_projection"] = "auto"
    path.write_text(json.dumps(record))
    resumed = Simulation.resume(tmp_path / "run")
    assert resumed._engine.integrator.config.fixcm_projection == "ase"
    assert resumed.degrees_of_freedom == 3 * len(resumed._atoms)
    np.testing.assert_array_equal(resumed.committed().positions, want.positions)
    resumed.run(steps=2)
    np.testing.assert_array_equal(resumed.committed().positions, after.positions)
    resumed.close()


def test_plan_records_without_the_field_read_as_ase():
    from fastmace.simulation import _integrator, _kept_projection

    old = {"class": "Langevin", "fields": {"timestep_fs": 1.0, "temperature_K": 300.0, "friction_per_fs": 0.02,
                                          "fixcm": True}}
    assert _integrator(old).fixcm_projection == "ase"
    auto = _integrator({**old, "fields": {**old["fields"], "fixcm_projection": "auto"}})
    assert _kept_projection(auto, {"integrator": {"fixcm": True}}).fixcm_projection == "ase"
    # What "auto" resolves to now is what it ran with: the record stays "auto" (save, load, save: same files).
    assert _kept_projection(auto, {"integrator": {"fixcm_projection": "mass_weighted"}}).fixcm_projection == "auto"
    assert _kept_projection(auto, None).fixcm_projection == "auto"  # nothing stored: the plan decides


def test_redraw_all_with_an_explicit_ase_projection_is_refused(loaded, planner):
    from fastmace.errors import PlanRefused
    from fastmace.mc import Swap, Transmute

    sim = plan_simulation(loaded, mc=True, fixcm_projection="ase")
    with pytest.raises(PlanRefused, match="redraw_all") as raised:
        sim.run(steps=2, mc=[Transmute(["Na", "K"], every=2, attempts=1)])  # redraw_all: the fixcm default
    assert "mc-redraw-all-ase-projection" in [reason.code for reason in raised.value.reasons]
    with pytest.raises(PlanRefused, match="redraw_all"):
        sim.run(steps=2, mc=[Swap([("Na", "K")], every=2, attempts=1, velocities="redraw_all")])
    sim.run(steps=2, mc=[Swap([("Na", "K")], every=2, attempts=1)])  # exchange keeps the momentum: allowed
    auto = plan_simulation(loaded, mc=True)  # the plan's mass_weighted: allowed
    auto.run(steps=2, mc=[Transmute(["Na", "K"], every=2, attempts=1)])
