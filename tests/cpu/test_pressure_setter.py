"""BAOAB's pressure setter: scheduled pressure targets as device fills (SPEC §10.1; plan 3b-T05).

``LangevinBAOAB.set_pressure`` and ``set_external_stress`` (GPa) fill the device tensor the half steps
read, in place, between steps, as ``set_temperature`` fills ``sqrt(kB T)``. Checked on the CPU in
float64:

- a pressure ramp reproduces ASE's ``LangevinBAOAB`` whose ``externalstress`` is replaced between
  steps the same way (both take effect from the next cell-force update);
- the fill keeps the tensor's address, so captured graphs read the new target without recapture;
  replayed and pipelined engines equal eager ones under a ramp;
- a run whose target was set to P equals, bit for bit, a run configured with P;
- the current target is in the restart's integrator record, and an interrupted run with a ramp
  resumes bit for bit as the uninterrupted one;
- fixed-cell integrators and malformed targets are refused.
"""

import dataclasses
import warnings

import numpy as np
import pytest
import torch
from ase import units
from ase.md.langevinbaoab import LangevinBAOAB as ASELangevinBAOAB
from test_canonical_restart import NEIGHBOURS, assert_same_values, lj_engine, restart_atoms
from test_engine import argon, build
from test_integrator_qualification import (
    SEED,
    TIMESTEP_FS,
    CalculatorForces,
    calculator,
    config,
    crystal,
    deviation,
)

from fastmace.config import EngineConfig, IntegratorConfig
from fastmace.integrators.baoab import LangevinBAOAB
from fastmace.integrators.langevin import Langevin
from fastmace.io import restart
from fastmace.state import SimulationState

STEPS = 60
TEMPERATURE_K = 300.0
TOLERANCE = 1e-9


def ramp_GPa(step, hydrostatic):
    """0 -> 2 GPa pressure over the run (ASE sign: negative stress); anisotropic for the full cell."""
    pressure = 2.0 * step / STEPS
    if hydrostatic:
        return -pressure
    return (-pressure, -1.5 * pressure, -0.5 * pressure, 0.1 * pressure, 0.0, -0.2 * pressure)


def run_ase(hydrostatic):
    atoms = crystal()
    atoms.calc = calculator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ASE's heuristic-mass notice
        dyn = ASELangevinBAOAB(atoms, timestep=TIMESTEP_FS * units.fs, T_tau=100.0 * units.fs,
                               P_tau=400.0 * units.fs, hydrostatic=hydrostatic,
                               externalstress=np.asarray(ramp_GPa(0, hydrostatic)) * units.GPa,
                               temperature_K=TEMPERATURE_K, rng=np.random.default_rng(SEED))
    for step in range(1, STEPS + 1):
        dyn._set_externalstress_hydrostatic(np.asarray(ramp_GPa(step, hydrostatic)) * units.GPa, hydrostatic)
        dyn.run(1)
    return atoms


def run_fastmace(hydrostatic):
    atoms = crystal()
    state = SimulationState.from_atoms(atoms, "cpu", SEED, compute_stress=True)
    field = CalculatorForces(atoms)
    dyn = LangevinBAOAB(state, config(hydrostatic, True), temperature_K=TEMPERATURE_K)
    dyn.set_external_stress(ramp_GPa(0, hydrostatic))
    field(state)
    dyn.start(state)
    for step in range(1, STEPS + 1):
        dyn.set_external_stress(ramp_GPa(step, hydrostatic))
        dyn.prepare_step(state)
        dyn.first_half(state)
        field(state)
        dyn.second_half(state)
    return state


@pytest.mark.parametrize("hydrostatic", [True, False], ids=["hydrostatic", "full"])
def test_pressure_ramp_reproduces_ase(hydrostatic):
    ase_atoms = run_ase(hydrostatic)
    state = run_fastmace(hydrostatic)
    assert deviation(state, ase_atoms) < TOLERANCE
    assert ase_atoms.get_volume() < crystal().get_volume()  # the rising pressure compressed it


# --------------------------------------------------------------------------- the fill


def npt_integrator(cell_mode="hydrostatic", **options):
    state = SimulationState.from_atoms(argon(), "cpu", 7, compute_stress=True)
    return LangevinBAOAB(state, IntegratorConfig(ensemble="npt", cell_mode=cell_mode, **options))


@pytest.mark.parametrize("cell_mode", ["hydrostatic", "full"])
def test_fill_is_in_place(cell_mode):
    dyn = npt_integrator(cell_mode)
    tensor, pointer = dyn.externalstress, dyn.externalstress.data_ptr()
    dyn.set_pressure(1.5)
    assert dyn.externalstress is tensor and tensor.data_ptr() == pointer
    expected = -1.5 * units.GPa if cell_mode == "hydrostatic" else -1.5 * units.GPa * np.identity(3)
    np.testing.assert_array_equal(tensor.numpy(), expected)
    np.testing.assert_array_equal(dyn.externalstress_host, expected)


def test_full_cell_takes_voigt_components():
    dyn = npt_integrator("full")
    dyn.set_external_stress((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    expected = np.array([[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]]) * units.GPa
    np.testing.assert_array_equal(dyn.externalstress.numpy(), expected)


@pytest.mark.parametrize("cell_mode", ["hydrostatic", "full"])
def test_setting_the_target_equals_configuring_it(cell_mode):
    """A target set before the first step gives the same numbers as a config with that pressure."""
    configured = lj_engine(npt=True, cell_mode=cell_mode, mode="checked")
    filled = lj_engine(npt=True, cell_mode=cell_mode, mode="checked")
    # lj_engine configures 0.5 GPa: the fill goes elsewhere and back, the arithmetic must be the config's.
    filled.integrator.set_pressure(3.0)
    filled.integrator.set_pressure(0.5)
    configured.run(20)
    filled.run(20)
    assert_same_values(filled.sync(), configured.sync())


def ramp(engine, steps, start=0):
    for step in range(start + 1, start + steps + 1):
        engine.integrator.set_temperature(min(2.0 * step, 300.0))
        engine.integrator.set_pressure(0.5 + 0.05 * step)
        engine.step()


@pytest.mark.parametrize("cell_mode", ["hydrostatic", "full"])
def test_replayed_engines_follow_the_fills(cell_mode):
    """Pipelined steps replayed from 'captured' stages (speculative graphs are not offered for NPT)."""
    # As test_npt_modes_agree: exact buckets and no early rebuilds, which would reorder the edges.
    exact = dataclasses.replace(NEIGHBOURS, edge_multiple=1, edge_bucket_fraction=0.0)
    eager = build(npt=True, cell_mode=cell_mode, neighbours=exact)
    replayed = build(npt=True, cell_mode=cell_mode, neighbours=exact, replay=True,
                     engine=EngineConfig(pipelined="on", pipeline_rebuild_fraction=1.0, device="cpu"))
    ramp(eager, 40)
    ramp(replayed, 20)
    replayed.flush()
    replayed._pipe_edges = 1  # step 21's B is redone for its count, in step 22's call after step 22's fill
    ramp(replayed, 20, start=20)
    assert_same_values(replayed.sync(), eager.sync())
    assert replayed.captures and replayed.pipelined_redos >= 1
    assert not np.array_equal(eager.sync()["cell"], argon().cell.array)


def test_registry_names_the_target():
    engine = lj_engine(npt=True, mode="pipelined")
    entries = engine.registry.entries()
    policy, tensor = entries["integrator.externalstress"]
    assert tensor is engine.integrator.externalstress
    assert policy.restart == "derived"


# --------------------------------------------------------------------------- restart


@pytest.mark.parametrize("cell_mode", ["hydrostatic", "full"])
def test_restart_records_the_current_target(tmp_path, cell_mode):
    engine = lj_engine(npt=True, cell_mode=cell_mode, mode="checked")
    ramp(engine, 5)
    path = tmp_path / "restart.npz"
    restart.write_engine_restart(path, engine, name="fastmace")
    record = restart.integrator_record(restart.read_restart(path))
    np.testing.assert_array_equal(np.asarray(record["externalstress"]), engine.integrator.externalstress_host)
    assert record["temperature_K"] == 10.0


def test_fixed_cell_records_no_external_stress(tmp_path):
    engine = lj_engine(npt=False, mode="checked")
    engine.integrator.set_temperature(10.0)
    path = tmp_path / "restart.npz"
    restart.write_engine_restart(path, engine, name="fastmace")
    assert "externalstress" not in restart.integrator_record(restart.read_restart(path))


@pytest.mark.parametrize("mode", ["checked", "pipelined"])
@pytest.mark.parametrize("cell_mode", ["hydrostatic", "full"])
def test_interrupted_ramp_resumes_bitwise(tmp_path, mode, cell_mode):
    """Restart at step 24 of a pressure ramp: the resumed run equals the uninterrupted one, same cadence."""
    whole = lj_engine(npt=True, cell_mode=cell_mode, mode=mode)
    ramp(whole, 24)
    restart.write_engine_restart(tmp_path / "whole.npz", whole, name="fastmace")
    ramp(whole, 20, start=24)

    first = lj_engine(npt=True, cell_mode=cell_mode, mode=mode)
    ramp(first, 24)
    path = tmp_path / "first.npz"
    restart.write_engine_restart(path, first, name="fastmace")
    stored = restart.read_restart(path)
    configured = -0.5 * units.GPa  # lj_engine's 0.5 GPa: the restart's target must differ from it
    assert not np.allclose(restart.integrator_record(stored)["externalstress"], configured)
    resumed = lj_engine(restart_atoms(argon(), stored), npt=True, cell_mode=cell_mode, mode=mode)
    restart.resume_engine(stored, resumed)
    restart.restore_targets(stored, resumed.integrator)
    np.testing.assert_array_equal(resumed.integrator.externalstress.numpy(), first.integrator.externalstress.numpy())
    assert resumed.integrator.temperature_K == first.integrator.temperature_K
    ramp(resumed, 20, start=24)
    assert_same_values(resumed.sync(), whole.sync())


def test_restore_targets_reads_files_without_a_pressure():
    """Restart files before 3b-T05 have no "externalstress": the configured target stays."""
    dyn = npt_integrator(pressure_GPa=0.5)
    before = dyn.externalstress.clone()
    restart.restore_targets({"integrator": '{"class": "LangevinBAOAB", "config": {}, "temperature_K": 5.0}'}, dyn)
    assert torch.equal(dyn.externalstress, before) and dyn.temperature_K == 5.0


# --------------------------------------------------------------------------- refusals


def test_fixed_cell_integrators_refuse():
    state = SimulationState.from_atoms(argon(), "cpu", 7)
    dyn = Langevin(state, IntegratorConfig())
    with pytest.raises(NotImplementedError, match="fixed cell"):
        dyn.set_pressure(1.0)
    with pytest.raises(NotImplementedError, match="fixed cell"):
        dyn.set_external_stress(-1.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1", (1.0, 2.0, 3.0)])
def test_malformed_targets_are_refused(value):
    dyn = npt_integrator("full")
    before = dyn.externalstress.clone()
    with pytest.raises(ValueError):
        dyn.set_external_stress(value)
    with pytest.raises(ValueError):
        dyn.set_pressure(value)
    assert torch.equal(dyn.externalstress, before)


def test_hydrostatic_needs_a_scalar():
    dyn = npt_integrator("hydrostatic")
    with pytest.raises(ValueError, match="hydrostatic"):
        dyn.set_external_stress((1.0, 1.0, 1.0, 0.0, 0.0, 0.0))
