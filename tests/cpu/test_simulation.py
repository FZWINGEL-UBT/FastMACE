"""``fm.Simulation`` on the CPU (plan task 1b-T17; SPEC §7.2, §7.6, §9.2, §9.6).

Tiny MH-1-class models in float64 (and float32 where a model is passed), NaCl, the minimal planner
on the CPU without compilation (eager stages). Covers: the constructor's refusals and defaults
(owner decisions D1, D7), ``run`` along schedules (every check before the first step, the state
unchanged on a refusal), ``mc=`` checks and the phase-3 refusal, output through the chunked writer,
``NonFiniteError``, ``committed()``, the frame hook, the report and ``metadata.json``.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest
import torch

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace import integrators
from fastmace.config import EngineConfig
from fastmace.engine import constraints
from fastmace.errors import NonFiniteError, PlanRefused, SpeciesNotInSuperset
from fastmace.io import output as output_io
from fastmace.io.output import Output
from fastmace.planner import minimal
from fastmace.planner.workload import Workload
from fastmace.protocols import hold, ramp

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

NA_CL = (11, 17)
TIMESTEP_FS = 1.0
LANGEVIN = integrators.Langevin(timestep_fs=TIMESTEP_FS, temperature_K=300.0, friction_per_fs=0.01)


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    pytest.importorskip("cuequivariance_torch")
    from fastmace.model import loading

    model = tiny_mace_model(list(NA_CL), **FAMILIES["mh1"])
    path = tmp_path_factory.mktemp("models") / "tiny-mh1-11-17.model"
    torch.save(model, path)
    loading._LICENCES_SHOWN.add(str(path))  # no licence warning noise
    return loading.load_pristine(str(path), trusted=True)


@pytest.fixture(autouse=True)
def cpu_planner(monkeypatch):
    """The Simulation's planner: the CPU without compilation, with a fresh process record per test."""
    record = constraints.ProcessRecord()
    planner = minimal.MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ={},
                                     process=record)
    monkeypatch.setattr(minimal, "default_planner", lambda force_field: planner)
    return planner


def _ff(loaded, atoms, precision="float64"):
    from fastmace.model.forcefield import ForceField

    return ForceField(loaded, atoms, precision=precision)


def _simulation(loaded, *, atoms=None, integrator=LANGEVIN, workload=None, output=None, seed=1, **options):
    from fastmace.simulation import Simulation

    atoms = nacl(repeat=1, rattle=0.05, seed=3) if atoms is None else atoms
    return Simulation(atoms, _ff(loaded, atoms), integrator=integrator, workload=workload, output=output, seed=seed,
                      **options)


def _positions(sim):
    return sim.committed().positions.copy()


def _spy_targets(sim):
    """[(step about to run, target K in effect)] for every step the simulation runs from now on."""
    engine = sim._engine
    integrator = engine.integrator
    current = {"T": None}
    original_set, original_step = integrator.set_temperature, engine.step

    def set_temperature(value):
        current["T"] = float(value)
        original_set(value)

    seen = []

    def step():
        seen.append((engine.state.step + 1, current["T"]))
        original_step()

    integrator.set_temperature = set_temperature
    engine.step = step
    return seen


# --------------------------------------------------------------------------- constructor


def test_default_workload_is_the_integrator_temperature(loaded):  # owner decision D7
    sim = _simulation(loaded)
    assert sim.plan.workload == Workload(max_temperature_K=300.0)
    assert sim.plan.decision("workload.max_temperature_K")[1] == "300 K"
    assert sim.plan.tier == "graph"


def test_gate_is_opt_in_and_skipped_by_default(loaded, tmp_path):  # owner decision D1
    from fastmace.planner.selfcheck import GateRecord

    sim = _simulation(loaded, output=Output(tmp_path / "run", trajectory_every=None, restart_every=None))
    assert isinstance(sim.gate, GateRecord)
    assert (sim.gate.status, sim.gate.check, sim.gate.reason) == ("skipped", "skip", 'opt-in: check="gate"')
    assert "gate:    not run" in sim.report.text()
    metadata = json.loads((tmp_path / "run" / "metadata.json").read_text())
    assert metadata["gate"]["status"] == "skipped"


# --------------------------------------------------------------------------- the gate (1b-T16), opt-in


@pytest.fixture
def no_engines_yet():
    """No production engine built in this process yet, so the gate's references run in-process (as T16's tests)."""
    from fastmace.engine import base

    saved = (base._PRODUCTION_ENGINES.count, base._PRODUCTION_ENGINES.first)
    base._forget_production_engines()
    yield
    base._PRODUCTION_ENGINES.count, base._PRODUCTION_ENGINES.first = saved


def _no_reference(*args, **kwargs):
    raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 12.00 GiB")


@pytest.mark.parametrize("npt", [False, True], ids=["nvt", "npt"])
def test_check_gate_passes_and_leaves_the_trajectory(loaded, no_engines_yet, tmp_path, npt):
    integrator = LANGEVIN
    if npt:
        integrator = integrators.LangevinBAOAB(timestep_fs=TIMESTEP_FS, temperature_K=300.0, friction_per_fs=0.01,
                                               pressure_GPa=0.5)
    gated = _simulation(loaded, integrator=integrator, check="gate", seed=8,
                        output=Output(tmp_path / "run", trajectory_every=None, restart_every=None))
    assert gated.gate.status == "passed"
    assert gated.gate.check == "gate"
    assert gated.gate.references["ran"] == "in-process"
    assert (gated.gate.stress_eV_per_A3 is None) == (not npt)
    assert "gate:    passed" in gated.report.text()
    assert "gate" in gated.report.setup_s
    metadata = json.loads((tmp_path / "run" / "metadata.json").read_text())
    assert metadata["gate"]["status"] == "passed"
    plain = _simulation(loaded, integrator=integrator, seed=8)
    for sim in (gated, plain):
        sim.run(steps=3)
    np.testing.assert_array_equal(_positions(gated), _positions(plain))  # the gate restores the state


def test_a_requested_gate_that_is_unavailable_refuses(loaded, no_engines_yet, monkeypatch):
    from fastmace.engine import base
    from fastmace.errors import GateUnavailable
    from fastmace.planner import selfcheck

    monkeypatch.setattr(selfcheck, "_load_reference", _no_reference)
    with pytest.raises(GateUnavailable, match="allow_unavailable") as raised:
        _simulation(loaded, check="gate")
    assert raised.value.gate_record.status == "unavailable"
    assert not base.production_engine_built()  # refused before any engine was built


def test_allow_unavailable_records_it_and_runs(loaded, no_engines_yet, monkeypatch):
    from fastmace.planner import selfcheck

    monkeypatch.setattr(selfcheck, "_load_reference", _no_reference)
    sim = _simulation(loaded, check="allow_unavailable")
    assert sim.gate.status == "unavailable" and "out of memory" in sim.gate.reason
    assert "gate:    unavailable" in sim.report.text()
    sim.run(steps=2)
    assert sim.committed().step == 2


def test_a_failing_gate_refuses_before_any_step(loaded, no_engines_yet, monkeypatch):
    from fastmace.errors import GateFailed
    from fastmace.planner import selfcheck

    original = selfcheck._read_outputs

    def poisoned(engine, stress):
        energy, forces, values = original(engine, stress)
        return energy + 1.0, forces, values  # 1 eV off: fails on energy

    monkeypatch.setattr(selfcheck, "_read_outputs", poisoned)
    with pytest.raises(GateFailed) as raised:
        _simulation(loaded, check="gate")
    assert raised.value.gate_record.status == "failed"


def test_unknown_check_mode_is_refused():
    from fastmace.simulation import Simulation

    with pytest.raises(ValueError, match="check"):
        Simulation(None, None, integrator=None, seed=1, check="maybe")


def test_species_with_a_force_field_is_refused(loaded):
    from fastmace.simulation import Simulation

    atoms = nacl(repeat=1)
    with pytest.raises(ValueError, match="species"):
        Simulation(atoms, _ff(loaded, atoms), species=["Na", "Cl"], integrator=LANGEVIN, seed=1)


def test_a_path_is_refused_with_a_pointer_to_fm_load(tmp_path):
    from fastmace.simulation import Simulation

    with pytest.raises(TypeError, match="fm.load"):
        Simulation(nacl(repeat=1), str(tmp_path / "some.model"), integrator=LANGEVIN, seed=1)


def test_integrator_must_be_a_public_record(loaded):
    from fastmace.config import IntegratorConfig
    from fastmace.simulation import Simulation

    atoms = nacl(repeat=1)
    with pytest.raises(TypeError, match="fm.integrators"):
        Simulation(atoms, _ff(loaded, atoms), integrator=IntegratorConfig(temperature_K=300.0), seed=1)


def test_integrator_temperature_above_the_workload_is_refused(loaded):
    with pytest.raises(PlanRefused) as caught:
        _simulation(loaded, workload=Workload(max_temperature_K=200.0))
    assert "target-above-max-temperature" in caught.value.codes


def test_non_periodic_atoms_are_refused_by_the_planner(loaded):
    atoms = nacl(repeat=1)
    atoms.pbc = (True, True, False)
    with pytest.raises(PlanRefused) as caught:
        _simulation(loaded, atoms=atoms)
    assert "non-3d-pbc" in caught.value.codes


def test_simulation_from_a_model_equals_simulation_from_a_force_field(loaded):
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    by_model = Simulation(atoms, loaded, integrator=LANGEVIN, seed=5)
    by_ff = Simulation(atoms, ForceField(loaded, atoms, precision="float32"), integrator=LANGEVIN, seed=5)
    assert by_model.force_field.precision == "float32"
    assert by_model.force_field.species == NA_CL
    assert by_model.plan == by_ff.plan
    for sim in (by_model, by_ff):
        sim.run(steps=4)
    np.testing.assert_array_equal(_positions(by_model), _positions(by_ff))
    assert by_model.committed().energy == by_ff.committed().energy


def test_species_with_a_model_sets_the_superset(loaded):
    from fastmace.simulation import Simulation

    sim = Simulation(nacl(repeat=1), loaded, species=["Cl", "Na"], integrator=LANGEVIN, seed=1)
    assert sim.force_field.species == NA_CL
    assert sim.plan.superset == NA_CL


def test_the_callers_atoms_are_not_changed(loaded):
    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    before = atoms.get_positions().copy()
    sim = _simulation(loaded, atoms=atoms)
    sim.run(steps=3)
    np.testing.assert_array_equal(atoms.get_positions(), before)


# --------------------------------------------------------------------------- run and schedules


def test_run_without_schedule_needs_steps_and_keeps_the_target(loaded):
    sim = _simulation(loaded)
    with pytest.raises(ValueError, match="steps"):
        sim.run()
    seen = _spy_targets(sim)
    sim.run(steps=3)
    assert sim.committed().step == 3
    assert [step for step, _ in seen] == [1, 2, 3]
    assert all(target in (None, 300.0) for _, target in seen)  # never filled with anything else
    assert float(sim._engine.integrator.temperature_K) == 300.0


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_run_refuses_non_positive_or_non_integer_steps(loaded, bad):
    sim = _simulation(loaded)
    with pytest.raises((TypeError, ValueError)):
        sim.run(steps=bad)
    assert sim.committed().step == 0


def test_run_along_a_schedule_in_pieces_equals_one_run(loaded):
    schedule = ramp(300.0, 200.0, ps=0.004) + hold(200.0, ps=0.003)  # 4 + 3 steps at 1 fs
    whole = _simulation(loaded, seed=7)
    pieces = _simulation(loaded, seed=7)
    seen = _spy_targets(pieces)
    whole.run(schedule=schedule)
    pieces.run(schedule=schedule, steps=3)
    assert pieces.schedule.step == 3 and pieces.schedule.remaining == 4
    pieces.run()
    assert pieces.committed().step == whole.committed().step == 7
    assert pieces.schedule.finished
    np.testing.assert_array_equal(_positions(pieces), _positions(whole))
    resolved = schedule.resolve(TIMESTEP_FS)
    assert [target for _, target in seen] == resolved.targets(1, 7)
    with pytest.raises(ValueError, match="finished"):
        pieces.run()
    with pytest.raises(ValueError, match="finished"):
        pieces.run(steps=1)
    pieces.run(schedule=hold(200.0, ps=0.002))  # a new schedule continues
    assert pieces.committed().step == 9


def _unchanged(sim, before, step=0):
    assert sim.committed().step == step
    np.testing.assert_array_equal(_positions(sim), before)


def test_a_schedule_above_max_temperature_is_refused_before_any_step(loaded):
    sim = _simulation(loaded, workload=Workload(max_temperature_K=400.0))
    before = _positions(sim)
    with pytest.raises(PlanRefused) as caught:
        sim.run(schedule=ramp(300.0, 500.0, ps=0.004))
    assert caught.value.codes == ("schedule-above-max-temperature",)
    _unchanged(sim, before)
    assert sim.schedule is None
    sim.run(schedule=ramp(300.0, 400.0, ps=0.004), steps=2)  # within the workload: fine
    assert sim.committed().step == 2


def test_steps_beyond_the_schedule_are_refused_before_any_step(loaded):
    sim = _simulation(loaded)
    before = _positions(sim)
    with pytest.raises(ValueError, match="beyond the schedule"):
        sim.run(schedule=hold(300.0, ps=0.003), steps=4)
    _unchanged(sim, before)
    assert sim.schedule is None
    sim.run(schedule=hold(300.0, ps=0.003), steps=2)
    with pytest.raises(ValueError, match="beyond the schedule"):
        sim.run(steps=2)
    _unchanged(sim, _positions(sim), step=2)


def test_schedule_must_be_a_schedule(loaded):
    sim = _simulation(loaded)
    with pytest.raises(TypeError, match="Schedule"):
        sim.run(schedule=[300.0, 200.0], steps=1)


# --------------------------------------------------------------------------- mc= (duck-typed; MC is phase 3)


@dataclasses.dataclass
class Swap:
    pairs: tuple
    every: int = 10
    attempts: int = 5


@dataclasses.dataclass
class Displace:
    every: int = 10


def _mc_workload(**changes):
    fields = dict(max_temperature_K=300.0, species_changes=True, mc_moves=("swap",), min_mc_temperature_K=250.0)
    fields.update(changes)
    return Workload(**fields)


def test_a_valid_mc_block_is_refused_with_mc_phase_3(loaded):
    sim = _simulation(loaded, workload=_mc_workload())
    before = _positions(sim)
    with pytest.raises(PlanRefused) as caught:
        sim.run(steps=2, mc=Swap(pairs=(("Na", "Cl"),)))
    assert caught.value.codes == ("mc-phase-3",)
    assert "MC: phase 3" in str(caught.value)
    _unchanged(sim, before)
    sim.run(steps=1, mc=None)  # detach: runs
    sim.run(steps=1)  # nothing attached: runs
    assert sim.committed().step == 2


def test_mc_with_an_undeclared_move_type(loaded):
    sim = _simulation(loaded, workload=_mc_workload())
    with pytest.raises(PlanRefused) as caught:
        sim.run(steps=2, mc=Displace())
    assert caught.value.codes == ("mc-move-not-declared", "mc-phase-3")


def test_mc_with_no_declared_moves(loaded):
    sim = _simulation(loaded)
    with pytest.raises(PlanRefused) as caught:
        sim.run(steps=2, mc=Swap(pairs=(("Na", "Cl"),)))
    assert "mc-move-not-declared" in caught.value.codes


def test_mc_below_the_minimum_mc_temperature_over_every_step(loaded):
    sim = _simulation(loaded, workload=_mc_workload())
    schedule = ramp(300.0, 240.0, ps=0.006)  # 290, 280, ..., 240 K
    with pytest.raises(PlanRefused) as caught:
        sim.run(schedule=schedule, mc=Swap(pairs=(("Na", "Cl"),)))
    assert caught.value.codes == ("mc-below-min-temperature", "mc-phase-3")
    with pytest.raises(PlanRefused) as caught:  # the first 4 steps stay at or above 250 K
        sim.run(schedule=schedule, steps=4, mc=Swap(pairs=(("Na", "Cl"),)))
    assert caught.value.codes == ("mc-phase-3",)
    assert sim.schedule is None and sim.committed().step == 0


def test_mc_naming_a_species_outside_the_superset(loaded):
    sim = _simulation(loaded, workload=_mc_workload())
    with pytest.raises(SpeciesNotInSuperset) as caught:
        sim.run(steps=2, mc=Swap(pairs=(("Na", "Nb"),)))
    assert caught.value.codes == ("species-not-in-superset", "mc-phase-3")
    assert 2 in caught.value.rules


def test_mc_refusals_come_before_the_schedule_attaches(loaded):
    sim = _simulation(loaded, workload=_mc_workload())
    with pytest.raises(PlanRefused):
        sim.run(schedule=hold(300.0, ps=0.003), mc=Swap(pairs=(("Na", "Cl"),)))
    assert sim.schedule is None


# --------------------------------------------------------------------------- non-finite energies (§9.6)


def _poison(sim, at_step):
    engine = sim._engine
    original = engine.step

    def step():
        original()
        engine.flush()
        if engine.state.step == at_step:
            engine.state.energy.fill_(math.nan)

    engine.step = step


def test_a_nan_energy_stops_the_run_at_the_thermo_row(loaded, tmp_path):
    sim = _simulation(loaded, output=Output(tmp_path / "run", thermo_every=1, trajectory_every=None,
                                            restart_every=None))
    _poison(sim, 3)
    with pytest.raises(NonFiniteError, match="step 3") as caught:
        sim.run(steps=10)
    assert "energy" in str(caught.value)
    # The row is written one step behind (asynchronous output), so the run stops at step 3 or 4.
    assert sim.committed().step in (3, 4)
    sim.close()
    header, rows = output_io.read_thermo(tmp_path / "run")
    assert [int(row[0]) for row in rows] == [0, 1, 2]  # nothing written for the non-finite step
    metadata = json.loads((tmp_path / "run" / "metadata.json").read_text())
    assert metadata["timing"]["status"] == "non-finite"


def test_a_nan_energy_stops_the_run_at_its_end_without_output(loaded):
    sim = _simulation(loaded)
    _poison(sim, 4)  # the committed energy of the last step
    with pytest.raises(NonFiniteError, match="step 4"):
        sim.run(steps=4)


# --------------------------------------------------------------------------- output, report, metadata


def test_output_rows_and_frames_through_the_chunked_writer(loaded, tmp_path):
    directory = tmp_path / "run"
    sim = _simulation(loaded, output=Output(directory, thermo_every=2, trajectory_every=3, restart_every=None))
    sim.run(schedule=ramp(300.0, 280.0, ps=0.004), steps=4)
    sim.run(steps=2, schedule=hold(280.0, ps=0.002))
    sim.close()
    header, rows = output_io.read_thermo(directory)
    assert header[:5] == ["step", "time_ps", "phase", "target_K", "temperature_K"]
    assert [int(row[0]) for row in rows] == [0, 2, 4, 6]
    assert [row[2] for row in rows] == ["cooling", "cooling", "cooling", "hold"]
    assert [float(row[3]) for row in rows] == [300.0, 290.0, 280.0, 280.0]
    frames = output_io.read_frames(directory)
    assert len(frames) == 3  # steps 0, 3, 6
    np.testing.assert_array_equal(frames[-1].get_positions(), _positions(sim))
    assert sorted(output_io.chunk_indices(directory)) == [0]


def test_output_values_equal_a_run_without_output(loaded, tmp_path):
    with_output = _simulation(loaded, seed=9, output=Output(tmp_path / "a", thermo_every=1, trajectory_every=1,
                                                            restart_every=None))
    without = _simulation(loaded, seed=9)
    for sim in (with_output, without):
        sim.run(steps=5)
    np.testing.assert_array_equal(_positions(with_output), _positions(without))
    with_output.close()
    frames = output_io.read_frames(tmp_path / "a")
    assert len(frames) == 6
    np.testing.assert_array_equal(frames[-1].get_positions(), _positions(without))


def test_a_directory_with_chunks_is_refused_before_any_step(loaded, tmp_path):
    out = Output(tmp_path / "run", thermo_every=1, trajectory_every=None, restart_every=None)
    first = _simulation(loaded, output=out)
    first.run(steps=1)
    first.close()
    with pytest.raises(FileExistsError, match="resume"):
        _simulation(loaded, output=out)


def test_metadata_json_and_the_report(loaded, tmp_path):
    directory = tmp_path / "run"
    sim = _simulation(loaded, output=Output(directory, thermo_every=1, trajectory_every=None, restart_every=None))
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["schema"] == "fastmace.metadata/1"
    assert metadata["gate"]["status"] == "skipped"
    assert metadata["report"]["speed"]["status"] == "pending"
    assert metadata["force_field_identity"]["digest"] == sim.force_field.identity.digest
    assert metadata["random"]["seed"] == 1
    sim.run(steps=3)
    sim.close()
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["report"]["speed"]["status"] == "whole-run mean"
    assert metadata["report"]["speed"]["last_step"] == 3
    assert metadata["timing"]["steps"] == 3
    assert metadata["timing"]["status"] == "complete"
    assert set(metadata["timing"]["setup_s"]) >= {"plan", "build"}
    assert sim.metadata()["schema"] == "fastmace.metadata/1"


def test_speed_is_measured_over_the_window(loaded, monkeypatch):
    from fastmace import report

    monkeypatch.setattr(report, "DEFAULT_WARMUP_STEPS", 2)
    monkeypatch.setattr(report, "DEFAULT_WINDOW_STEPS", 3)
    sim = _simulation(loaded)
    sim.run(steps=4)
    assert sim.report.measured is None  # steps 2-5 not reached yet
    sim.run(steps=2)
    assert sim.report.measured["first_step"] == 2 and sim.report.measured["last_step"] == 5
    assert sim.report.speed["status"] == "whole-run mean"


# --------------------------------------------------------------------------- committed() and the frame hook (T21)


def test_committed_is_a_read_only_snapshot(loaded):
    sim = _simulation(loaded)
    sim.run(steps=2)
    committed = sim.committed()
    assert committed.step == 2
    assert committed.positions.shape == (8, 3) and committed.momenta.shape == (8, 3)
    assert committed.forces.shape == (8, 3) and committed.cell.shape == (3, 3)
    assert committed.stress is None  # NVT
    assert isinstance(committed.energy, float)
    with pytest.raises(ValueError):
        committed.positions[0, 0] = 1.0
    sim.run(steps=1)
    assert committed.step == 2 and sim.committed().step == 3


def test_committed_has_the_stress_for_npt(loaded):
    baoab = integrators.LangevinBAOAB(timestep_fs=TIMESTEP_FS, temperature_K=300.0, friction_per_fs=0.01,
                                      pressure_GPa=0.5)
    sim = _simulation(loaded, integrator=baoab)
    sim.run(steps=2)
    committed = sim.committed()
    assert committed.stress.shape == (3, 3)
    assert not np.array_equal(committed.cell, nacl(repeat=1).cell.array)


def test_submit_frame_writes_through_the_async_queue(loaded):
    sim = _simulation(loaded)
    frames = []
    sim.run(steps=2)
    sim.submit_frame(lambda step, values: frames.append((step, values["positions"].copy())))
    want = _positions(sim)
    sim.run(steps=1)  # drained after the next step is launched
    assert [step for step, _ in frames] == [2]
    np.testing.assert_array_equal(frames[0][1], want)
    sim.submit_frame(lambda step, values: frames.append((step, None)))
    sim.flush_output()
    assert [step for step, _ in frames] == [2, 3]


# --------------------------------------------------------------------------- restart and resume (§10.6, after 1b-T14)


def _restart_output(directory, every=3):
    return Output(directory, thermo_every=1, trajectory_every=1, restart_every=every)


def test_interrupted_equals_uninterrupted_with_the_same_cadence(loaded, tmp_path):
    from fastmace.simulation import Simulation

    schedule = ramp(300.0, 250.0, ps=0.007)  # 7 steps at 1 fs
    whole = _simulation(loaded, seed=11, output=_restart_output(tmp_path / "whole"))
    whole.run(schedule=schedule)
    whole.close()

    first = _simulation(loaded, seed=11, output=_restart_output(tmp_path / "parts"))
    first.run(schedule=schedule, steps=5)  # the restart file is from step 3; steps 4 and 5 are lost
    del first  # killed: nothing closed or flushed beyond what the run wrote
    resumed = Simulation.resume(tmp_path / "parts")
    assert resumed.committed().step == 3
    assert resumed.schedule.step == 3 and resumed.schedule.remaining == 4
    resumed.run()
    resumed.close()

    np.testing.assert_array_equal(_positions(resumed), _positions(whole))
    np.testing.assert_array_equal(resumed.committed().momenta, whole.committed().momenta)
    assert resumed.committed().energy == whole.committed().energy
    want_header, want_rows = output_io.read_thermo(tmp_path / "whole")
    got_header, got_rows = output_io.read_thermo(tmp_path / "parts")
    assert got_header == want_header and got_rows == want_rows  # no duplicate or missing row
    frames = output_io.read_frames(tmp_path / "parts")
    assert len(frames) == len(output_io.read_frames(tmp_path / "whole")) == 8
    metadata = json.loads((tmp_path / "parts" / "metadata.json").read_text())
    assert [point["step"] for point in metadata["restart_points"]] == [3, 6]


def test_npt_interrupted_equals_uninterrupted(loaded, tmp_path):
    from fastmace.simulation import Simulation

    baoab = integrators.LangevinBAOAB(timestep_fs=TIMESTEP_FS, temperature_K=300.0, friction_per_fs=0.01,
                                      pressure_GPa=0.5, cell_mode="full")
    whole = _simulation(loaded, integrator=baoab, seed=4, output=_restart_output(tmp_path / "whole", every=2))
    whole.run(steps=5)
    first = _simulation(loaded, integrator=baoab, seed=4, output=_restart_output(tmp_path / "parts", every=2))
    first.run(steps=3)
    del first
    resumed = Simulation.resume(tmp_path / "parts")
    assert resumed.committed().step == 2
    resumed.run(steps=3)
    np.testing.assert_array_equal(_positions(resumed), _positions(whole))
    np.testing.assert_array_equal(resumed.committed().cell, whole.committed().cell)


def test_save_load_save_gives_identical_files(loaded, tmp_path):
    from fastmace.simulation import Simulation

    sim = _simulation(loaded, seed=2)
    sim.run(schedule=hold(300.0, ps=0.005), steps=2)
    sim.save_restart(tmp_path / "a" / "restart.npz")
    resumed = Simulation.resume(tmp_path / "a")
    resumed.save_restart(tmp_path / "b" / "restart.npz")
    assert (tmp_path / "a" / "restart.npz").read_bytes() == (tmp_path / "b" / "restart.npz").read_bytes()
    first = json.loads((tmp_path / "a" / "simulation.json").read_text())
    second = json.loads((tmp_path / "b" / "simulation.json").read_text())
    # The resume loads the checkpoint again, checked by its SHA-256: that is its trust decision.
    assert (first["model"].pop("trust"), second["model"].pop("trust")) == ("trusted", "sha256")
    assert first == second
    assert first["model"]["sha256"] == loaded.sha256
    assert first["structure"]["numbers"] == [int(z) for z in nacl(repeat=1).numbers]


def test_resume_continues_the_target_without_a_schedule(loaded, tmp_path):
    from fastmace.simulation import Simulation

    sim = _simulation(loaded, workload=Workload(max_temperature_K=400.0))
    sim.run(schedule=ramp(300.0, 350.0, ps=0.005))  # finished at 350 K
    sim.save_restart(tmp_path / "restart.npz")
    resumed = Simulation.resume(tmp_path)
    assert float(resumed._engine.integrator.temperature_K) == 350.0
    assert resumed.schedule.finished
    assert resumed.plan.workload == Workload(max_temperature_K=400.0)


def test_resume_refuses_another_force_field_identity(loaded, tmp_path):
    from fastmace.errors import RestartMismatch
    from fastmace.simulation import Simulation

    sim = _simulation(loaded)
    sim.run(steps=1)
    sim.save_restart(tmp_path / "restart.npz")
    path = tmp_path / "simulation.json"
    record = json.loads(path.read_text())
    record["force_field"]["precision"] = "float32"
    path.write_text(json.dumps(record))
    with pytest.raises(RestartMismatch, match="force_precision"):
        Simulation.resume(tmp_path)


def test_a_plan_change_needs_the_flag_and_makes_revision_1(loaded, tmp_path, monkeypatch, cpu_planner):
    from fastmace.config import NeighbourConfig
    from fastmace.errors import RestartMismatch
    from fastmace.simulation import Simulation

    sim = _simulation(loaded)
    sim.run(steps=2)
    sim.save_restart(tmp_path / "restart.npz")
    assert Simulation.resume(tmp_path).plan.revision == 0  # same plan: same revision
    other = minimal.MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ={},
                                   neighbours=NeighbourConfig(skin=0.5), process=constraints.ProcessRecord())
    monkeypatch.setattr(minimal, "default_planner", lambda force_field: other)
    with pytest.raises(RestartMismatch, match="allow_plan_change"):
        Simulation.resume(tmp_path)
    resumed = Simulation.resume(tmp_path, allow_plan_change=True)
    assert resumed.plan.revision == 1
    assert resumed.plan.predecessor.revision == 0
    assert resumed.plan.step == 2
    assert resumed.plan.neighbours.skin == 0.5
    assert resumed._engine.state.plan_revision == 1
    resumed.run(steps=2)
    resumed.save_restart(tmp_path / "again" / "restart.npz")
    again = json.loads((tmp_path / "again" / "simulation.json").read_text())
    assert again["plan_revision"] == 1


def test_resume_refuses_a_restart_without_identity(loaded, tmp_path):
    from fastmace.errors import RestartMismatch
    from fastmace.io import restart
    from fastmace.simulation import Simulation

    sim = _simulation(loaded)
    sim.save_restart(tmp_path / "restart.npz")
    state = sim._engine.state
    state.force_field_identity = None  # the main.py adapter writes no identity
    restart.write_restart(tmp_path / "restart.npz", state, engine="fastmace")
    with pytest.raises(RestartMismatch, match="identity"):
        Simulation.resume(tmp_path)


def test_resume_without_a_restart_file(tmp_path):
    from fastmace.simulation import Simulation

    with pytest.raises(FileNotFoundError, match="restart"):
        Simulation.resume(tmp_path)


# --------------------------------------------------------------------------- host writes (owner decision D4)


def test_set_positions_equals_a_fresh_simulation_at_those_positions(loaded):
    sim = _simulation(loaded, seed=6)
    sim.run(steps=3)
    moved = sim.committed().positions + 0.01
    momenta = sim.committed().momenta
    sim.set_positions(moved)
    committed = sim.committed()
    np.testing.assert_array_equal(committed.positions, moved)
    np.testing.assert_array_equal(committed.momenta, momenta)
    atoms = nacl(repeat=1)
    atoms.set_positions(moved)
    fresh = _simulation(loaded, atoms=atoms)
    np.testing.assert_array_equal(committed.forces, fresh.committed().forces)
    assert committed.energy == fresh.committed().energy
    assert committed.step == 3
    sim.run(steps=2)
    assert sim.committed().step == 5


def test_set_momenta(loaded):
    sim = _simulation(loaded, seed=6)
    sim.run(steps=2)
    momenta = np.full((8, 3), 0.01)
    sim.set_momenta(momenta)
    np.testing.assert_allclose(sim.committed().momenta, momenta, rtol=1e-14, atol=0)
    sim.run(steps=1)


@pytest.mark.parametrize("bad", [np.zeros((7, 3)), np.full((8, 3), np.nan)])
def test_host_writes_refuse_wrong_shapes_and_non_finite_values(loaded, bad):
    sim = _simulation(loaded)
    before = _positions(sim)
    with pytest.raises(ValueError):
        sim.set_positions(bad)
    with pytest.raises(ValueError):
        sim.set_momenta(bad)
    np.testing.assert_array_equal(_positions(sim), before)
