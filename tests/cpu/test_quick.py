"""``fm.md`` and ``RunResult`` on the CPU (plan task 1b-T20; SPEC §7.7 level 2).

A tiny MH-1-class model, NaCl, the minimal planner on the CPU without compilation (as
``test_simulation.py``). ``fm.md`` builds a ``Simulation`` and nothing else, so its plan and gate
equal the equivalent ``Simulation``'s; its inferences lose to explicit arguments; every
``ValueError`` of §7.7 comes before anything is loaded; NVE runs (4-T10), D3 runs
(3b-T04: MACE plus D3, with the gate's reference MACE plus torch-dftd), and MC runs (on T09's
driver); ``summary`` is the run report (1b-T15) and ``metadata`` the ``metadata.json`` record.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace import integrators
from fastmace.config import EngineConfig
from fastmace.engine import constraints
from fastmace.errors import PlanRefused
from fastmace.io.output import Output
from fastmace.mc import Swap
from fastmace.planner import minimal
from fastmace.protocols import hold, ramp

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

NA_CL = (11, 17)
SWAP = Swap(pairs=[("Na", "Cl")], every=2, attempts=5)


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    pytest.importorskip("cuequivariance_torch")
    from fastmace.model import loading

    model = tiny_mace_model(list(NA_CL), **FAMILIES["mh1"])
    path = tmp_path_factory.mktemp("models") / "tiny-mh1-11-17.model"
    torch.save(model, path)
    loading._LICENCES_SHOWN.add(str(path))  # no licence warning noise
    return str(path)


@pytest.fixture(scope="module")
def loaded(model_path):
    from fastmace.model import loading

    return loading.load_pristine(model_path, trusted=True)


@pytest.fixture(autouse=True)
def cpu_planner(monkeypatch):
    """The planner of every Simulation: the CPU without compilation, a fresh process record per test."""
    record = constraints.ProcessRecord()
    planner = minimal.MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ={},
                                     process=record)
    monkeypatch.setattr(minimal, "default_planner", lambda force_field: planner)
    return planner


@pytest.fixture
def no_engines_yet():
    """No production engine built in this process yet, so the gate's references run in-process."""
    from fastmace.engine import base

    saved = (base._PRODUCTION_ENGINES.count, base._PRODUCTION_ENGINES.first)
    base._forget_production_engines()
    yield
    base._PRODUCTION_ENGINES.count, base._PRODUCTION_ENGINES.first = saved


def _atoms():
    return nacl(repeat=1, rattle=0.05, seed=3)


def _md(model, tmp_path, **options):
    from fastmace.quick import md

    options.setdefault("output", tmp_path / "run")
    options.setdefault("seed", 5)
    options.setdefault("steps", 3)
    if "schedule" not in options:
        options.setdefault("temperature_K", 300.0)
    structure = options.pop("structure", _atoms())
    if isinstance(model, str):
        options.setdefault("trusted", True)
    return md(structure, model=model, **options)


def _simulation(loaded, integrator, *, seed=5, check="skip", output=None):
    from fastmace.simulation import Simulation

    return Simulation(_atoms(), loaded, integrator=integrator, seed=seed, check=check, output=output)


def _gate_fields(gate):
    """The gate record without its timings."""
    record = gate.to_metadata()
    record.pop("seconds", None)
    return record


# --------------------------------------------------------------------------- equal to Simulation


def test_md_equals_the_equivalent_simulation(model_path, loaded, tmp_path):
    from fastmace import quick
    from fastmace.quick import RunResult

    run = _md(model_path, tmp_path)
    assert isinstance(run, RunResult)
    langevin = integrators.Langevin(timestep_fs=0.5, temperature_K=300.0,
                                    friction_per_fs=quick.default_friction_per_fs())
    sim = _simulation(loaded, langevin)
    assert run.plan == sim.plan
    assert run.gate == sim.gate
    assert run.gate.status == "skipped"
    assert run.simulation.integrator == langevin
    assert run.simulation.workload == sim.workload


def test_md_equals_simulation_with_the_gate(model_path, loaded, tmp_path, no_engines_yet):
    from fastmace import quick

    run = _md(model_path, tmp_path, check="gate")
    assert run.gate.status == "passed" and run.gate.check == "gate"
    langevin = integrators.Langevin(timestep_fs=0.5, temperature_K=300.0,
                                    friction_per_fs=quick.default_friction_per_fs())
    from fastmace.engine import base

    base._forget_production_engines()  # the same condition for both: references in-process
    sim = _simulation(loaded, langevin, check="gate")
    assert run.plan == sim.plan
    assert _gate_fields(run.gate) == _gate_fields(sim.gate)
    assert "gate:    passed" in run.summary


def test_npt_is_baoab(model_path, loaded, tmp_path):
    from fastmace import quick

    run = _md(model_path, tmp_path, ensemble="NPT", pressure_GPa=0.5)
    baoab = integrators.LangevinBAOAB(timestep_fs=0.5, temperature_K=300.0,
                                      friction_per_fs=quick.default_friction_per_fs(), pressure_GPa=0.5)
    assert run.simulation.integrator == baoab
    assert run.plan == _simulation(loaded, baoab).plan
    assert "NPT barostat: experimental" in run.summary


def test_friction_default_is_main_py(model_path, tmp_path):
    run = _md(model_path, tmp_path)
    # main.py's --friction 0.02 is per ASE time unit, about 0.001965/fs (conductor answer, T01 point 5).
    assert run.plan.integrator.friction == pytest.approx(0.02, rel=1e-15)
    assert run.simulation.integrator.friction_per_fs == pytest.approx(0.001965, rel=1e-3)


# --------------------------------------------------------------------------- RunResult


def test_run_result_is_the_report_and_the_metadata(model_path, tmp_path):
    run = _md(model_path, tmp_path, steps=4)
    sim = run.simulation
    assert run.summary == sim.report.text()
    assert "speed:" in run.summary and "pending" not in run.summary  # whole-run mean after the run
    assert run.output == str(tmp_path / "run")
    written = json.loads((tmp_path / "run" / "metadata.json").read_text())
    assert run.metadata == written
    assert written["report"]["text"] == run.summary
    current = json.loads(json.dumps(sim.metadata()))
    assert {key: value for key, value in written.items() if key != "created_utc"} == {
        key: value for key, value in current.items() if key != "created_utc"}
    assert int(sim.committed().step) == 4


def test_seed_is_drawn_and_recorded(model_path, tmp_path):
    run = _md(model_path, tmp_path, seed=None)
    seed = run.simulation.seed
    assert isinstance(seed, int) and seed >= 0
    assert json.dumps(run.metadata).count(str(seed)) >= 1


def test_output_default_for_atoms(model_path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = _md(model_path, tmp_path, output=None)
    assert Path(run.output).resolve() == (tmp_path / "fastmace-run").resolve()
    assert (tmp_path / "fastmace-run" / "metadata.json").is_file()


def test_output_default_for_a_file(model_path, tmp_path, monkeypatch):
    from ase.io import write

    monkeypatch.chdir(tmp_path)
    write(tmp_path / "NaCl.cif", _atoms())
    run = _md(model_path, tmp_path, output=None, structure=str(tmp_path / "NaCl.cif"))
    assert Path(run.output).resolve() == (tmp_path / "NaCl.fastmace").resolve()
    assert len(run.simulation.committed().positions) == 8


def test_explicit_output_record(model_path, tmp_path):
    output = Output(tmp_path / "mine", trajectory_every=None, thermo_every=2, restart_every=None)
    run = _md(model_path, tmp_path, output=output)
    assert run.simulation.output == output
    assert run.output == str(tmp_path / "mine")


def test_repeat(model_path, tmp_path):
    run = _md(model_path, tmp_path, repeat=(2, 1, 1))
    assert len(run.simulation.committed().positions) == 16
    assert "16 atoms (2×1×1)" in run.summary


def test_default_repeat_is_never_expanded(model_path, tmp_path):
    run = _md(model_path, tmp_path)
    assert len(run.simulation.committed().positions) == 8
    assert "(1×1×1)" in run.summary
    assert "small cell" in run.summary  # 5.64 Å < 2 × cutoff


# --------------------------------------------------------------------------- inference


def test_schedule_infers_max_temperature_and_start(model_path, tmp_path):
    schedule = ramp(100.0, 400.0, ps=0.002) + hold(400.0, ps=0.001)
    run = _md(model_path, tmp_path, schedule=schedule, steps=None)
    sim = run.simulation
    assert sim.workload.max_temperature_K == pytest.approx(400.0)
    assert sim.integrator.temperature_K == pytest.approx(100.0)
    assert sim.schedule.finished  # steps=None runs the whole schedule


def test_explicit_integrator_wins(model_path, tmp_path):
    langevin = integrators.Langevin(timestep_fs=1.0, temperature_K=350.0, friction_per_fs=0.01)
    run = _md(model_path, tmp_path, integrator=langevin, temperature_K=None)
    assert run.simulation.integrator == langevin
    assert run.simulation.workload.max_temperature_K == 350.0


def test_explicit_baoab_with_default_ensemble(model_path, tmp_path):
    baoab = integrators.LangevinBAOAB(temperature_K=300.0, friction_per_fs=0.01, pressure_GPa=0.2)
    run = _md(model_path, tmp_path, integrator=baoab, temperature_K=None)
    assert run.simulation.integrator == baoab


# --------------------------------------------------------------------------- ValueErrors of §7.7


@pytest.mark.parametrize("options,match", [
    ({"ensemble": "NPH"}, "ensemble"),
    ({"ensemble": "NPT", "integrator": integrators.Langevin(temperature_K=300.0, friction_per_fs=0.01),
      "temperature_K": None}, "ensemble"),
    ({"temperature_K": None}, "temperature_K"),
    ({"schedule": hold(300.0, ps=0.001)}, "temperature_K"),
    ({"steps": None}, "steps"),
    ({"repeat": (0, 1, 1)}, "repeat"),
    ({"repeat": (1, 1)}, "repeat"),
    ({"mc": SWAP, "temperature_K": 0.0}, "temperature_K=0"),
    ({"mc": SWAP, "temperature_K": None,
      "schedule": hold(300.0, ps=0.001), "steps": None}, "min_mc_temperature_K"),
    ({"mc": SWAP, "min_mc_temperature_K": 0.0},
     "min_mc_temperature_K"),
    ({"check": "always"}, "check"),
    ({"dispersion": "d3"}, "d3bj"),
])
def test_value_errors_before_anything_is_loaded(tmp_path, options, match):
    from fastmace.quick import md

    options = {"output": tmp_path / "run", "seed": 1, "steps": 3, "temperature_K": 300.0, **options}
    # A path that does not exist: every check runs before the model is loaded.
    with pytest.raises(ValueError, match=match):
        md(_atoms(), model=str(tmp_path / "absent.model"), trusted=True, **options)
    assert not (tmp_path / "run").exists()


def test_integrator_conflicting_temperature(tmp_path):
    from fastmace.quick import md

    langevin = integrators.Langevin(temperature_K=350.0, friction_per_fs=0.01)
    with pytest.raises(ValueError, match="temperature_K"):
        md(_atoms(), model=str(tmp_path / "absent.model"), trusted=True, integrator=langevin, temperature_K=300.0,
           steps=3, seed=1)


def test_multi_head_label_without_head(tmp_path):
    from fastmace.quick import md

    with pytest.raises(ValueError, match="matpes_r2scan"):
        md(_atoms(), model="mh1", temperature_K=300.0, steps=3, seed=1, output=tmp_path / "run")


# --------------------------------------------------------------------------- NVE (4-T10)


def _moving():
    """NaCl with momenta at about 300 K (fm.md draws no velocities)."""
    import numpy as np
    from ase import units

    atoms = _atoms()
    scale = np.sqrt(atoms.get_masses() * units.kB * 300)[:, None]
    atoms.set_momenta(np.random.default_rng(5).standard_normal((len(atoms), 3)) * scale)
    return atoms


def test_nve_runs_velocity_verlet_from_the_structures_momenta(model_path, loaded, tmp_path):
    from fastmace import quick
    from fastmace.planner.workload import Workload
    from fastmace.simulation import Simulation

    atoms = _moving()
    run = _md(model_path, tmp_path, ensemble="NVE", temperature_K=None, structure=atoms)
    nve = integrators.NVE(timestep_fs=0.5)
    assert run.simulation.integrator == nve
    highest = quick.nve_max_temperature_K(atoms)
    assert highest == pytest.approx(2 * atoms.get_temperature())
    assert run.simulation.workload == Workload(max_temperature_K=highest)
    def recorded(record):  # every max_temperature_K value in metadata.json
        if isinstance(record, dict):
            return [v for k, v in record.items() if k == "max_temperature_K"] + [
                x for v in record.values() for x in recorded(v)]
        return [x for v in record for x in recorded(v)] if isinstance(record, list) else []

    assert recorded(run.metadata) and set(recorded(run.metadata)) == {highest}
    sim = Simulation(atoms, loaded, integrator=nve, seed=5, workload=Workload(max_temperature_K=highest))
    assert run.plan == sim.plan
    assert "NVE VelocityVerlet 0.5 fs" in run.summary.splitlines()[0]


def test_nve_from_rest_has_the_floor(model_path, tmp_path):
    from fastmace import quick

    run = _md(model_path, tmp_path, ensemble="NVE", temperature_K=None)
    assert run.simulation.workload.max_temperature_K == quick.NVE_MIN_TEMPERATURE_K == 1.0


def test_explicit_nve_integrator(model_path, tmp_path):
    nve = integrators.NVE(timestep_fs=1.0, fixcm=False, dof="ase")
    run = _md(model_path, tmp_path, integrator=nve, temperature_K=None, structure=_moving())
    assert run.simulation.integrator == nve
    assert run.plan.integrator.ensemble == "nve"


@pytest.mark.parametrize("options, match", [
    ({"temperature_K": 300.0}, "temperature_K"),
    ({"temperature_K": None, "schedule": hold(300.0, ps=0.001), "steps": None}, "schedule"),
    ({"temperature_K": None, "mc": SWAP}, "mc"),
    ({"temperature_K": None, "pressure_GPa": 0.1}, "pressure_GPa"),
])
def test_nve_value_errors_before_anything_is_loaded(tmp_path, options, match):
    from fastmace.quick import md

    options = {"output": tmp_path / "run", "seed": 1, "steps": 3, **options}
    with pytest.raises(ValueError, match=match):
        md(_atoms(), model=str(tmp_path / "absent.model"), trusted=True, ensemble="NVE", **options)
    assert not (tmp_path / "run").exists()


def test_every_integrator_record_is_accepted(tmp_path):
    """integrator= takes every fm.integrators record (4-T10); anything else is a TypeError naming them."""
    from fastmace.quick import md

    with pytest.raises(TypeError, match="NoseHooverChain"):
        md(_atoms(), model=str(tmp_path / "absent.model"), trusted=True, integrator=object(), steps=3, seed=1)
    nhc = integrators.NoseHooverChain(timestep_fs=0.5, temperature_K=300.0, tdamp_fs=50.0)
    with pytest.raises(ValueError, match="ensemble"):
        md(_atoms(), model=str(tmp_path / "absent.model"), trusted=True, integrator=nhc, ensemble="NPT", steps=3,
           seed=1)


def _r2scan(loaded):
    import dataclasses

    return dataclasses.replace(loaded, functional="r2scan")


def test_d3bj_runs_and_is_labelled(loaded, tmp_path):
    from fastmace import D3
    from fastmace.model.dispersed import DispersedForceField

    run = _md(_r2scan(loaded), tmp_path, dispersion="d3bj")
    assert run.plan.dispersion == D3(functional="r2scan")
    assert isinstance(run.simulation._engine.force_field, DispersedForceField)
    assert "physics: D3(BJ) functional r2scan" in run.summary and "non-stock" in run.summary
    assert "D3: experimental" in run.summary
    assert run.metadata["force_field_identity"]["physics"]["dispersion"]["functional"] == "r2scan"
    assert "functional=r2scan" in run.metadata["force_field"]["dispersion"]
    assert run.metadata["force_field"]["d3"]["builds"] >= 1


def test_d3_gate_compares_against_mace_plus_torch_dftd(loaded, tmp_path, no_engines_yet):
    pytest.importorskip("torch_dftd")
    run = _md(_r2scan(loaded), tmp_path, dispersion="d3bj", check="gate")
    assert run.gate.status == "passed"
    assert "non-stock (MACE plus torch-dftd" in json.dumps(run.gate.references)  # the reference's info()


def test_d3bj_refused_for_mace_off(tmp_path):
    from fastmace.quick import md

    with pytest.raises(ValueError, match="twice") as raised:
        md(_atoms(), model="off23-small", dispersion="d3bj", temperature_K=300.0, steps=3, seed=1)
    assert not isinstance(raised.value, PlanRefused)


def test_d3bj_refused_for_a_path(tmp_path):
    from fastmace.quick import md

    with pytest.raises(ValueError, match="unknown"):
        md(_atoms(), model=str(tmp_path / "absent.model"), trusted=True, dispersion="d3bj", temperature_K=300.0,
           steps=3, seed=1)


def test_explicit_d3_with_another_functional_names_the_warning(loaded, tmp_path):
    from fastmace import D3

    run = _md(_r2scan(loaded), tmp_path, dispersion=D3(functional="pbe"))
    assert "differs from the training functional" in run.summary
    assert run.plan.dispersion == D3(functional="pbe")


@pytest.fixture
def drivers(monkeypatch):
    """T09's MCDriver, observed (tests/_support/fake_mc.py's spy); the drivers made, in order."""
    from _support import fake_mc

    return fake_mc.spy(monkeypatch)


def _swap(every=2, attempts=5):
    from fastmace import mc

    return mc.Swap(pairs=[("Na", "Cl")], every=every, attempts=attempts)


def test_mc_runs_at_a_constant_temperature(model_path, tmp_path, drivers):
    run = _md(model_path, tmp_path, mc=_swap(), steps=4)
    (driver,) = drivers
    assert [step for step, _ in driver.spied] == [2, 4]
    workload = run.simulation.workload
    assert workload.mc_moves == ("swap",) and workload.species_changes
    assert workload.min_mc_temperature_K == pytest.approx(300.0)
    assert run.metadata["counters"]["mc"]["swap"]["attempted"] == 10
    assert "swap Na-Cl every 2, 5 attempts" in run.summary
    assert run.metadata["report"]["mc"]["ranges"] == [[0, 4]]


def test_mc_with_a_schedule_attaches_only_at_or_above_the_minimum(model_path, tmp_path, drivers):
    # 0.5 fs steps: the ramp targets 175, 250, 325, 400 K, then 400, 400 K; blocks every step.
    schedule = ramp(100.0, 400.0, ps=0.002) + hold(400.0, ps=0.001)
    run = _md(model_path, tmp_path, schedule=schedule, steps=None, mc=_swap(every=1, attempts=1),
              min_mc_temperature_K=250.0)
    assert run.simulation.schedule.finished
    assert [step for driver in drivers for step, _ in driver.spied] == [2, 3, 4, 5, 6]
    assert run.metadata["report"]["mc"]["ranges"] == [[1, 6]]  # steps 2-6 run with MC: attached from step 1
    assert run.simulation.workload.min_mc_temperature_K == 250.0


def test_mc_ranges_split_a_schedule(model_path):
    from fastmace.protocols import ScheduleState
    from fastmace.quick import mc_ranges

    schedule = ramp(300.0, 200.0, ps=0.002) + ramp(200.0, 300.0, ps=0.002)  # 275 250 225 200 225 250 275 300
    position = ScheduleState(schedule, 0.5)
    assert [position.resolved.target(k) for k in range(1, 9)] == pytest.approx([275, 250, 225, 200, 225, 250, 275,
                                                                                300])
    assert mc_ranges(position, 8, 250.0) == [(2, True), (3, False), (3, True)]
    assert mc_ranges(position, 8, 100.0) == [(8, True)]
    assert mc_ranges(position, 8, 400.0) == [(8, False)]


def test_mc_species_join_the_superset(model_path, tmp_path, drivers, monkeypatch):
    from fastmace import mc, quick

    assert quick.mc_species(mc.Transmute(species=["Na", "K"], every=1, attempts=1)) == {11, 19}
    assert quick.mc_species([_swap(), mc.Swap(pairs=[(11, "Br")], every=1, attempts=1)]) == {11, 17, 35}
    run = _md(model_path, tmp_path, mc=_swap(), steps=2)
    assert set(run.simulation.force_field.species) == {11, 17}


def test_mc_with_a_schedule_needs_min_mc_temperature(tmp_path):
    from fastmace.quick import md

    with pytest.raises(ValueError, match="min_mc_temperature_K"):
        md(_atoms(), model=str(tmp_path / "absent.model"), trusted=True, mc=_swap(),
           schedule=hold(300.0, ps=0.001), seed=1)


def test_mc_is_no_longer_refused(tmp_path):
    from fastmace.quick import md

    with pytest.raises(Exception) as raised:  # the absent model, not a phase refusal
        md(_atoms(), model=str(tmp_path / "absent.model"), trusted=True, mc=_swap(), temperature_K=300.0, steps=3,
           seed=1)
    assert not isinstance(raised.value, PlanRefused)


def test_no_refusal_is_left_in_fm_md(tmp_path):
    """NVE (4-T10), D3 (3b-T04) and MC (3-T15) all run: no PlanRefused before the model is loaded."""
    from fastmace.quick import md

    with pytest.raises(Exception) as raised:
        md(_atoms(), model=str(tmp_path / "absent.model"), trusted=True, ensemble="NVE", steps=3, seed=1)
    assert not isinstance(raised.value, PlanRefused)


def test_displace_declares_its_largest_delta(tmp_path):
    """fm.md sets Workload.max_displacement_A from the Displace moves, so mc=Displace(...) plans."""
    from fastmace import mc, quick

    moves = [mc.Displace(delta_A=0.1, every=2, attempts=3), mc.Displace(delta_A=0.25, every=5, attempts=1), _swap()]
    assert quick.max_displacement_A(moves) == 0.25
    assert quick.max_displacement_A([_swap()]) is None
    with pytest.raises(Exception) as raised:  # the absent model, not the workload
        quick.md(_atoms(), model=str(tmp_path / "absent.model"), trusted=True, mc=moves[0], temperature_K=300.0,
                 steps=3, seed=1)
    assert "max_displacement_A" not in str(raised.value)



# --------------------------------------------------------------------------- initial_velocities (owner, 28 Oct 2026)


def test_initial_velocities_values():
    from fastmace import quick

    assert quick._initial_velocities("keep") == "keep"
    assert quick._initial_velocities("maxwell_boltzmann") == "maxwell_boltzmann"
    assert quick._initial_velocities(10) == 10.0
    for bad in ("mb", 0.0, -5.0, float("nan"), True, None):
        with pytest.raises(ValueError, match="initial_velocities"):
            quick._initial_velocities(bad)


def test_the_draw_is_seeded_and_only_for_a_structure_at_rest():
    import numpy as np

    from fastmace import quick

    first, second, other = _atoms(), _atoms(), _atoms()
    assert quick.draw_initial_velocities(first, 300.0, 7)
    assert quick.draw_initial_velocities(second, 300.0, 7)
    assert quick.draw_initial_velocities(other, 300.0, 8)
    assert np.array_equal(first.get_momenta(), second.get_momenta())
    assert not np.array_equal(first.get_momenta(), other.get_momenta())
    moving = first.get_momenta().copy()
    assert not quick.draw_initial_velocities(first, 1000.0, 9)  # already moves: kept
    assert np.array_equal(first.get_momenta(), moving)


def test_maxwell_boltzmann_needs_a_starting_target_above_zero(tmp_path):
    """Refused before the model is loaded (the model path does not exist)."""
    absent = str(tmp_path / "absent.model")
    with pytest.raises(ValueError, match="starting target, which is 0 K"):
        _md(absent, tmp_path, schedule=ramp(0, 300, ps=0.01), initial_velocities="maxwell_boltzmann")
    with pytest.raises(ValueError, match="NVE has none"):
        _md(absent, tmp_path, ensemble="NVE", temperature_K=None, initial_velocities="maxwell_boltzmann")


def test_csvr_from_rest_starts_with_initial_velocities(loaded, tmp_path):
    """A CSVR ramp from 0 K on a structure at rest: refused by default, runs with initial_velocities=10 K,
    and the report and metadata say what was drawn."""
    from fastmace.errors import InputError

    csvr = integrators.CSVR(timestep_fs=0.5, temperature_K=0.0, tau_fs=10.0)
    with pytest.raises(InputError, match="kinetic energy is zero"):
        _md(loaded, tmp_path, integrator=csvr, schedule=ramp(0, 300, ps=0.002), output=tmp_path / "keep")
    run = _md(loaded, tmp_path, integrator=csvr, schedule=ramp(0, 300, ps=0.002), initial_velocities=10.0,
              output=tmp_path / "drawn")
    note = "starting velocities: Maxwell-Boltzmann at 10 K, drawn from seed 5"
    assert note in run.summary
    assert note in json.dumps(run.metadata)


def test_initial_velocities_keeps_a_moving_structure(loaded, tmp_path):
    from fastmace import quick

    atoms = _atoms()
    quick.draw_initial_velocities(atoms, 300.0, 1)
    run = _md(loaded, tmp_path, structure=atoms, initial_velocities="maxwell_boltzmann")
    assert "starting velocities: the structure's own" in run.summary


def test_fastmace_run_takes_initial_velocities():
    """fastmace run passes every fm.md argument; fastmace md has no flag for it (SPEC §7.5's flag set)."""
    from fastmace.cli import run as run_command

    assert "initial_velocities" in run_command._md_parameters()
