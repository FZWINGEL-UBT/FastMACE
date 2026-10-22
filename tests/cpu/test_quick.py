"""``fm.md`` and ``RunResult`` on the CPU (plan task 1b-T20; SPEC §7.7 level 2).

A tiny MH-1-class model, NaCl, the minimal planner on the CPU without compilation (as
``test_simulation.py``). ``fm.md`` builds a ``Simulation`` and nothing else, so its plan and gate
equal the equivalent ``Simulation``'s; its inferences lose to explicit arguments; every
``ValueError`` of §7.7 comes before anything is loaded; NVE and D3 are refused with their
phase, and MC runs (on T09's driver); ``summary`` is the run report (1b-T15) and ``metadata``
the ``metadata.json`` record.
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


# --------------------------------------------------------------------------- refusals with their phase


def test_nve_is_phase_4(tmp_path):
    from fastmace.quick import md

    with pytest.raises(PlanRefused, match="phase 4") as raised:
        md(_atoms(), model=str(tmp_path / "absent.model"), trusted=True, ensemble="NVE", temperature_K=300.0,
           steps=3, seed=1)
    assert "nve-phase-4" in raised.value.codes


def test_d3bj_is_expanded_then_refused_phase_3b(tmp_path):
    from fastmace.quick import md

    # A label whose checkpoint need not be present: the refusal comes before loading.
    with pytest.raises(PlanRefused, match="phase 3b") as raised:
        md(_atoms(), model="mh1-r2scan", dispersion="d3bj", temperature_K=300.0, steps=3, seed=1,
           output=tmp_path / "run")
    assert "d3-phase-3b" in raised.value.codes
    assert "functional=r2scan" in str(raised.value) and "pair_cutoff_bohr=40" in str(raised.value)


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


def test_explicit_d3_with_another_functional_names_the_warning(tmp_path):
    from fastmace import D3
    from fastmace.quick import md

    with pytest.raises(PlanRefused, match="phase 3b") as raised:
        md(_atoms(), model="mh1-r2scan", dispersion=D3(functional="pbe"), temperature_K=300.0, steps=3, seed=1)
    assert "differs from the training functional" in str(raised.value)


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


def test_every_refusal_at_once(tmp_path):
    from fastmace.quick import md

    with pytest.raises(PlanRefused) as raised:
        md(_atoms(), model="mh1-r2scan", ensemble="NVE", dispersion="d3bj", mc=_swap(), temperature_K=300.0, steps=3,
           seed=1)
    assert set(raised.value.codes) == {"nve-phase-4", "d3-phase-3b"}
