"""``fastmace.ase.LBFGS`` and ``FIRE``: ASE relaxation scripts on the graph engine (plan task 4-T10; SPEC §7.7,
decision 48).

The tiny MH-1-like model of ``test_events`` on rattled NaCl on the CPU. Each class is checked for its argument
list against ASE 3.27 (``inspect.signature``), its record from ASE's arguments, the refusals with their
reasons, ``FrechetCellFilter`` read into ``fm.optimize.FrechetCell`` and every other filter refused, and the
§7.7 acceptance: the plan of the equivalent ``Simulation``, the same relaxation, ASE's log and ``trajectory=``
frame for frame, observers at ASE's steps, and an observer change refused. Against ASE's own optimiser on
MACE's calculator in float64 for ``HORIZON`` steps (the random model's landscape is chaotic later; 4-T05).
"""

from __future__ import annotations

import inspect
import re

import numpy as np
import pytest
import test_events
from ase import units
from ase.filters import ExpCellFilter, FrechetCellFilter, StrainFilter, UnitCellFilter
from ase.parallel import world

from _support.structures import nacl
from fastmace import optimize
from fastmace.errors import PlanRefused

loaded, cpu_planner = test_events.loaded, test_events.cpu_planner
pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]
HORIZON = 12
NAMES = ("LBFGS", "FIRE")


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def structure(seed=3):
    return nacl(repeat=1, rattle=0.12, seed=seed)


def strained(seed=3):
    atoms = structure(seed)
    atoms.set_cell(atoms.cell.array @ np.array([[1.03, 0.01, 0.0], [0.0, 0.99, 0.0], [0.0, 0.0, 1.02]]),
                   scale_atoms=True)
    return atoms


def _ours(name, atoms, loaded, **options):
    import fastmace.ase

    options.setdefault("logfile", None)
    return getattr(fastmace.ase, name)(atoms, model=loaded, **options)


def _simulation(loaded, record, atoms):
    from fastmace.simulation import Simulation

    return Simulation(atoms, loaded, optimizer=record, seed=0)


def _ase_class(name):
    import ase.optimize

    return getattr(ase.optimize, name)


# --------------------------------------------------------------------------- arguments


@pytest.mark.parametrize("name", NAMES)
def test_every_ase_argument_is_supported_or_refused(name):
    from fastmace.ase import md
    from fastmace.ase import optimize as shims

    ours = getattr(shims, name)
    assert set(ours.SUPPORTED) | set(ours.REFUSED) == set(md.ASE_OPTIMIZER_ARGUMENTS[name])
    assert not set(ours.SUPPORTED) & set(ours.REFUSED)
    parameters = inspect.signature(ours.__init__).parameters
    for argument in ours.SUPPORTED:
        assert argument in parameters, argument
    for argument in shims.ADDITIONS:
        assert parameters[argument].kind is inspect.Parameter.KEYWORD_ONLY, argument
    theirs = [p for p in inspect.signature(_ase_class(name).__init__).parameters.values()
              if p.kind is p.POSITIONAL_OR_KEYWORD]
    mine = [p for p in parameters.values() if p.kind is p.POSITIONAL_OR_KEYWORD]
    assert [p.name for p in mine] == [p.name for p in theirs]
    assert [p.default for p in mine] == [p.default for p in theirs]
    assert set(ours.REFUSED) == {"master", "comm"} | ({"use_line_search"} if name == "LBFGS"
                                                       else {"maxmove", "position_reset_callback"})


def test_lbfgs_record_from_ases_arguments(loaded):
    assert _ours("LBFGS", structure(), loaded).record == optimize.LBFGS()
    dyn = _ours("LBFGS", structure(), loaded, maxstep=0.1, memory=20, damping=0.5, alpha=50.0)
    assert dyn.record == optimize.LBFGS(maxstep_A=0.1, memory=20, damping=0.5, alpha=50.0)


def test_fire_record_from_ases_arguments(loaded):
    assert _ours("FIRE", structure(), loaded).record == optimize.FIRE()
    dyn = _ours("FIRE", structure(), loaded, dt=0.05, maxstep=0.1, dtmax=0.5, Nmin=3, finc=1.2, fdec=0.4,
                astart=0.2, fa=0.98, a=0.2, downhill_check=True)
    assert dyn.record == optimize.FIRE(dt=0.05, maxstep_A=0.1, dtmax=0.5, Nmin=3, finc=1.2, fdec=0.4, astart=0.2,
                                       fa=0.98, a=0.2, downhill_check=True)


@pytest.mark.parametrize("name, argument, value, reason", [
    ("LBFGS", "restart", "opt.json", "restart"), ("FIRE", "restart", "opt.json", "restart"),
    ("LBFGS", "use_line_search", True, "line search"), ("FIRE", "maxmove", 0.1, "maxstep="),
    ("FIRE", "position_reset_callback", lambda *a: None, "callback"),
    ("LBFGS", "comm", object(), "comm"), ("FIRE", "master", True, "master")])
def test_refused_arguments_name_their_reason(loaded, name, argument, value, reason):
    with pytest.raises(PlanRefused, match=reason):
        _ours(name, structure(), loaded, **{argument: value})


@pytest.mark.parametrize("name", NAMES)
def test_ase_defaults_of_refused_arguments_are_accepted(loaded, name):
    assert _ours(name, structure(), loaded, restart=None, comm=world, master=None).simulation is None
    with pytest.raises(TypeError, match="seed"):
        _ours(name, structure(), loaded, seed=1)  # a relaxation has no noise


def test_frechet_cell_filter_becomes_the_device_filter(loaded):
    atoms = strained()
    dyn = _ours("LBFGS", FrechetCellFilter(atoms), loaded)
    assert dyn.record == optimize.LBFGS(cell_filter=optimize.FrechetCell())
    assert dyn.atoms.atoms is atoms
    voigt = (1, 1, 1, 0, 0, 0)
    dyn = _ours("FIRE", FrechetCellFilter(strained(), scalar_pressure=0.1 * units.GPa, hydrostatic_strain=True,
                                          mask=voigt, exp_cell_factor=3.0), loaded)
    cell = dyn.record.cell_filter
    assert cell.scalar_pressure_GPa == pytest.approx(0.1, rel=1e-14)
    assert cell.hydrostatic_strain and not cell.constant_volume and cell.exp_cell_factor == 3.0
    assert cell.mask == ((1, 0, 0), (0, 1, 0), (0, 0, 1))


@pytest.mark.parametrize("make", [UnitCellFilter, ExpCellFilter, StrainFilter])
def test_other_filters_are_refused_with_the_reason(loaded, make):
    with pytest.raises(PlanRefused, match="FrechetCellFilter") as raised:
        _ours("LBFGS", make(strained()), loaded)
    assert "ase-cell-filter" in raised.value.codes


@pytest.mark.parametrize("name", NAMES)
def test_todict_as_ase_3_27(loaded, name):
    """ASE 3.27's ``Optimizer.todict`` (``restart`` as a string; ``maxstep``, ``alpha``, ``max_steps`` and
    ``fmax`` where the optimiser has them). Written out: ASE 3.29, in the CPU environment, changed it."""
    dyn = _ours(name, structure(), loaded)
    expected = {"type": "optimization", "optimizer": name, "restart": "None", "maxstep": 0.2,
                "max_steps": 0}  # BaseDynamics sets max_steps = 0
    if name == "LBFGS":
        expected["alpha"] = 70.0
    assert dyn.todict() == expected
    dyn.run(fmax=1e-9, steps=2)
    assert dyn.todict() == expected | {"max_steps": 2, "fmax": 1e-9}


# --------------------------------------------------------------------------- the relaxation


@pytest.mark.parametrize("name", NAMES)
def test_a_ported_script_gets_the_plan_and_relaxation_of_the_equivalent_simulation(loaded, name):
    atoms = structure()
    dyn = _ours(name, atoms, loaded)
    converged = dyn.run(fmax=0.05, steps=6)
    sim = _simulation(loaded, dyn.record, structure())
    relaxed, summary = sim.relax(fmax=0.05, steps=6)
    assert dyn.plan.tier == "graph" and dyn.plan == sim.plan
    assert converged is summary["converged"] and dyn.nsteps == summary["steps"]
    np.testing.assert_array_equal(atoms.positions, relaxed.positions)
    assert atoms.get_potential_energy() == sim.committed().energy


def test_frechet_relaxation_equals_the_simulations(loaded):
    atoms = strained()
    dyn = _ours("LBFGS", FrechetCellFilter(atoms, scalar_pressure=0.05 * units.GPa), loaded)
    dyn.run(fmax=0.01, steps=5)
    sim = _simulation(loaded, dyn.record, strained())
    relaxed, _ = sim.relax(fmax=0.01, steps=5)
    assert dyn.plan == sim.plan
    np.testing.assert_array_equal(atoms.cell.array, relaxed.cell.array)
    np.testing.assert_array_equal(atoms.positions, relaxed.positions)
    assert not np.array_equal(atoms.cell.array, strained().cell.array)


@pytest.mark.parametrize("name", NAMES)
def test_positions_follow_ase_on_maces_calculator(loaded, name):
    """float64: the shim against ASE's own optimiser on MACE's calculator, as 4-T05 and 4-T06 test the engine."""
    from mace.calculators import MACECalculator

    from fastmace.model.forcefield import ForceField

    reference = structure()
    reference.calc = MACECalculator(model_paths=str(loaded.path), device="cpu", default_dtype="float64")
    _ase_class(name)(reference, logfile=None).run(fmax=1e-9, steps=HORIZON)
    atoms = structure()
    import fastmace.ase

    dyn = getattr(fastmace.ase, name)(atoms, logfile=None, force_field=ForceField(loaded, atoms, precision="float64"))
    dyn.run(fmax=1e-9, steps=HORIZON)
    assert dyn.nsteps == HORIZON
    np.testing.assert_allclose(atoms.positions, reference.positions, rtol=0, atol=1e-10)


def test_the_log_is_ases_format(loaded, capsys):
    dyn = _ours("LBFGS", structure(), loaded, logfile="-")
    dyn.run(fmax=1e-9, steps=3)
    lines = capsys.readouterr().out.splitlines()
    assert re.fullmatch(r" {5}  Step     Time          Energy          fmax", lines[0])
    rows = [line for line in lines if line.startswith("LBFGS:")]
    assert len(rows) == 4
    energies = [e for summary in dyn.relaxations for e in summary["energies_eV"][1:]]
    for step, row in enumerate(rows):
        match = re.fullmatch(r"LBFGS:\s+(\d+) \d\d:\d\d:\d\d\s+(-?\d+\.\d{6})\s+(\d+\.\d{6})", row)
        assert match and int(match.group(1)) == step
    assert float(rows[-1].split()[3]) == pytest.approx(energies[-1], abs=1e-6)
    dyn.run(fmax=1e-9, steps=2)  # a second run continues the numbering, without a second header
    more = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert [int(line.split()[1]) for line in more] == [4, 5]


def test_the_log_file_appends_as_ases(loaded):
    _ours("FIRE", structure(), loaded, logfile="opt.log").run(fmax=1e-9, steps=2)
    text = open("opt.log").read().splitlines()
    assert len(text) == 4 and text[1].startswith("FIRE:")


@pytest.mark.parametrize("loginterval", [1, 2])
def test_trajectory_reads_back_frame_for_frame(loaded, loginterval):
    from ase.io import read

    dyn = _ours("LBFGS", structure(), loaded, trajectory="opt.traj", loginterval=loginterval)
    dyn.run(fmax=1e-9, steps=4)
    frames = read("opt.traj", index=":")
    sim = _simulation(loaded, dyn.record, structure())
    expected = [sim.committed()]
    for _ in range(4):
        sim.relax(fmax=1e-9, steps=1)
        expected.append(sim.committed())
    expected = expected[::loginterval]
    assert len(frames) == len(expected)
    for frame, committed in zip(frames, expected):
        np.testing.assert_array_equal(frame.positions, committed.positions)
        np.testing.assert_array_equal(frame.cell.array, committed.cell)
        assert frame.get_potential_energy() == committed.energy
        np.testing.assert_array_equal(frame.get_forces(), committed.forces)


def test_trajectory_with_a_filter_holds_the_atoms_and_their_cell(loaded):
    from ase.io import read

    atoms = strained()
    _ours("FIRE", FrechetCellFilter(atoms), loaded, trajectory="cell.traj").run(fmax=1e-9, steps=3)
    frames = read("cell.traj", index=":")
    assert len(frames) == 4 and len(frames[-1]) == len(atoms)
    np.testing.assert_array_equal(frames[-1].cell.array, atoms.cell.array)


def test_observers_are_called_at_ases_steps(loaded):
    from ase.calculators.lj import LennardJones

    def record(opt, seen):
        opt.attach(lambda: seen.append(opt.nsteps), interval=2)
        opt.attach(lambda: seen.append(-opt.nsteps), interval=-3)

    expected, got = [], []
    reference_atoms = structure()
    reference_atoms.calc = LennardJones()
    reference = _ase_class("LBFGS")(reference_atoms, logfile=None)
    record(reference, expected)
    reference.run(fmax=1e-12, steps=5)
    dyn = _ours("LBFGS", structure(), loaded)
    record(dyn, got)
    dyn.run(fmax=1e-12, steps=5)
    assert got == expected == [0, 2, -3, 4]


def test_observers_see_the_relaxed_state(loaded):
    atoms = structure()
    dyn = _ours("LBFGS", atoms, loaded)
    seen = []
    dyn.attach(lambda: seen.append((atoms.positions.copy(), atoms.get_potential_energy())), interval=2)
    dyn.run(fmax=1e-9, steps=2)
    sim = _simulation(loaded, dyn.record, structure())
    sim.relax(fmax=1e-9, steps=2)
    np.testing.assert_array_equal(seen[-1][0], sim.committed().positions)
    assert seen[-1][1] == sim.committed().energy


def test_an_observer_that_changes_atoms_is_refused(loaded):
    atoms = structure()
    dyn = _ours("FIRE", atoms, loaded)

    def move():
        if dyn.nsteps == 2:
            atoms.positions += 0.01

    dyn.attach(move, interval=2)
    with pytest.raises(PlanRefused, match="history") as raised:
        dyn.run(fmax=1e-9, steps=4)
    assert "observer-relaxation-change" in raised.value.codes


def test_irun_and_step_as_ase(loaded):
    dyn = _ours("LBFGS", structure(), loaded)
    flags = list(dyn.irun(fmax=1e-9, steps=3))
    assert flags == [False] * 4 and dyn.nsteps == 3
    dyn.step()
    assert dyn.nsteps == 4
    sim = _simulation(loaded, dyn.record, structure())
    sim.relax(fmax=1e-9, steps=4)
    np.testing.assert_array_equal(dyn.atoms.positions, sim.committed().positions)
    assert dyn.converged() is False


def test_a_converged_start_takes_no_step(loaded):
    dyn = _ours("LBFGS", structure(), loaded)
    assert dyn.run(fmax=1e3, steps=10) is True
    assert dyn.nsteps == 0 and dyn.summary["steps"] == 0
