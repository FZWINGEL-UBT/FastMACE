"""The phase 4 ``fastmace.ase`` classes: SPEC §7.7's acceptance for the new integrators (plan task 4-T10).

Tiny MH-1-family model on the CPU without compilation, as ``test_ase_md.py``. Each class is checked for
its argument list against ASE (``inspect.signature``), its record from ASE's units, and the §7.7
acceptance: the plan of the equivalent ``Simulation``, the same run, the ``trajectory=`` file frame for
frame, and observer changes equal to ``Simulation`` events. The integrators' physics against ASE is
4-T02's and 4-T03's.
"""

from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest
import torch
from ase import units
from ase.parallel import world

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace import integrators
from fastmace.config import EngineConfig
from fastmace.engine import constraints
from fastmace.errors import PlanRefused
from fastmace.planner import minimal

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

NA_CL = (11, 17)
DT = 1.0 * units.fs
TDAMP, PDAMP = 50 * units.fs, 500 * units.fs
PRESSURE_GPA = 0.1


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
    """The planner of every Simulation here: the CPU without compilation, a fresh process record per test."""
    record = constraints.ProcessRecord()
    planner = minimal.MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ={},
                                     process=record)
    monkeypatch.setattr(minimal, "default_planner", lambda force_field: planner)
    return planner


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _atoms():
    atoms = nacl(repeat=1, rattle=0.05, seed=3)  # with momenta at about 300 K: NVE and the barostats move
    scale = np.sqrt(atoms.get_masses() * units.kB * 300)[:, None]
    atoms.set_momenta(np.random.default_rng(5).standard_normal((len(atoms), 3)) * scale)
    return atoms


_NPT = dict(temperature_K=300, pressure_au=PRESSURE_GPA * units.GPa, taut=TDAMP, taup=PDAMP,
            compressibility_au=0.05 / units.GPa)
_NPT_RECORD = dict(timestep_fs=1.0, temperature_K=300.0, taut_fs=50.0, pressure_GPa=PRESSURE_GPA, taup_fs=500.0,
                   compressibility_per_GPa=0.05)

# Each class: its ASE arguments for a test run (ASE units), and the record they must give.
CASES = {
    "VelocityVerlet": ({}, lambda **dof: integrators.NVE(timestep_fs=1.0, **dof)),
    "Bussi": (
        dict(temperature_K=300, taut=TDAMP),
        lambda **dof: integrators.CSVR(timestep_fs=1.0, temperature_K=300.0, tau_fs=50.0, **dof)),
    "NoseHooverChainNVT": (
        dict(temperature_K=300, tdamp=TDAMP),
        lambda **dof: integrators.NoseHooverChain(timestep_fs=1.0, temperature_K=300.0, tdamp_fs=50.0, **dof)),
    "IsotropicMTKNPT": (
        dict(temperature_K=300, pressure_au=PRESSURE_GPA * units.GPa, tdamp=TDAMP, pdamp=PDAMP),
        lambda **dof: integrators.MTK(timestep_fs=1.0, temperature_K=300.0, pressure_GPa=PRESSURE_GPA,
                                      tdamp_fs=50.0, pdamp_fs=500.0, cell_mode="hydrostatic", **dof)),
    "MTKNPT": (
        dict(temperature_K=300, pressure_au=PRESSURE_GPA * units.GPa, tdamp=TDAMP, pdamp=PDAMP, tchain=2, pchain=4,
             tloop=2, ploop=3),
        lambda **dof: integrators.MTK(timestep_fs=1.0, temperature_K=300.0, pressure_GPa=PRESSURE_GPA,
                                      tdamp_fs=50.0, pdamp_fs=500.0, tchain=2, pchain=4, tloop=2, ploop=3,
                                      cell_mode="full", **dof)),
    "NVTBerendsen": (
        dict(temperature_K=300, taut=TDAMP),
        lambda **dof: integrators.Berendsen(timestep_fs=1.0, temperature_K=300.0, taut_fs=50.0, **dof)),
    "NPTBerendsen": (_NPT, lambda **dof: integrators.Berendsen(**_NPT_RECORD, **dof)),
    "Inhomogeneous_NPTBerendsen": (
        dict(_NPT, mask=(1, 0, 1)), lambda **dof: integrators.Berendsen(**_NPT_RECORD, mask=(1, 0, 1), **dof)),
}
NAMES = tuple(CASES)
# The classes whose ASE original has no fixcm: dof="ase" by default (module docstring of fastmace.ase.md).
WITHOUT_FIXCM = ("VelocityVerlet", "Bussi", "NoseHooverChainNVT", "IsotropicMTKNPT", "MTKNPT")
WITH_FIXCM = tuple(name for name in NAMES if name not in WITHOUT_FIXCM)
ASE_DOF = {"fixcm": False, "dof": "ase"}


def _dyn(loaded, name, atoms=None, **options):
    import fastmace.ase

    arguments, _ = CASES[name]
    options.setdefault("seed", 1)
    atoms = _atoms() if atoms is None else atoms
    return getattr(fastmace.ase, name)(atoms, DT, **arguments, model=loaded, **options)


def _records_close(ours, theirs):
    """Record equality up to the ulps of the unit conversions (ASE units -> record)."""
    import dataclasses

    assert type(ours) is type(theirs)
    for field in dataclasses.fields(ours):
        a, b = getattr(ours, field.name), getattr(theirs, field.name)
        if isinstance(a, float):
            assert a == pytest.approx(b, rel=1e-14, abs=1e-300), field.name
        else:
            assert a == b, field.name


def _simulation(loaded, record, atoms=None, seed=1, output=None, host_callbacks_every=None):
    from fastmace import quick
    from fastmace.model.forcefield import ForceField
    from fastmace.planner.workload import Workload
    from fastmace.simulation import Simulation

    atoms = _atoms() if atoms is None else atoms
    temperature = getattr(record, "temperature_K", None)
    temperature = quick.nve_max_temperature_K(atoms) if temperature is None else float(temperature)
    workload = Workload(max_temperature_K=temperature, host_callbacks_every=host_callbacks_every)
    return Simulation(atoms, ForceField(loaded, atoms, precision="float32"), integrator=record, seed=seed,
                      output=output, workload=workload)


# --------------------------------------------------------------------------- arguments


def _ase_class(name):
    from fastmace.ase import md

    return getattr(importlib.import_module(md.ASE_MODULES[name]), name)


@pytest.mark.parametrize("name", NAMES)
def test_every_ase_argument_is_supported_or_refused(name):
    from fastmace.ase import md

    ours = getattr(md, name)
    assert set(ours.SUPPORTED) | set(ours.REFUSED) == set(md.ASE_ARGUMENTS[name])
    assert not set(ours.SUPPORTED) & set(ours.REFUSED)
    parameters = {}
    for klass in ours.__mro__:  # Inhomogeneous_NPTBerendsen passes **kwargs on, as ASE's
        if klass is md._Dynamics:
            break
        for argument, parameter in inspect.signature(klass.__init__).parameters.items():
            parameters.setdefault(argument, parameter)
    for argument in ours.SUPPORTED:
        assert argument in parameters, argument
    for argument in (*md.ADDITIONS, *(("dof",) if name in WITHOUT_FIXCM else ())):
        assert parameters[argument].kind is inspect.Parameter.KEYWORD_ONLY, argument
    # ASE's positional arguments in ASE's order, with ASE's defaults.
    theirs = [p for p in inspect.signature(_ase_class(name).__init__).parameters.values()
              if p.kind is p.POSITIONAL_OR_KEYWORD]
    mine = [p for p in inspect.signature(ours.__init__).parameters.values() if p.kind is p.POSITIONAL_OR_KEYWORD]
    assert [p.name for p in mine] == [p.name for p in theirs]
    assert [p.default for p in mine] == [p.default for p in theirs]


@pytest.mark.parametrize("name", NAMES)
def test_the_class_is_mapped(name):
    from fastmace.ase import md

    assert name in md.MAPPED and not hasattr(md, "LATER")


@pytest.mark.parametrize("name", NAMES)
def test_record_from_ase_units(loaded, name):
    expected = CASES[name][1](**ASE_DOF) if name in WITHOUT_FIXCM else CASES[name][1](fixcm=True)  # ASE's default
    _records_close(_dyn(loaded, name).integrator, expected)


@pytest.mark.parametrize("name", WITH_FIXCM)
def test_fixcm_passes_through(loaded, name):
    assert _dyn(loaded, name, fixcm=False).integrator.fixcm is False
    with pytest.raises(TypeError, match="dof"):
        _dyn(loaded, name, dof="ase")  # not an argument of these classes: fixcm is ASE's


@pytest.mark.parametrize("name", WITHOUT_FIXCM)
def test_dof_auto_is_fastmaces_convention(loaded, name):
    fixcm = "auto" if name in ("VelocityVerlet", "Bussi") else True  # NHC and MTK take no FixAtoms
    _records_close(_dyn(loaded, name, dof="auto").integrator, CASES[name][1](fixcm=fixcm, dof="auto"))
    with pytest.raises(ValueError, match="dof"):
        _dyn(loaded, name, dof="3n")


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("argument, value", [("comm", object()), ("communicator", object()), ("master", True)])
def test_refused_arguments_name_themselves(loaded, name, argument, value):
    with pytest.raises(PlanRefused, match=argument):
        _dyn(loaded, name, **{argument: value})


@pytest.mark.parametrize("name", NAMES)
def test_ase_defaults_of_refused_arguments_are_accepted_and_others_raise(loaded, name):
    assert _dyn(loaded, name, comm=world, master=None).simulation is None
    if name == "Bussi":
        with pytest.raises(PlanRefused, match="seed="):
            _dyn(loaded, name, rng=np.random.default_rng(1))
        assert _dyn(loaded, name, rng=None).simulation is None
    else:
        with pytest.raises(TypeError, match="rng"):
            _dyn(loaded, name, rng=np.random.default_rng(1))  # ASE's class takes no rng either


@pytest.mark.parametrize("name", NAMES)
def test_todict_as_ase(loaded, name):
    from ase.calculators.lj import LennardJones

    atoms = _atoms()
    atoms.calc = LennardJones()
    reference = _ase_class(name)(atoms, DT, **CASES[name][0])
    assert _dyn(loaded, name).todict() == reference.todict()


# --------------------------------------------------------------------------- ASE's own checks and units


def test_npt_berendsen_takes_ases_deprecated_units_and_defaults(loaded):
    from fastmace.ase import NPTBerendsen

    with pytest.warns(FutureWarning):
        dyn = NPTBerendsen(_atoms(), DT, temperature_K=300, pressure=1000.0, compressibility=5e-7, model=loaded,
                           seed=1)
    record = dyn.integrator
    assert record.pressure_GPa == pytest.approx(0.1, rel=1e-12)  # 1000 bar
    assert record.compressibility_per_GPa == pytest.approx(5e-3, rel=1e-12)  # 5e-7 per bar, 1 GPa = 1e4 bar
    assert record.taut_fs == pytest.approx(500.0, rel=1e-14) and record.taup_fs == pytest.approx(1000.0, rel=1e-14)
    with pytest.raises(TypeError, match="Exactly one"):
        NPTBerendsen(_atoms(), DT, temperature_K=300, compressibility_au=1.0, model=loaded)
    with pytest.raises(TypeError, match="compressibility"):
        NPTBerendsen(_atoms(), DT, temperature_K=300, pressure_au=0.0, model=loaded)


def test_nvt_berendsen_needs_taut_and_takes_temperature_in_kelvin(loaded):
    from fastmace.ase import NVTBerendsen

    with pytest.raises(TypeError, match="taut"):
        NVTBerendsen(_atoms(), DT, temperature_K=300, model=loaded)
    assert NVTBerendsen(_atoms(), DT, 450, TDAMP, model=loaded).integrator.temperature_K == 450.0


def test_bussi_refuses_a_start_at_rest_as_ase(loaded):
    from fastmace.ase import Bussi

    atoms = _atoms()
    atoms.set_momenta(np.zeros((len(atoms), 3)))
    with pytest.raises(ValueError, match="Initial kinetic energy is zero"):
        Bussi(atoms, DT, 300, TDAMP, model=loaded)


def test_velocity_verlet_sizes_the_workload_from_the_momenta(loaded):
    from fastmace import quick

    atoms = _atoms()
    expected = quick.nve_max_temperature_K(atoms)
    dyn = _dyn(loaded, "VelocityVerlet", atoms=atoms)
    dyn.run(1)
    assert dyn.simulation.workload.max_temperature_K == expected


# --------------------------------------------------------------------------- plan, run, files, observers


@pytest.mark.parametrize("name", NAMES)
def test_a_ported_script_gets_the_plan_of_the_equivalent_simulation(loaded, name):
    dyn = _dyn(loaded, name)
    dyn.run(2)
    assert dyn.plan.tier == "graph"
    assert dyn.plan == _simulation(loaded, dyn.integrator).plan


@pytest.mark.parametrize("name", NAMES)
def test_run_advances_like_the_simulation_and_writes_atoms(loaded, name):
    atoms = _atoms()
    dyn = _dyn(loaded, name, atoms=atoms)
    dyn.attach(lambda: None, interval=3)
    dyn.run(5)
    dyn.run(2)
    sim = _simulation(loaded, dyn.integrator, host_callbacks_every=3)
    sim.run(steps=7)
    committed, ours = sim.committed(), dyn.simulation.committed()
    for field in ("positions", "momenta", "cell"):
        np.testing.assert_array_equal(getattr(ours, field), getattr(committed, field), err_msg=field)
    np.testing.assert_array_equal(atoms.positions, committed.positions)
    np.testing.assert_array_equal(atoms.cell.array, committed.cell)
    assert atoms.get_potential_energy() == committed.energy


@pytest.mark.parametrize("name", NAMES)
def test_trajectory_reads_back_frame_for_frame(loaded, name, tmp_path):
    from ase.io import read

    from fastmace.io import output as output_io
    from fastmace.io.output import Output

    dyn = _dyn(loaded, name, trajectory="run.traj", loginterval=2)
    dyn.run(4)
    dyn.run(2)
    dyn.close()
    frames = read("run.traj", index=":")
    sim = _simulation(loaded, dyn.integrator, output=Output(str(tmp_path / "sim"), trajectory_every=2,
                                                            thermo_every=2, restart_every=None))
    sim.run(steps=6)
    sim.close()
    expected = output_io.read_frames(tmp_path / "sim")
    assert len(frames) == len(expected) == 4
    for frame, reference in zip(frames, expected):
        np.testing.assert_array_equal(frame.positions, reference.positions)
        np.testing.assert_array_equal(frame.get_momenta(), reference.get_momenta())
        np.testing.assert_array_equal(frame.cell, reference.cell)
        assert frame.get_potential_energy() == reference.get_potential_energy()
        np.testing.assert_array_equal(frame.get_forces(), reference.get_forces())


@pytest.mark.parametrize("name", NAMES)
def test_an_observer_change_equals_the_simulation_event(loaded, name):
    atoms = _atoms()
    dyn = _dyn(loaded, name, atoms=atoms)

    def change():
        if dyn.nsteps == 2:
            atoms.positions += 0.01
            atoms.set_momenta(atoms.get_momenta() * 0.5)

    dyn.attach(change, interval=2)
    dyn.run(4)
    sim = _simulation(loaded, dyn.integrator, host_callbacks_every=2)
    sim.run(steps=2)
    committed = sim.committed()
    sim.update(positions=committed.positions + 0.01, momenta=committed.momenta * 0.5)
    sim.run(steps=2)
    ours, theirs = dyn.simulation.committed(), sim.committed()
    for field in ("positions", "momenta", "cell", "forces"):
        np.testing.assert_array_equal(getattr(ours, field), getattr(theirs, field), err_msg=field)
    assert ours.energy == theirs.energy


@pytest.mark.parametrize("name", ["IsotropicMTKNPT", "MTKNPT", "NPTBerendsen"])
def test_an_observer_cell_change_raises_under_the_barostat_too(loaded, name):
    atoms = _atoms()
    dyn = _dyn(loaded, name, atoms=atoms)
    dyn.attach(lambda: atoms.set_cell(atoms.cell * 1.01, scale_atoms=False) if dyn.nsteps == 2 else None,
               interval=2)
    with pytest.raises(PlanRefused, match="cell"):
        dyn.run(4)
    assert dyn.simulation.committed().step == 2


@pytest.mark.parametrize("name", NAMES)
def test_the_log_counts_the_runs_degrees_of_freedom(loaded, name):
    """``dof="ase"`` counts 3N, as ASE's class and its ``MDLogger``: the log reads bare frames. ``fixcm``
    (Berendsen) counts 3N - 3 (decision 37), and the log reads its frames with ``FixCom``, as Langevin's."""
    dyn = _dyn(loaded, name, logfile="run.log")
    dyn.run(2)
    dyn.close()
    if name in WITHOUT_FIXCM:
        assert dyn.simulation.degrees_of_freedom == 3 * len(dyn.atoms)
        assert dyn._log_frame is None
    else:
        assert dyn.simulation.degrees_of_freedom == 3 * len(dyn.atoms) - 3
        assert dyn._log_frame is not None


HEADERS = {"VelocityVerlet": "NVE VelocityVerlet 1 fs", "Bussi": "NVT CSVR (Bussi) 1 fs",
           "NoseHooverChainNVT": "NVT Nose-Hoover chain 1 fs", "IsotropicMTKNPT": "NPT MTK hydrostatic 1 fs",
           "MTKNPT": "NPT MTK full 1 fs", "NVTBerendsen": "NVT Berendsen 1 fs", "NPTBerendsen": "NPT Berendsen 1 fs",
           "Inhomogeneous_NPTBerendsen": "NPT Berendsen 1 fs"}


@pytest.mark.parametrize("name", NAMES)
def test_the_report_names_the_integrator_and_its_label(loaded, name):
    dyn = _dyn(loaded, name)
    dyn.run(1)
    assert dyn.summary.splitlines()[0].endswith(HEADERS[name])
    barostat = "NPT barostat: experimental: not yet qualified (§10.2)"
    assert (barostat in dyn.summary) is (dyn.integrator.to_config(seed=0).ensemble == "npt")
    berendsen = f"{HEADERS[name].removesuffix(' 1 fs')}: non-canonical: equilibration only (§10.2)"
    assert (berendsen in dyn.summary) is ("Berendsen" in name)
    assert ("non-canonical" in dyn.summary) is ("Berendsen" in name)


# --------------------------------------------------------------------------- FixAtoms (4-T04, decision 44)

FIXED = [0, 5]
# Decision 44: FixAtoms with the fixed-cell schemes whose ASE class takes it (NVE, CSVR, NVT Berendsen and
# Langevin), without fixcm; refused for NHC and MTK (ASE refuses them) and for every variable cell.
FIX_ATOMS_RUNS = ("VelocityVerlet", "Bussi", "NVTBerendsen")
FIX_ATOMS_REFUSED = ("NoseHooverChainNVT", "IsotropicMTKNPT", "MTKNPT", "NPTBerendsen", "Inhomogeneous_NPTBerendsen")


def _fixed():
    from ase.constraints import FixAtoms

    atoms = _atoms()
    atoms.set_constraint(FixAtoms(indices=FIXED))
    return atoms


@pytest.mark.parametrize("name", FIX_ATOMS_RUNS)
def test_fix_atoms_runs_where_decision_44_allows_it(loaded, name):
    options = {"fixcm": False} if name in WITH_FIXCM else {}
    atoms = _fixed()
    start = atoms.positions[FIXED].copy()
    dyn = _dyn(loaded, name, atoms=atoms, logfile="run.log", **options)
    dyn.run(4)
    dyn.close()
    np.testing.assert_array_equal(atoms.positions[FIXED], start)
    np.testing.assert_array_equal(atoms.get_momenta()[FIXED], 0.0)
    assert dyn.simulation.degrees_of_freedom == 3 * (len(atoms) - len(FIXED))
    sim = _simulation(loaded, dyn.integrator, atoms=_fixed())
    sim.run(steps=4)
    assert dyn.plan == sim.plan
    np.testing.assert_array_equal(dyn.simulation.committed().positions, sim.committed().positions)


def test_fix_atoms_with_ases_default_fixcm_is_refused(loaded):
    """``NVTBerendsen``'s ``fixcm=True`` (ASE's default) with FixAtoms: decision 44, as ``Langevin``'s."""
    dyn = _dyn(loaded, "NVTBerendsen", atoms=_fixed())
    with pytest.raises(PlanRefused, match="fixcm"):
        dyn.run(1)


@pytest.mark.parametrize("name", ["VelocityVerlet", "Bussi"])
def test_fix_atoms_with_fastmaces_dof_runs_without_fixcm(loaded, name):
    """``dof="auto"`` resolves ``fixcm="auto"``: off with FixAtoms, 3(N - n_fixed) degrees of freedom (4-T04)."""
    dyn = _dyn(loaded, name, atoms=_fixed(), dof="auto")
    dyn.run(2)
    assert dyn.simulation.degrees_of_freedom == 3 * (len(dyn.atoms) - len(FIXED))
    assert dyn.plan.integrator.fixcm is False


@pytest.mark.parametrize("name", FIX_ATOMS_REFUSED)
def test_fix_atoms_is_refused_elsewhere_with_the_reason(loaded, name):
    options = {"fixcm": False} if name in WITH_FIXCM else {}
    dyn = _dyn(loaded, name, atoms=_fixed(), **options)
    with pytest.raises(PlanRefused, match="FixAtoms|fixed atoms|constraints") as raised:
        dyn.run(1)
    assert raised.value.codes  # rule 1's codes ("constraints", "fix-atoms-variable-cell"); the reason above


def test_fix_atoms_trajectory_keeps_the_constraint(loaded):
    from ase.io import read

    dyn = _dyn(loaded, "VelocityVerlet", atoms=_fixed(), trajectory="run.traj", loginterval=2)
    dyn.run(4)
    dyn.close()
    frames = read("run.traj", index=":")
    assert len(frames) == 3
    for frame in frames:
        assert [list(c.index) for c in frame.constraints] == [FIXED]
