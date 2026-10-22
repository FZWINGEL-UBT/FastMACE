"""``fastmace.ase.Langevin`` and ``LangevinBAOAB``: ASE scripts on the graph engine (SPEC §7.7 level 1).

Plan task 1b-T21. Tiny MH-1-family model on the CPU without compilation, as ``test_simulation.py``.
The ASE classes are the installed ones (ASE 3.27 is the reference; a changed argument list fails
here, which is the point).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from ase import units
from ase.constraints import FixAtoms
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
FRICTION = 0.01 / units.fs


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
    monkeypatch.chdir(tmp_path)  # ./fastmace-run/ and relative trajectories land here


def _atoms():
    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    atoms.set_momenta(np.zeros((len(atoms), 3)))
    return atoms


def _langevin(loaded, atoms=None, **options):
    from fastmace.ase import Langevin

    atoms = _atoms() if atoms is None else atoms
    options.setdefault("seed", 1)
    return Langevin(atoms, DT, temperature_K=300, friction=FRICTION, model=loaded, **options)


def _simulation(loaded, atoms=None, integrator=None, seed=1, output=None, host_callbacks_every=None):
    from fastmace.model.forcefield import ForceField
    from fastmace.planner.workload import Workload
    from fastmace.simulation import Simulation

    atoms = _atoms() if atoms is None else atoms
    integrator = integrator or integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01)
    workload = Workload(max_temperature_K=float(integrator.temperature_K), host_callbacks_every=host_callbacks_every)
    return Simulation(atoms, ForceField(loaded, atoms, precision="float32"), integrator=integrator, seed=seed,
                      output=output, workload=workload)


# --------------------------------------------------------------------------- arguments


def _ase_arguments(cls):
    """Every constructor argument of an ASE dynamics class, along the whole class chain (``**kwargs``
    of one class go to the next), without ``self``, ``*args`` and ``**kwargs``."""
    names = []
    for klass in cls.__mro__:
        if klass is object or "__init__" not in vars(klass):
            continue
        for name, parameter in inspect.signature(klass.__init__).parameters.items():
            if name == "self" or parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                continue
            if name not in names:
                names.append(name)
    return names


@pytest.mark.parametrize("name", ["Langevin", "LangevinBAOAB"])
def test_every_ase_argument_is_supported_or_refused(name):
    import ase.md.langevin
    import ase.md.langevinbaoab

    from fastmace.ase import md

    ase_class = {"Langevin": ase.md.langevin.Langevin, "LangevinBAOAB": ase.md.langevinbaoab.LangevinBAOAB}[name]
    ours = getattr(md, name)
    ase_names = set(_ase_arguments(ase_class))
    known = set(ours.SUPPORTED) | set(ours.REFUSED)
    assert ase_names == known, (f"ASE's {name} takes {sorted(ase_names - known)} that fastmace.ase neither supports "
                                f"nor refuses, or no longer takes {sorted(known - ase_names)}")
    assert not set(ours.SUPPORTED) & set(ours.REFUSED)
    ours_signature = inspect.signature(ours.__init__).parameters
    for argument in ours.SUPPORTED:
        assert argument in ours_signature, argument
    for argument in md.ADDITIONS:
        assert ours_signature[argument].kind is inspect.Parameter.KEYWORD_ONLY, argument


def test_langevin_list_is_spec_7_7():
    from fastmace.ase import md

    assert set(md.Langevin.SUPPORTED) == {"atoms", "timestep", "temperature", "temperature_K", "friction", "fixcm",
                                          "trajectory", "logfile", "loginterval", "append_trajectory"}
    assert set(md.Langevin.REFUSED) == {"rng", "comm", "master"}


def test_an_unknown_argument_is_a_type_error_naming_it(loaded):
    with pytest.raises(TypeError, match="pressure"):
        _langevin(loaded, pressure=1.0)


@pytest.mark.parametrize("argument, value", [("rng", np.random.default_rng(1)), ("comm", object()),
                                             ("communicator", object()), ("master", True)])
def test_refused_arguments_name_themselves(loaded, argument, value):
    with pytest.raises(PlanRefused, match=argument):
        _langevin(loaded, **{argument: value})


def test_ase_defaults_of_refused_arguments_are_accepted(loaded):
    dyn = _langevin(loaded, rng=None, comm=world, master=None, communicator=world)
    assert dyn.simulation is None  # nothing planned before the first run or step


def test_model_or_force_field_exactly_one(loaded):
    from fastmace.ase import Langevin
    from fastmace.model.forcefield import ForceField

    atoms = _atoms()
    with pytest.raises(TypeError, match="model= or force_field="):
        Langevin(atoms, DT, temperature_K=300, friction=FRICTION)
    with pytest.raises(TypeError, match="model= or force_field="):
        Langevin(atoms, DT, temperature_K=300, friction=FRICTION, model=loaded,
                 force_field=ForceField(loaded, atoms))


def test_the_additions_are_keyword_only(loaded):
    from fastmace.ase import Langevin

    with pytest.raises(TypeError):
        Langevin(_atoms(), DT, None, FRICTION, True, loaded)  # model= cannot be positional


def test_temperature_exactly_one_as_ase(loaded):
    from fastmace.ase import Langevin

    with pytest.raises(TypeError, match="temperature"):
        Langevin(_atoms(), DT, friction=FRICTION, model=loaded, seed=1)
    with pytest.raises(TypeError, match="temperature"):
        Langevin(_atoms(), DT, 0.02, FRICTION, temperature_K=300, model=loaded, seed=1)
    with pytest.warns(FutureWarning):  # ASE's Langevin: temperature in eV, deprecated
        dyn = Langevin(_atoms(), DT, 300 * units.kB, FRICTION, model=loaded, seed=1)
    assert dyn.integrator.temperature_K == pytest.approx(300, rel=1e-14)


def test_friction_is_in_ase_units_never_reinterpreted(loaded):
    dyn = _langevin(loaded)
    assert dyn.todict()["friction"] == FRICTION
    assert dyn.integrator.friction_per_fs == pytest.approx(0.01, rel=1e-15)
    # friction=0.02 without /units.fs is about 0.002/fs, as in ASE
    from fastmace.ase import Langevin

    slow = Langevin(_atoms(), DT, temperature_K=300, friction=0.02, model=loaded, seed=1)
    assert slow.integrator.friction_per_fs == pytest.approx(0.02 * units.fs, rel=1e-15)
    assert slow.integrator.to_config(seed=1).friction == pytest.approx(0.02, rel=2e-16)


def test_todict_get_time_and_dt_as_ase(loaded):
    import ase.md.langevin

    reference = ase.md.langevin.Langevin(_atoms(), DT, temperature_K=300, friction=FRICTION)
    dyn = _langevin(loaded)
    assert dyn.todict() == reference.todict()
    assert dyn.dt == DT and dyn.nsteps == 0 and dyn.get_time() == 0.0
    dyn.run(3)
    assert dyn.nsteps == 3 and dyn.get_time() == 3 * DT


def test_fix_atoms_is_refused_with_the_phase_4_reason(loaded):
    atoms = _atoms()
    atoms.set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(PlanRefused, match="phase 4"):
        _langevin(loaded, atoms=atoms)


def test_dispersion_follows_fm_md(loaded):
    """'d3bj' expands from the training functional as fm.md does (a model loaded from a path has none,
    so only an explicit fm.D3 is accepted); a complete record is refused until phase 3b, shown."""
    import fastmace as fm

    with pytest.raises(ValueError, match="training functional is unknown") as error:
        _langevin(loaded, dispersion="d3bj")
    assert not isinstance(error.value, PlanRefused)
    with pytest.raises(PlanRefused, match=r"D3: phase 3b; the expanded record is D3\(functional=pbe"):
        _langevin(loaded, dispersion=fm.D3(functional="pbe"))
    with pytest.raises(ValueError, match="not 'd3bj'"):
        _langevin(loaded, dispersion="d4")


def test_d3bj_expansion_for_a_label_matches_fm_md(loaded, monkeypatch):
    """The label path: MH-1's r2SCAN head expands to the r2scan set, MACE-OFF raises (D3 counted twice)."""
    import dataclasses

    from fastmace.ase import md

    r2scan = dataclasses.replace(loaded, label="mh1", head="matpes_r2scan", functional="r2scan")
    with pytest.raises(PlanRefused, match="functional=r2scan"):
        md._dispersion("d3bj", r2scan)
    off = dataclasses.replace(loaded, label="mace-off-small", functional="wb97m-d3bj")
    with pytest.raises(ValueError, match="count it twice"):
        md._dispersion("d3bj", off)


def test_check_is_passed_through(loaded):
    with pytest.raises(ValueError, match="opt-in"):
        _langevin(loaded, check="always")
    dyn = _langevin(loaded)
    dyn.run(1)
    assert dyn.gate.status == "skipped"
    assert dyn.simulation.check == "skip"


def test_check_gate_reaches_the_simulation(loaded, monkeypatch):
    from fastmace import simulation

    seen = {}

    class Spy(simulation.Simulation):
        def __init__(self, *args, check="skip", **kwargs):
            seen["check"] = check
            super().__init__(*args, check="skip", **kwargs)

    monkeypatch.setattr(simulation, "Simulation", Spy)
    _langevin(loaded, check="gate").run(1)
    assert seen["check"] == "gate"


def test_an_unmapped_ase_class_names_the_mapped_ones():
    import fastmace.ase

    with pytest.raises(PlanRefused, match="Langevin, LangevinBAOAB") as error:
        from fastmace.ase import VelocityVerlet  # noqa: F401
    assert "phase 4" in str(error.value)
    with pytest.raises(AttributeError):
        fastmace.ase.NoSuchThing  # noqa: B018


# --------------------------------------------------------------------------- BAOAB mapping


def _baoab(loaded, **options):
    from fastmace.ase import LangevinBAOAB

    options.setdefault("temperature_K", 300)
    options.setdefault("externalstress", 0.0)
    options.setdefault("T_tau", 100 * units.fs)
    return LangevinBAOAB(_atoms(), DT, model=loaded, seed=1, **options)


def test_baoab_maps_ase_arguments(loaded):
    dyn = _baoab(loaded, externalstress=-1.0 * units.GPa, hydrostatic=True, P_tau=2000 * units.fs,
                 P_mass_factor=2.0, disable_cell_langevin=True)
    record = dyn.integrator
    assert record.cell_mode == "hydrostatic"
    assert record.external_stress_GPa == pytest.approx(-1.0, rel=1e-14)
    assert record.friction_per_fs == pytest.approx(0.01, rel=1e-15)
    assert record.barostat_tau_fs == pytest.approx(2000, rel=1e-14)
    assert record.barostat_mass_factor == 2.0 and record.disable_cell_langevin


def test_baoab_hydrostatic_false_is_the_full_cell(loaded):
    assert _baoab(loaded).integrator.cell_mode == "full"


@pytest.mark.parametrize("stress, voigt", [
    (0.5, (0.5, 0.5, 0.5, 0.0, 0.0, 0.0)),
    ((0.1, 0.2, 0.3), (0.1, 0.2, 0.3, 0.0, 0.0, 0.0)),
    ((0.1, 0.2, 0.3, 0.4, 0.5, 0.6), (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)),
    ([[0.1, 0.6, 0.5], [0.6, 0.2, 0.4], [0.5, 0.4, 0.3]], (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)),
])
def test_baoab_stress_shapes_become_voigt_gpa(loaded, stress, voigt):
    record = _baoab(loaded, externalstress=np.asarray(stress) * units.GPa).integrator
    np.testing.assert_allclose(np.broadcast_to(record.external_stress_GPa, 6), voigt, rtol=1e-14, atol=1e-15)


def test_baoab_refuses_a_non_symmetric_stress(loaded):
    with pytest.raises(PlanRefused, match="symmetric"):
        _baoab(loaded, externalstress=[[0.1, 0.2, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])


def test_baoab_hydrostatic_needs_a_scalar_as_ase(loaded):
    with pytest.raises(ValueError, match="scalar"):
        _baoab(loaded, externalstress=(0.1, 0.2, 0.3), hydrostatic=True)


def test_baoab_refusals(loaded):
    with pytest.raises(PlanRefused, match="use `Langevin`"):
        _baoab(loaded, externalstress=None)
    with pytest.raises(PlanRefused, match="phase 4"):
        _baoab(loaded, temperature_K=None)
    with pytest.raises(PlanRefused, match="P_mass"):
        _baoab(loaded, P_mass=1000.0)


def test_baoab_default_t_tau_is_ases(loaded):
    with pytest.warns(UserWarning, match="T_tau"):
        dyn = _baoab(loaded, T_tau=None)
    assert dyn.integrator.friction_per_fs == pytest.approx(1 / 50, rel=1e-14)


def test_baoab_todict_as_ase(loaded):
    import ase.md.langevinbaoab
    from ase.calculators.lj import LennardJones

    atoms = _atoms()
    atoms.calc = LennardJones()
    with pytest.warns(UserWarning):
        reference = ase.md.langevinbaoab.LangevinBAOAB(atoms, DT, temperature_K=300, externalstress=0.0,
                                                       T_tau=100 * units.fs, rng=1)
    assert _baoab(loaded).todict() == reference.todict()


# --------------------------------------------------------------------------- plan and run


def _plans_equal(a, b):
    """Plan equality up to the ulps of the unit conversions (ASE units -> record -> config): friction,
    timestep and external stress."""
    import dataclasses

    changes = {}
    for name in ("friction", "timestep_fs", "external_stress_GPa"):
        ours, theirs = getattr(a.integrator, name), getattr(b.integrator, name)
        np.testing.assert_allclose(np.asarray(ours, dtype=float), np.asarray(theirs, dtype=float), rtol=1e-15, atol=0)
        changes[name] = theirs
    a = dataclasses.replace(a, integrator=dataclasses.replace(a.integrator, **changes))
    assert a == b


def test_a_ported_script_gets_the_plan_of_the_equivalent_simulation(loaded):
    dyn = _langevin(loaded)
    dyn.run(2)
    assert dyn.plan.tier == "graph"
    _plans_equal(dyn.plan, _simulation(loaded).plan)


def test_baoab_plan_equals_the_simulations(loaded):
    dyn = _baoab(loaded, externalstress=-0.1 * units.GPa, hydrostatic=True)
    dyn.run(1)
    record = integrators.LangevinBAOAB(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01,
                                       external_stress_GPa=-0.1)
    _plans_equal(dyn.plan, _simulation(loaded, integrator=record).plan)


def test_run_advances_like_the_simulation_and_writes_atoms(loaded):
    atoms = _atoms()
    dyn = _langevin(loaded, atoms=atoms)
    dyn.attach(lambda: None, interval=3)
    dyn.run(5)
    dyn.run(2)
    sim = _simulation(loaded, host_callbacks_every=3)
    sim.run(steps=7)
    committed = sim.committed()
    np.testing.assert_array_equal(dyn.simulation.committed().positions, committed.positions)
    np.testing.assert_array_equal(atoms.positions, committed.positions)
    np.testing.assert_array_equal(atoms.get_momenta(), committed.momenta)
    assert atoms.get_potential_energy() == committed.energy
    np.testing.assert_array_equal(atoms.get_forces(), committed.forces)


def test_observer_calls_follow_ase(loaded):
    """attach(interval > 0) every interval steps from step 0; interval <= 0 once, at step |interval|."""
    import ase.md.langevin
    from ase.calculators.lj import LennardJones

    def record(dyn, seen, tag):
        dyn.attach(lambda: seen.append((tag, dyn.nsteps)), interval=2)
        dyn.attach(lambda: seen.append((tag + "-once", dyn.nsteps)), interval=-3)

    expected, got = [], []
    atoms = _atoms()
    atoms.calc = LennardJones()
    reference = ase.md.langevin.Langevin(atoms, DT, temperature_K=300, friction=FRICTION)
    record(reference, expected, "x")
    reference.run(4)
    reference.run(3)
    dyn = _langevin(loaded)
    record(dyn, got, "x")
    dyn.run(4)
    dyn.run(3)
    assert got == expected


def test_observer_args_and_write_objects(loaded):
    dyn = _langevin(loaded)
    seen = []

    class Writer:
        def write(self):
            seen.append("write")

    dyn.attach(seen.append, 2, "arg")
    dyn.attach(Writer(), interval=4)
    dyn.run(4)
    assert seen == ["arg", "write", "arg", "arg", "write"]
    assert len(dyn.observers) == 2


def test_observers_see_the_committed_state_without_a_new_evaluation(loaded):
    atoms = _atoms()
    dyn = _langevin(loaded, atoms=atoms)
    seen = []

    def observer():
        committed = dyn.simulation.committed()
        seen.append(dyn.nsteps)
        np.testing.assert_array_equal(atoms.positions, committed.positions)
        np.testing.assert_array_equal(atoms.get_momenta(), committed.momenta)
        assert atoms.get_potential_energy() == committed.energy
        np.testing.assert_array_equal(atoms.get_forces(), committed.forces)

    dyn.attach(observer, interval=2)
    dyn.run(4)
    assert seen == [0, 2, 4]


def test_an_observer_that_moves_atoms_equals_simulation_set_positions(loaded):
    atoms = _atoms()
    dyn = _langevin(loaded, atoms=atoms)

    def shift():
        if dyn.nsteps == 3:
            atoms.positions += 0.01

    dyn.attach(shift, interval=3)
    dyn.run(5)
    sim = _simulation(loaded, host_callbacks_every=3)
    sim.run(steps=3)
    sim.set_positions(sim.committed().positions + 0.01)
    sim.run(steps=2)
    np.testing.assert_array_equal(dyn.simulation.committed().positions, sim.committed().positions)
    np.testing.assert_array_equal(dyn.simulation.committed().momenta, sim.committed().momenta)
    assert any("host writes" in line for line in dyn.simulation.report.warnings())


def test_an_observer_that_changes_momenta_equals_simulation_set_momenta(loaded):
    atoms = _atoms()
    dyn = _langevin(loaded, atoms=atoms)

    def kick():
        if dyn.nsteps == 2:
            atoms.set_momenta(atoms.get_momenta() * 0.5)

    dyn.attach(kick, interval=2)
    dyn.run(4)
    sim = _simulation(loaded, host_callbacks_every=2)
    sim.run(steps=2)
    sim.set_momenta(sim.committed().momenta * 0.5)
    sim.run(steps=2)
    np.testing.assert_array_equal(dyn.simulation.committed().positions, sim.committed().positions)
    np.testing.assert_array_equal(dyn.simulation.committed().momenta, sim.committed().momenta)


# --------------------------------------------------------------------------- composition changes (3-T13)


def _observe(loaded, atoms, change, at=2, interval=2, steps=4):
    """A Langevin run whose observer applies ``change(atoms)`` at step ``at``."""
    dyn = _langevin(loaded, atoms=atoms)
    dyn.attach(lambda: change(atoms) if dyn.nsteps == at else None, interval=interval)
    dyn.run(steps)
    return dyn


def _assert_same_state(dyn, sim):
    ours, theirs = dyn.simulation.committed(), sim.committed()
    assert ours.step == theirs.step
    for name in ("species", "masses", "positions", "momenta", "forces"):
        np.testing.assert_array_equal(getattr(ours, name), getattr(theirs, name), err_msg=name)
    assert ours.energy == theirs.energy


def _swap(atoms):
    atoms.numbers[[0, 1]] = atoms.numbers[[1, 0]]


def test_an_observer_species_swap_equals_simulation_set_species(loaded):
    atoms = _atoms()
    assert not atoms.has("masses")  # implicit masses: ASE lets them follow the species
    dyn = _observe(loaded, atoms, _swap)
    sim = _simulation(loaded, host_callbacks_every=2)
    sim.run(steps=2)
    swapped = sim.committed().species.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    sim.set_species(swapped)  # the masses follow the standard masses, as ASE's implicit ones
    sim.run(steps=2)
    _assert_same_state(dyn, sim)
    np.testing.assert_array_equal(atoms.numbers, swapped)
    assert not atoms.has("masses")  # written back only where the engine changed them


def test_an_observer_transmutation_equals_simulation_set_species(loaded):
    def transmute(atoms):
        atoms.numbers[0] = 17 if atoms.numbers[0] == 11 else 11

    atoms = _atoms()
    dyn = _observe(loaded, atoms, transmute)
    sim = _simulation(loaded, host_callbacks_every=2)
    sim.run(steps=2)
    species = sim.committed().species.copy()
    species[0] = 17 if species[0] == 11 else 11
    sim.set_species(species)
    sim.run(steps=2)
    _assert_same_state(dyn, sim)
    assert atoms.get_masses()[0] == sim.committed().masses[0]


def test_explicit_masses_stay_through_an_observer_species_swap(loaded):
    masses = np.linspace(20.0, 40.0, len(_atoms()))
    atoms = _atoms()
    atoms.set_masses(masses)
    dyn = _observe(loaded, atoms, _swap)
    reference = _atoms()
    reference.set_masses(masses)
    sim = _simulation(loaded, atoms=reference, host_callbacks_every=2)
    sim.run(steps=2)
    committed = sim.committed()
    swapped = committed.species.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    sim.update(species=swapped, masses=committed.masses)
    sim.run(steps=2)
    _assert_same_state(dyn, sim)
    np.testing.assert_array_equal(dyn.simulation.committed().masses, masses)


def test_an_observer_mass_change_equals_simulation_set_masses(loaded):
    def heavier(atoms):
        atoms.set_masses(atoms.get_masses() * 1.1)

    dyn = _observe(loaded, _atoms(), heavier)
    sim = _simulation(loaded, host_callbacks_every=2)
    sim.run(steps=2)
    sim.set_masses(sim.committed().masses * 1.1)  # the momenta stay, as in ASE
    sim.run(steps=2)
    _assert_same_state(dyn, sim)


def test_an_observer_composition_and_motion_change_is_one_event(loaded, monkeypatch):
    from fastmace.simulation import Simulation

    calls = []
    update = Simulation.update

    def counted(self, **fields):
        calls.append(sorted(name for name, value in fields.items() if value is not None))
        return update(self, **fields)

    monkeypatch.setattr(Simulation, "update", counted)

    def change(atoms):
        _swap(atoms)
        atoms.positions += 0.01
        atoms.set_momenta(atoms.get_momenta() * 0.5)

    dyn = _observe(loaded, _atoms(), change)
    assert calls == [["masses", "momenta", "positions", "species"]]
    sim = _simulation(loaded, host_callbacks_every=2)
    sim.run(steps=2)
    committed = sim.committed()
    reference = _atoms()
    reference.numbers = committed.species.copy()
    _swap(reference)
    sim.update(species=reference.numbers, masses=reference.get_masses(), positions=committed.positions + 0.01,
               momenta=committed.momenta * 0.5)
    sim.run(steps=2)
    _assert_same_state(dyn, sim)


def test_an_observer_species_outside_the_superset_raises_before_anything_is_applied(loaded):
    def lithium(atoms):
        atoms.numbers[0] = 3
        atoms.positions += 0.01

    atoms = _atoms()
    dyn = _langevin(loaded, atoms=atoms)
    dyn.attach(lambda: lithium(atoms) if dyn.nsteps == 2 else None, interval=2)
    with pytest.raises(PlanRefused, match="superset") as raised:
        dyn.run(4)
    assert "observer-species" in raised.value.codes
    committed = dyn.simulation.committed()
    assert committed.step == 2
    np.testing.assert_array_equal(committed.species, _atoms().numbers)
    sim = _simulation(loaded, host_callbacks_every=2)
    sim.run(steps=2)
    np.testing.assert_array_equal(committed.positions, sim.committed().positions)


def test_an_observer_charge_change_needs_a_model_that_takes_charges(loaded):
    """The tiny MH-1 model takes no per-atom charges: the change is refused, as through ``Simulation``,
    before anything is applied."""
    atoms = _atoms()
    dyn = _langevin(loaded, atoms=atoms)
    dyn.attach(lambda: atoms.set_initial_charges(np.full(len(atoms), 0.1)) if dyn.nsteps == 2 else None,
               interval=2)
    with pytest.raises(PlanRefused, match="charges") as raised:
        dyn.run(4)
    assert "observer-charges" in raised.value.codes
    assert dyn.simulation.committed().step == 2
    assert dyn.simulation.committed().charges is None
    with pytest.raises(ValueError, match="charges"):
        _simulation(loaded).set_charges(np.full(len(atoms), 0.1))


def test_observers_declare_the_shortest_interval_as_host_callbacks(loaded):
    dyn = _langevin(loaded)
    dyn.attach(lambda: None, interval=5)
    dyn.attach(lambda: None, interval=3)
    dyn.attach(lambda: None, interval=-2)  # once, at step 2: not a cadence
    dyn.run(1)
    assert dyn.simulation.workload.host_callbacks_every == 3
    _plans_equal(dyn.plan, _simulation(loaded, host_callbacks_every=3).plan)
    # Decision 38 (3-T06b): observers keep the graph tier, with a host sync at each call.
    assert dyn.plan.tier == "graph"
    assert dyn.plan.decision("workload.host_callbacks_every")[2].code == "host-callbacks-graph-sync"
    plain = _langevin(loaded, output="plain")
    plain.run(1)
    assert plain.simulation.workload.host_callbacks_every is None


def test_frames_after_an_observer_species_change_carry_the_new_species(loaded):
    from ase.io import read

    atoms = _atoms()
    dyn = _langevin(loaded, atoms=atoms, trajectory="run.traj", loginterval=2)
    dyn.attach(lambda: _swap(atoms) if dyn.nsteps == 2 else None, interval=2)
    dyn.run(4)
    dyn.close()
    frames = read("run.traj", index=":")
    original = _atoms().numbers
    swapped = original.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    np.testing.assert_array_equal(frames[1].numbers, original)  # step 2 was written before the change
    np.testing.assert_array_equal(frames[2].numbers, swapped)
    np.testing.assert_array_equal(frames[2].get_masses(), dyn.simulation.committed().masses)


def _count(atoms):
    atoms.pop()


def _cell(atoms):
    atoms.set_cell(atoms.cell * 1.01, scale_atoms=False)


def _pbc(atoms):
    atoms.pbc = (True, True, False)


def _constraint(atoms):
    atoms.set_constraint(FixAtoms(indices=[0]))


def _info(atoms):
    atoms.info["charge"] = 1


def _calc(atoms):
    from ase.calculators.lj import LennardJones

    atoms.calc = LennardJones()


@pytest.mark.parametrize("change, reason", [
    (_count, "atom count"), (_cell, "cell"), (_pbc, "PBC"), (_constraint, "constraints"), (_info, "info"),
    (_calc, "atoms.calc")])
def test_forbidden_observer_changes_raise_with_their_reason(loaded, change, reason):
    atoms = _atoms()
    dyn = _langevin(loaded, atoms=atoms)
    dyn.attach(lambda: change(atoms) if dyn.nsteps == 2 else None, interval=2)
    with pytest.raises(PlanRefused, match=reason):
        dyn.run(4)
    assert dyn.simulation.committed().step == 2


# --------------------------------------------------------------------------- files and report


def test_trajectory_reads_back_frame_for_frame(loaded, tmp_path):
    from ase.io import read

    from fastmace.io import output as output_io
    from fastmace.io.output import Output

    dyn = _langevin(loaded, trajectory="run.traj", loginterval=2)
    dyn.run(4)
    dyn.run(2)
    dyn.close()
    frames = read("run.traj", index=":")
    sim = _simulation(loaded, output=Output(str(tmp_path / "sim"), trajectory_every=2, thermo_every=2,
                                            restart_every=None))
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


def test_the_files_never_stop_the_run(loaded, monkeypatch):
    """trajectory= and logfile= are written by the Simulation's sink: one Simulation.run per run()."""
    from fastmace import simulation

    calls = []
    original = simulation.Simulation.run

    def run(self, steps=None, schedule=None, mc=...):
        calls.append(steps)
        return original(self, steps=steps, schedule=schedule, mc=mc)

    monkeypatch.setattr(simulation.Simulation, "run", run)
    dyn = _langevin(loaded, trajectory="run.traj", logfile="run.log", loginterval=2)
    dyn.run(6)
    assert calls == [6]


def test_a_run_call_costs_what_simulation_run_costs(loaded, monkeypatch, tmp_path):
    """1b-T21b: per run() the shim builds no metadata record of its own (provenance.fastmace_block
    queries the software versions, about 0.25 s), takes no extra committed() sync without observers,
    and writes metadata.json and report.txt once each beside the Simulation's own write."""
    from fastmace import provenance, simulation
    from fastmace.io.output import Output

    counts = {"block": 0, "committed": 0, "write_metadata": 0}
    original_block, original_write = provenance.fastmace_block, provenance.write_metadata
    original_committed = simulation.Simulation.committed

    def block(*args, **kwargs):
        counts["block"] += 1
        return original_block(*args, **kwargs)

    def write(*args, **kwargs):
        counts["write_metadata"] += 1
        return original_write(*args, **kwargs)

    def committed(self):
        counts["committed"] += 1
        return original_committed(self)

    monkeypatch.setattr(provenance, "fastmace_block", block)
    monkeypatch.setattr(provenance, "write_metadata", write)
    monkeypatch.setattr(simulation.Simulation, "committed", committed)

    def per_call(run):
        run(2)  # setup and the step-0 work
        before = dict(counts)
        run(3)
        return {key: counts[key] - before[key] for key in counts}

    sim = _simulation(loaded, output=Output(str(tmp_path / "sim")))
    reference = per_call(lambda steps: sim.run(steps=steps))
    shim = per_call(_langevin(loaded).run)
    assert shim["block"] == reference["block"] == 1
    assert shim["write_metadata"] == reference["write_metadata"] + 1  # the merged "ase" block
    assert shim["committed"] == reference["committed"] + 1 == 1  # atoms hold the final state


def test_append_trajectory_keeps_frames_as_ase(loaded):
    from ase.io import read

    first = _langevin(loaded, trajectory="run.traj", loginterval=2)
    first.run(2)
    first.close()
    # A second run never reuses the first one's output chunks (restart data is never deleted).
    with pytest.raises(FileExistsError, match="output chunks"):
        _langevin(loaded, trajectory="run.traj", loginterval=2, append_trajectory=True).run(2)
    dyn = _langevin(loaded, trajectory="run.traj", loginterval=2, append_trajectory=True, output="second")
    dyn.run(2)
    dyn.close()
    assert len(read("run.traj", index=":")) == 4
    dyn = _langevin(loaded, trajectory="run.traj", loginterval=2, output="third")
    dyn.run(2)
    dyn.close()
    assert len(read("run.traj", index=":")) == 2


@pytest.mark.parametrize("projection", ["auto", "ase"])
def test_the_log_file_is_ases_format(loaded, projection):
    """ASE's MDLogger over the trajectory's frames gives the log; under the default mass-weighted projection
    (decision 37) with ASE's FixCom, so its temperature counts 3N - 3 (3-T22c), with "ase" over 3N."""
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.constraints import FixCom
    from ase.io import read
    from ase.md.logger import MDLogger

    dyn = _langevin(loaded, trajectory="run.traj", logfile="run.log", loginterval=2, fixcm_projection=projection)
    dyn.run(4)
    dyn.close()

    class Clock:
        step = 0

        def get_time(self):
            return self.step * DT

    clock = Clock()
    frames = read("run.traj", index=":")
    atoms = frames[0].copy()
    if projection == "auto":
        atoms.set_constraint(FixCom())
    with MDLogger(clock, atoms, "expected.log", mode="w") as logger:
        for i, frame in enumerate(frames):
            clock.step = 2 * i
            atoms.set_positions(frame.positions, apply_constraint=False)
            atoms.cell = frame.cell
            atoms.set_momenta(frame.get_momenta(), apply_constraint=False)
            atoms.calc = SinglePointCalculator(atoms, energy=frame.get_potential_energy())
            logger()
    with open("run.log") as got, open("expected.log") as expected:
        assert got.read() == expected.read()


def test_output_defaults_next_to_the_trajectory_or_fastmace_run(loaded, tmp_path):
    import json

    dyn = _langevin(loaded, trajectory="sub/run.traj", loginterval=2)
    (tmp_path / "sub").mkdir()
    dyn.run(3)
    dyn.close()
    assert dyn.output == str(tmp_path / "sub" / "run.fastmace")
    metadata = json.loads((tmp_path / "sub" / "run.fastmace" / "metadata.json").read_text())
    assert metadata["ase"]["class"] == "Langevin"
    assert metadata["ase"]["trajectory"] == str(tmp_path / "sub" / "run.traj")
    assert metadata["ase"]["last_frame_step"] == 2
    assert dyn.metadata["ase"] == metadata["ase"]
    assert (tmp_path / "sub" / "run.fastmace" / "report.txt").read_text() == dyn.summary + "\n"
    plain = _langevin(loaded)
    plain.run(1)
    plain.close()
    assert plain.output == str(tmp_path / "fastmace-run")
    explicit = _langevin(loaded, output=tmp_path / "elsewhere")
    assert explicit.output == str(tmp_path / "elsewhere")


def test_summary_is_the_simulations_report(loaded, capsys):
    dyn = _langevin(loaded, trajectory="run.traj", logfile="run.log")
    assert dyn.summary is None and dyn.plan is None and dyn.gate is None
    dyn.run(2)
    printed = capsys.readouterr().out
    assert dyn.summary == dyn.simulation.report.text()
    assert "speed:   pending" in printed  # the setup report is printed at the first run
    output = next(line for line in dyn.summary.splitlines() if line.startswith("output:"))
    assert "run.traj" in output and "run.log" in output
    reference = _simulation(loaded).report.text().splitlines()
    skip = ("setup:", "speed:", "output:")
    assert [line for line in dyn.summary.splitlines() if not line.startswith(skip)] == \
        [line for line in reference if not line.startswith(skip)]


def test_summary_equals_fm_md_report_for_the_same_run(loaded, tmp_path):
    """SPEC §7.7: fm.md and the fastmace.ase integrators keep one report. Same model, structure,
    integrator, seed and steps; the setup times, the speed line and the output paths differ by nature."""
    import fastmace as fm

    atoms = _atoms()
    dyn = _langevin(loaded, atoms=atoms.copy(), trajectory="run.traj")
    dyn.run(3)
    record = integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01)
    run = fm.md(atoms.copy(), model=loaded, integrator=record, steps=3, seed=1, output=tmp_path / "md")
    skip = ("setup:", "speed:", "output:")
    assert [line for line in dyn.summary.splitlines() if not line.startswith(skip)] == \
        [line for line in run.summary.splitlines() if not line.startswith(skip)]
    _plans_equal(dyn.plan, run.plan)
    assert dyn.gate.status == run.gate.status == "skipped"


def test_step_advances_one_step_and_returns_forces(loaded):
    atoms = _atoms()
    dyn = _langevin(loaded, atoms=atoms)
    forces = dyn.step()
    assert dyn.simulation.committed().step == 1
    np.testing.assert_array_equal(forces, atoms.get_forces())


def test_frames_follow_a_species_change_made_through_the_simulation(loaded, tmp_path):
    """3-T13's finding, fixed in 3-T15: the trajectory takes each frame's species from the snapshot."""
    from ase.io import read

    path = tmp_path / "md.traj"
    dyn = _langevin(loaded, trajectory=str(path), loginterval=1)
    dyn.run(1)
    swapped = dyn.simulation.committed().species.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    dyn.simulation.set_species(swapped)  # directly, not through an observer
    dyn.run(1)
    dyn.close()
    frames = read(str(path), index=":")
    np.testing.assert_array_equal(frames[-1].numbers, swapped)
    np.testing.assert_array_equal(frames[-1].get_masses(), dyn.simulation.committed().masses)
    assert not np.array_equal(frames[0].numbers, swapped)
