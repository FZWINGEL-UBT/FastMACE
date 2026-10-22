"""Restart format version 3: the current composition and the MC record (plan task 3-T08; SPEC §10.6).

The format tests use ``test_restart.make_state`` (no force field); the resume tests run the tiny MH-1
model on the CPU in float64, where a resumed run equals the uninterrupted one bit for bit.
"""

import json

import numpy as np
import pytest
import torch
from ase.data import atomic_masses
from test_restart import make_state, write_full

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace import integrators
from fastmace.config import MC_STREAMS, EngineConfig
from fastmace.engine import constraints
from fastmace.io import output as output_io
from fastmace.io import restart as restart_io
from fastmace.io.output import Output
from fastmace.planner import minimal

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

NA_CL = (11, 17)
LANGEVIN = integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01)


def mc_state():
    """An MC record as T09's ``MCDriver.state_dict()`` gives it (contracts-3: ``MC_RECORD_KEYS``)."""
    return {
        "moves": [{"move": "swap", "pairs": [[11, 17]], "every": 10, "attempts": 4, "velocities": "exchange"},
                  {"move": "displace", "delta_A": 0.2, "every": 10, "attempts": 8, "adaptive": True}],
        "cycle_position": 7,
        "counters": {"swap": {"attempted": 8, "impossible": 1, "accepted": 3},
                     "displace": {"attempted": 16, "impossible": 0, "accepted": 9}},
        "streams": {stream: {"seed": 1000 + index, "offset": 24 * (index + 1)}
                    for index, stream in enumerate(MC_STREAMS)},
        "delta_A": {"displace": 0.1875},
    }


# --------------------------------------------------------------------------- the format


def test_the_writer_is_version_3():
    assert restart_io.FORMAT_VERSION == restart_io.NEXT_FORMAT_VERSION == 3


def test_a_v3_round_trip_holds_the_mc_record(tmp_path):
    state, integrator = make_state()
    state.mc = mc_state()
    path = tmp_path / restart_io.RESTART_NAME
    write_full(path, state, integrator)
    stored = restart_io.read_restart(path)
    assert int(stored["fastmace_restart"]) == 3
    assert restart_io.mc_record(stored) == mc_state()
    fresh, fresh_integrator = make_state(seed=5, steps=1)
    assert fresh.mc is None
    restart_io.load_state(stored, fresh, fresh_integrator)
    assert fresh.mc == mc_state()
    again = tmp_path / "again.npz"
    write_full(again, fresh, fresh_integrator)
    assert again.read_bytes() == path.read_bytes()  # save -> load -> save


def test_without_a_move_set_there_is_no_mc_record(tmp_path):
    state, integrator = make_state()
    path = tmp_path / restart_io.RESTART_NAME
    write_full(path, state, integrator)
    stored = restart_io.read_restart(path)
    assert "mc" not in stored and restart_io.mc_record(stored) is None
    fresh, fresh_integrator = make_state(seed=5, steps=1)
    fresh.mc = mc_state()  # a stale record does not survive a load of a run without MC
    restart_io.load_state(stored, fresh, fresh_integrator)
    assert fresh.mc is None


@pytest.mark.parametrize("change, match", [
    (lambda r: r.pop("delta_A"), "keys"),
    (lambda r: r.update(extra=1), "keys"),
    (lambda r: r.update(cycle_position=-1), "cycle_position"),
    (lambda r: r["counters"].update(teleport={"attempted": 0, "impossible": 0, "accepted": 0}), "teleport"),
    (lambda r: r["counters"]["swap"].pop("impossible"), "counters"),
    (lambda r: r["counters"]["swap"].update(accepted=1.5), "accepted"),
    (lambda r: r["streams"].pop("velocity"), "streams"),
    (lambda r: r["streams"]["proposal"].update(offset=-3), "offset"),
    (lambda r: r["moves"].append({"pairs": []}), "move"),
    (lambda r: r["delta_A"].update(displace=0.0), "delta_A"),
])
def test_a_malformed_mc_record_is_refused_before_writing(tmp_path, change, match):
    state, integrator = make_state()
    record = mc_state()
    change(record)
    state.mc = record
    path = tmp_path / restart_io.RESTART_NAME
    with pytest.raises(ValueError, match=match):
        write_full(path, state, integrator)
    assert not path.exists()


def test_a_v2_file_still_loads(tmp_path):
    """A version 2 file (fastmace-2): the same keys without the MC record."""
    state, integrator = make_state()
    arrays = restart_io.state_arrays(state, "fastmace", integrator=integrator, variable_cell=True,
                                     plan={"edge_bucket": 512, "neighbour_capacity": 4096},
                                     outputs={"output_chunk": 3, "output_rows": 41, "output_frames": 21})
    arrays["fastmace_restart"] = 2
    path = tmp_path / restart_io.RESTART_NAME
    np.savez(path, **arrays)
    stored = restart_io.read_restart(path)
    assert restart_io.format_version(stored) == 2 and restart_io.mc_record(stored) is None
    fresh, fresh_integrator = make_state(seed=5, steps=1)
    restart_io.load_state(stored, fresh, fresh_integrator)
    for name in ("positions", "velocities", "masses", "species", "cell", "forces", "energy", "charges"):
        assert torch.equal(getattr(fresh, name), getattr(state, name)), name
    assert fresh.mc is None and fresh.step == state.step
    assert restart_io.composition(stored)["species"] == [int(z) for z in state.species]


def test_the_composition_record(tmp_path):
    state, integrator = make_state()
    path = tmp_path / restart_io.RESTART_NAME
    write_full(path, state, integrator)
    composition = restart_io.composition(restart_io.read_restart(path))
    assert composition == {
        "species": [int(z) for z in state.species], "masses": [float(m) for m in state.masses.reshape(-1)],
        "charges": [float(q) for q in state.charges], "total_charge": 0.0, "spin_multiplicity": 1}
    assert restart_io.composition({"step": 0}) is None  # a research-engine file without species


def test_load_state_names_the_composition_on_a_species_mismatch(tmp_path):
    state, integrator = make_state()
    state.species[0] = 17
    path = tmp_path / restart_io.RESTART_NAME
    write_full(path, state, integrator)
    with pytest.raises(ValueError, match="composition"):
        restart_io.load_state(restart_io.read_restart(path), *make_state(steps=1))


# --------------------------------------------------------------------------- resume with the current composition


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    pytest.importorskip("cuequivariance_torch")
    from fastmace.model import loading

    model = tiny_mace_model(list(NA_CL), **FAMILIES["mh1"])
    path = tmp_path_factory.mktemp("models") / "tiny-mh1-11-17.model"
    torch.save(model, path)
    loading._LICENCES_SHOWN.add(str(path))
    return loading.load_pristine(str(path), trusted=True)


@pytest.fixture(autouse=True)
def cpu_planner(monkeypatch):
    planner = minimal.MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ={},
                                     process=constraints.ProcessRecord())
    monkeypatch.setattr(minimal, "default_planner", lambda force_field: planner)
    return planner


def _simulation(loaded, atoms, *, output=None, seed=11):
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    return Simulation(atoms, ForceField(loaded, atoms, precision="float64"), integrator=LANGEVIN, output=output,
                      seed=seed)


def _output(directory, every=3):
    return Output(directory, thermo_every=1, trajectory_every=1, restart_every=every)


def _swapped(atoms):
    """The structure after an accepted Na <-> Cl swap of atoms 0 and 1, with the new species' masses."""
    atoms = atoms.copy()
    numbers = atoms.numbers.copy()
    numbers[[0, 1]] = numbers[[1, 0]]
    assert numbers[0] != numbers[1]
    atoms.numbers = numbers
    atoms.set_masses(atomic_masses[numbers])
    return atoms


def _assert_same(got, want):
    a, b = got.committed(), want.committed()
    assert a.step == b.step
    np.testing.assert_array_equal(a.positions, b.positions)
    np.testing.assert_array_equal(a.momenta, b.momenta)
    np.testing.assert_array_equal(a.forces, b.forces)
    assert a.energy == b.energy


def test_resume_takes_the_composition_from_the_restart_not_the_initial_structure(loaded, tmp_path):
    """The initial structure in simulation.json is A; the run's composition at the restart step is B."""
    from fastmace.simulation import SIMULATION_NAME, Simulation

    initial = nacl(repeat=1, rattle=0.05, seed=3)
    current = _swapped(initial)
    whole = _simulation(loaded, current, output=_output(tmp_path / "whole"))
    whole.run(steps=7)
    whole.close()

    first = _simulation(loaded, current, output=_output(tmp_path / "parts"))
    first.run(steps=5)  # the restart file is from step 3
    del first
    record_path = tmp_path / "parts" / SIMULATION_NAME
    record = json.loads(record_path.read_text())
    record["structure"]["numbers"] = [int(z) for z in initial.numbers]
    record["structure"]["masses"] = [float(m) for m in initial.get_masses()]
    record_path.write_text(json.dumps(record))

    resumed = Simulation.resume(tmp_path / "parts")
    state = resumed._engine.state
    np.testing.assert_array_equal(state.species.numpy(), current.numbers)
    np.testing.assert_array_equal(state.masses.numpy().reshape(-1), current.get_masses())
    assert resumed.committed().step == 3
    resumed.run(steps=4)
    resumed.close()
    _assert_same(resumed, whole)
    assert output_io.read_thermo(tmp_path / "parts") == output_io.read_thermo(tmp_path / "whole")


def test_resume_after_a_host_species_change_equals_the_uninterrupted_run(loaded, tmp_path):
    """The species change at step 2 is a canonical point in both runs; the restart at step 3 follows it."""
    from fastmace.simulation import Simulation

    initial = nacl(repeat=1, rattle=0.05, seed=3)
    new_species = _swapped(initial).numbers

    def run_until(directory, steps):
        sim = _simulation(loaded, initial, output=_output(directory))
        sim.run(steps=2)
        try:
            sim.set_species(new_species)
        except NotImplementedError:
            pytest.skip("Simulation.set_species: plan task 3-T03 (merged before 3-T08)")
        sim.run(steps=steps - 2)
        return sim

    whole = run_until(tmp_path / "whole", 7)
    whole.close()
    first = run_until(tmp_path / "parts", 5)
    del first
    resumed = Simulation.resume(tmp_path / "parts")
    np.testing.assert_array_equal(resumed._engine.state.species.numpy(), new_species)
    resumed.run(steps=4)
    resumed.close()
    _assert_same(resumed, whole)


# --------------------------------------------------------------------------- restart writes at committed steps only


def test_no_restart_write_inside_an_mc_block(loaded, tmp_path, monkeypatch):
    sim = _simulation(loaded, nacl(repeat=1, rattle=0.05, seed=3))
    sim.run(steps=1)
    assert sim._mc_block_active is False
    sim.save_restart(tmp_path)
    before = (tmp_path / restart_io.RESTART_NAME).read_bytes()
    sim.run(steps=1)
    monkeypatch.setattr(sim, "_mc_block_active", True)
    with pytest.raises(RuntimeError, match="MC block"):
        sim.save_restart(tmp_path)
    assert (tmp_path / restart_io.RESTART_NAME).read_bytes() == before


def test_no_restart_write_from_an_overflowed_step(loaded, tmp_path, monkeypatch):
    sim = _simulation(loaded, nacl(repeat=1, rattle=0.05, seed=3))
    sim.run(steps=1)
    monkeypatch.setattr(sim._engine, "uncommitted", lambda: True, raising=False)
    with pytest.raises(RuntimeError, match="uncommitted"):
        sim.save_restart(tmp_path)
    assert not (tmp_path / restart_io.RESTART_NAME).exists()
    monkeypatch.setattr(sim._engine, "uncommitted", lambda: False, raising=False)
    sim.save_restart(tmp_path)
    assert (tmp_path / restart_io.RESTART_NAME).exists()


def test_the_mc_record_crosses_a_simulation_restart(loaded, tmp_path, monkeypatch):
    from _support import fake_mc
    from fastmace.simulation import Simulation, move_record

    drivers = fake_mc.install(monkeypatch)  # T09's driver API, on the host (3-T15)
    sim = _simulation(loaded, nacl(repeat=1, rattle=0.05, seed=3))
    sim.run(steps=2)
    sim._engine.state.mc = mc_state()  # T09's driver writes its state_dict here before a restart write
    sim.save_restart(tmp_path)
    resumed = Simulation.resume(tmp_path)
    (driver,) = drivers  # the resume re-attaches the restart's move set (3-T15) and loads its record
    assert driver.loaded == [mc_state()]
    assert [move_record(move) for move in driver.moves] == mc_state()["moves"]
    assert resumed._engine.state.mc == driver.state_dict()
