"""The complete state and the restart format (versions 2 and 3; plan tasks 1b-T05, 3-T08; SPEC §10.6, §9.5).

Format tests on a real ``SimulationState`` with a real integrator's noise layout and barostat state;
no force field is needed (the committed force results are filled with numbers). The integrator
trajectories across a restart are in ``test_integrators.py``; the 1a output and resume tests in
``test_io.py``.
"""

import dataclasses
import json
import time
import zipfile

import numpy as np
import pytest
import torch
from ase import units

from _support.structures import nacl
from fastmace.config import IntegratorConfig
from fastmace.errors import FastMACEError, RestartMismatch
from fastmace.integrators.baoab import LangevinBAOAB
from fastmace.integrators.langevin import Langevin
from fastmace.io import restart as restart_io
from fastmace.state import FIELDS_ADDED_IN_1B, ForceFieldIdentity, SimulationState

SEED = 11


def identity(**changes):
    fields = dict(checkpoint_sha256="ab" * 32, head="omat_pbe", species=(11, 17), force_precision="float32",
                  physics={"charge_spin": False, "d3": None, "polar": None, "reference_corrections": []},
                  reference_record_id="ref-1", mace_version="0.3.16", fastmace_version="0.0.1.dev0")
    fields.update(changes)
    return ForceFieldIdentity(**fields)


def plan_record(**changes):
    record = {"execution_mode": "graph", "backend": "cueq", "rewrites": ["fold_cutoff", "block_linear"],
              "mc_energy_precision": None, "edge_bucket": 512, "neighbour_capacity": 4096}
    record.update(changes)
    return record


@dataclasses.dataclass
class StubProtocol:
    """The protocol contract of plan task 1b-T07 (``to_restart`` / ``from_restart``) until it merges."""

    segments: list
    timestep_fs: float
    step: int
    target_K: float

    def to_restart(self):
        return {"segments": self.segments, "timestep_fs": self.timestep_fs, "step": self.step,
                "target_K": self.target_K}

    @classmethod
    def from_restart(cls, data):
        return cls(data["segments"], data["timestep_fs"], data["step"], data["target_K"])


def make_state(npt=True, cell_mode="hydrostatic", seed=SEED, steps=4):
    """A state with a partly consumed noise batch, committed results and (NPT) barostat numbers."""
    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    atoms.set_initial_charges(np.linspace(-0.5, 0.5, len(atoms)))
    state = SimulationState.from_atoms(atoms, "cpu", seed, compute_stress=npt)
    if npt:
        config = IntegratorConfig(ensemble="npt", cell_mode=cell_mode, noise_batch=7, friction=0.01,
                                  pressure_GPa=1.0)
        integrator = LangevinBAOAB(state, config, temperature_K=300.0)
    else:
        config = IntegratorConfig(ensemble="nvt", noise_batch=7, friction=0.01)
        integrator = Langevin(state, config, temperature_K=300.0)
    for _ in range(steps):
        integrator.prepare_step(state)
    numbers = np.random.default_rng(seed + 1)
    state.forces.copy_(torch.from_numpy(numbers.standard_normal(state.forces.shape)))
    state.energy.fill_(-12.25)
    state.velocities.copy_(torch.from_numpy(numbers.standard_normal(state.velocities.shape)) * 0.01)
    if npt:
        state.stress.copy_(torch.from_numpy(numbers.standard_normal((3, 3))) * 1e-3)
        state.p_eps.copy_(torch.from_numpy(numbers.standard_normal(state.p_eps.shape)))
        state.force_eps.copy_(torch.from_numpy(numbers.standard_normal(state.force_eps.shape)))
        state.cell.mul_(1.01)
    state.charges = torch.tensor(atoms.get_initial_charges(), dtype=torch.float64)
    state.total_charge = 0.0
    state.spin_multiplicity = 1
    state.step = steps
    state.plan_revision = 2
    state.force_field_identity = identity()
    state.protocol = StubProtocol([["ramp", 0.0, 300.0, 1000], ["hold", 300.0, 500]], 0.5, steps, 1.2)
    return state, integrator


def outputs():
    return {"output_chunk": 3, "output_rows": 41, "output_frames": 21}


def write_full(path, state, integrator):
    restart_io.write_restart(path, state, engine="fastmace", integrator=integrator,
                             variable_cell=integrator.variable_cell, plan=plan_record(), outputs=outputs())


# -------------------------------------------------------------------------- SimulationState

def test_the_1b_fields_are_in_the_state():
    """1b-T05 emptied FIELDS_ADDED_IN_1B except the MC state, which 3-T08 adds as ``SimulationState.mc``."""
    assert FIELDS_ADDED_IN_1B == {}
    fields = {field.name for field in dataclasses.fields(SimulationState)}
    assert {"charges", "total_charge", "spin_multiplicity", "pbc", "constraints", "p_eps", "force_eps",
            "protocol", "plan_revision", "force_field_identity", "mc"} <= fields
    atoms = nacl(repeat=1)
    atoms.pbc = (True, True, False)
    state = SimulationState.from_atoms(atoms, "cpu", 0)
    assert state.pbc == (True, True, False)
    assert state.constraints == () and state.plan_revision == 0
    assert state.charges is None and state.total_charge is None and state.spin_multiplicity is None
    assert state.p_eps is None and state.force_eps is None
    assert state.protocol is None and state.force_field_identity is None and state.mc is None


def test_baoab_keeps_the_barostat_in_the_state():
    for cell_mode, shape in (("hydrostatic", ()), ("full", (3, 3))):
        state, integrator = make_state(cell_mode=cell_mode, steps=0)
        assert tuple(state.p_eps.shape) == shape and state.p_eps.dtype == torch.float64
        assert integrator.p_eps is state.p_eps and integrator.force_eps is state.force_eps
        assert integrator.state_dict() == {}
    nvt, _ = make_state(npt=False, steps=0)
    assert nvt.p_eps is None and nvt.force_eps is None


# -------------------------------------------------------------------------- ForceFieldIdentity

# Pinned so that a change to the digest's definition is a deliberate one: restarts written before it
# would no longer resume.
PINNED_DIGEST = "3bc9e70238dcaf39eee72e5da241b1366d6a81cc4cba35d83a5b45e5118e02b3"


def test_identity_digest_is_stable():
    first, second = identity(), identity(physics={"reference_corrections": [], "polar": None, "d3": None,
                                                  "charge_spin": False}, species=[17, 11])
    assert first == second and hash(first) == hash(second)
    assert first.digest == second.digest == PINNED_DIGEST
    assert len(first.digest) == 64
    assert ForceFieldIdentity.from_dict(json.loads(json.dumps(first.to_dict()))) == first


@pytest.mark.parametrize("field, value", [
    ("checkpoint_sha256", "cd" * 32), ("head", "omol"), ("head", None), ("species", (11, 17, 35)),
    ("force_precision", "float64"), ("physics", {"charge_spin": True, "d3": None, "polar": None,
                                                 "reference_corrections": []}),
    ("physics", {"charge_spin": False, "d3": {"functional": "pbe", "damping": "bj"}, "polar": None,
                 "reference_corrections": []}),
    ("reference_record_id", "ref-2"), ("reference_record_id", None), ("mace_version", "0.3.17"),
    ("fastmace_version", "0.1.0"),
])
def test_identity_digests_differ_when_any_field_differs(field, value):
    other = identity(**{field: value})
    assert other.digest != identity().digest
    assert field in identity().differences(other)
    assert identity().differences(identity()) == ()


def test_every_identity_field_is_in_the_digest():
    """A new field that is not part of the digest would let resume accept another force field."""
    names = {field.name for field in dataclasses.fields(ForceFieldIdentity)}
    assert names == {"checkpoint_sha256", "head", "species", "force_precision", "physics", "reference_record_id",
                     "mace_version", "fastmace_version"}
    assert set(identity().to_dict()) == names | {"schema"}


@pytest.mark.parametrize("changes, match", [
    (dict(checkpoint_sha256="xyz"), "checkpoint_sha256"), (dict(checkpoint_sha256="AB" * 32), "checkpoint_sha256"),
    (dict(species=()), "species"), (dict(species=(11, 11)), "species"), (dict(species=(0, 11)), "species"),
    (dict(force_precision="float16"), "force_precision"), (dict(physics={"x": float("nan")}), "physics"),
    (dict(physics={"x": object()}), "physics"), (dict(mace_version=""), "mace_version"),
    (dict(head=3), "head"),
])
def test_identity_is_validated(changes, match):
    with pytest.raises(ValueError, match=match):
        identity(**changes)


def test_identity_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        identity().head = "omol"


# -------------------------------------------------------------------------- every §10.6 item

# SPEC §10.6, bullet by bullet: the restart entries that hold each item. JSON records are written
# "record.key". MC state is phase 3; the canonical-restart-point state is reset, not stored (1b-T14).
SPEC_10_6 = {
    "Physical state: positions": ["positions"],
    "momenta": ["momenta", "velocities"],
    "cell": ["cell", "cell_state"],
    "current per-atom species": ["species"],
    "masses": ["masses"],
    "per-atom charges where the model takes them": ["charges"],
    "total charge": ["physical.total_charge"],
    "spin": ["physical.spin_multiplicity"],
    "PBC": ["physical.pbc"],
    "constraints with their parameters": ["physical.constraints"],
    "barostat state (p_eps, cell force)": ["p_eps", "force_eps"],
    "energy, forces and stress of the committed step": ["energy", "forces", "stress"],
    "Integrator and protocol state: integrator parameters": ["integrator.class", "integrator.config"],
    "integrator internal state": ["integrator_keys"],
    "step": ["step"],
    "schedule and position in it": ["protocol.segments", "protocol.step"],
    "thermostat target": ["integrator.temperature_K", "protocol.target_K"],
    "the protocol recorded at start": ["protocol"],
    "Random state: every generator": ["rng_state"],
    "the random numbers already drawn ahead with their cursor": ["noise", "noise_next"],
    "Force-field identity": ["identity", "identity_digest"],
    "Plan: execution mode, backend, rewrites and MC energy precision": [
        "plan.execution_mode", "plan.backend", "plan.rewrites", "plan.mc_energy_precision"],
    "edge and capacity buckets": ["plan.edge_bucket", "plan.neighbour_capacity"],
    "the active plan revision (§9.5)": ["plan_revision"],
    "Output files at restart points: chunk indices": ["output_chunk", "output_rows", "output_frames"],
}
IDENTITY_ITEMS = {"checkpoint SHA-256": "checkpoint_sha256", "head": "head", "species superset": "species",
                  "force precision": "force_precision", "physics options": "physics",
                  "MACE and FastMACE versions": ("mace_version", "fastmace_version"),
                  "reference record id (§12.1)": "reference_record_id"}


def _lookup(stored, entry):
    name, _, key = entry.partition(".")
    assert name in stored, entry
    if key:
        record = json.loads(str(stored[name]))
        assert key in record, entry


def test_every_spec_item_is_in_the_file(tmp_path):
    state, integrator = make_state()
    path = tmp_path / restart_io.RESTART_NAME
    write_full(path, state, integrator)
    stored = restart_io.read_restart(path)
    for item, entries in SPEC_10_6.items():
        for entry in entries:
            _lookup(stored, entry)
    record = json.loads(str(stored["identity"]))
    for item, keys in IDENTITY_ITEMS.items():
        for key in (keys,) if isinstance(keys, str) else keys:
            assert key in record, item
    assert int(stored["fastmace_restart"]) == restart_io.FORMAT_VERSION == 4
    assert "mc" not in stored  # only with an attached move set (test_restart_v3.py)


# -------------------------------------------------------------------------- round trips

def test_save_load_save_gives_identical_files(tmp_path, monkeypatch):
    """Byte for byte, also when the second write is in another second (np.savez stamps the time)."""
    state, integrator = make_state(cell_mode="full")
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    write_full(first, state, integrator)
    fresh, fresh_integrator = make_state(cell_mode="full", seed=99, steps=1)
    restart_io.load_state(restart_io.read_restart(first), fresh, fresh_integrator)
    later = time.time() + 3600.0
    monkeypatch.setattr(time, "time", lambda: later)
    monkeypatch.setattr(time, "localtime", lambda *args: time.gmtime(later))
    write_full(second, fresh, fresh_integrator)
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize("npt, cell_mode", [(False, None), (True, "hydrostatic"), (True, "full")])
def test_load_restores_the_complete_state(tmp_path, npt, cell_mode):
    state, integrator = make_state(npt=npt, cell_mode=cell_mode or "hydrostatic")
    path = tmp_path / restart_io.RESTART_NAME
    write_full(path, state, integrator)
    fresh, fresh_integrator = make_state(npt=npt, cell_mode=cell_mode or "hydrostatic", seed=5, steps=1)
    fresh.protocol = StubProtocol([], 1.0, 0, 0.0)
    fresh.plan_revision = 0
    pointers = {name: getattr(fresh, name).data_ptr() for name in ("positions", "velocities", "cell", "charges")}
    restart_io.load_state(restart_io.read_restart(path), fresh, fresh_integrator)
    for name in ("positions", "velocities", "masses", "cell", "species", "forces", "energy", "charges"):
        assert torch.equal(getattr(fresh, name), getattr(state, name)), name
    assert {name: getattr(fresh, name).data_ptr() for name in pointers} == pointers  # in place
    if npt:
        for name in ("stress", "p_eps", "force_eps"):
            assert torch.equal(getattr(fresh, name), getattr(state, name)), name
        assert fresh_integrator.p_eps is fresh.p_eps
    assert fresh.step == state.step and fresh.plan_revision == 2
    assert fresh.protocol == state.protocol
    assert torch.equal(fresh.rng.batch, state.rng.batch) and fresh.rng.cursor == state.rng.cursor
    assert 0 < fresh.rng.cursor < 7
    # the generator continues with the same numbers as the original
    assert fresh.rng.generator.bit_generator.state == state.rng.generator.bit_generator.state
    for _ in range(10):  # across the end of the partly consumed batch
        integrator.prepare_step(state)
        fresh_integrator.prepare_step(fresh)
        assert torch.equal(fresh_integrator.xi, integrator.xi)
    # the force-field identity is the force field's, checked separately (check_identity)
    assert fresh.force_field_identity == identity()


def test_records_are_read_back(tmp_path):
    state, integrator = make_state()
    path = tmp_path / restart_io.RESTART_NAME
    write_full(path, state, integrator)
    stored = restart_io.read_restart(path)
    assert restart_io.format_version(stored) == 4
    assert restart_io.stored_identity(stored) == identity()
    assert restart_io.plan_record(stored) == plan_record()
    assert restart_io.protocol_record(stored) == state.protocol.to_restart()
    assert restart_io.output_indices(stored) == outputs()
    integrator_record = restart_io.integrator_record(stored)
    assert integrator_record["class"] == "LangevinBAOAB" and integrator_record["temperature_K"] == 300.0
    assert IntegratorConfig(**integrator_record["config"]) == integrator.config
    assert restart_io.physical_record(stored) == {"total_charge": 0.0, "spin_multiplicity": 1,
                                                  "pbc": [True, True, True], "constraints": [], "fixed_atoms": []}


def test_the_plan_record_holds_the_buckets(tmp_path):
    state, integrator = make_state()
    for missing in ("edge_bucket", "neighbour_capacity"):
        record = plan_record()
        del record[missing]
        with pytest.raises(ValueError, match=missing):
            restart_io.write_restart(tmp_path / "r.npz", state, engine="fastmace", integrator=integrator,
                                     variable_cell=True, plan=record)
    with pytest.raises(ValueError, match="JSON"):
        restart_io.write_restart(tmp_path / "r.npz", state, engine="fastmace", integrator=integrator,
                                 variable_cell=True, plan=plan_record(backend=object()))
    with pytest.raises(ValueError, match="output_rows"):
        restart_io.write_restart(tmp_path / "r.npz", state, engine="fastmace", integrator=integrator,
                                 variable_cell=True, plan=plan_record(), outputs={"output_chunk": 1})
    assert not (tmp_path / "r.npz").exists()


def test_a_protocol_without_a_stored_record_stays(tmp_path):
    """A state without a protocol object (main.py) writes none; the caller builds it from protocol_record."""
    state, integrator = make_state()
    state.protocol = None
    path = tmp_path / restart_io.RESTART_NAME
    write_full(path, state, integrator)
    stored = restart_io.read_restart(path)
    assert "protocol" not in stored and restart_io.protocol_record(stored) is None
    fresh, fresh_integrator = make_state(steps=1)
    restart_io.load_state(stored, fresh, fresh_integrator)
    assert fresh.protocol.step == 1  # unchanged


def test_load_refuses_another_structure(tmp_path):
    state, integrator = make_state()
    path = tmp_path / restart_io.RESTART_NAME
    write_full(path, state, integrator)
    stored = restart_io.read_restart(path)
    for change, match in ((dict(pbc=(True, True, False)), "pbc"), (dict(total_charge=1.0), "total_charge"),
                          (dict(spin_multiplicity=3), "spin_multiplicity")):
        other, other_integrator = make_state(steps=1)
        for name, value in change.items():
            setattr(other, name, value)
        with pytest.raises(ValueError, match=match):
            restart_io.load_state(stored, other, other_integrator)
    nvt, nvt_integrator = make_state(npt=False, steps=1)
    with pytest.raises(ValueError, match="p_eps"):
        restart_io.load_state(stored, nvt, nvt_integrator)


def test_a_newer_format_is_refused(tmp_path):
    state, integrator = make_state()
    path = tmp_path / restart_io.RESTART_NAME
    write_full(path, state, integrator)
    stored = dict(restart_io.read_restart(path))
    newer = restart_io.FORMAT_VERSION + 1
    stored["fastmace_restart"] = np.asarray(newer)
    with pytest.raises(ValueError, match=f"format version {newer}"):
        restart_io.load_state(stored, *make_state(steps=1))


def test_a_corrupted_identity_is_refused(tmp_path):
    state, integrator = make_state()
    path = tmp_path / restart_io.RESTART_NAME
    write_full(path, state, integrator)
    stored = dict(restart_io.read_restart(path))
    record = json.loads(str(stored["identity"]))
    record["head"] = "omol"
    stored["identity"] = np.asarray(json.dumps(record))
    with pytest.raises(ValueError, match="digest"):
        restart_io.stored_identity(stored)


# -------------------------------------------------------------------------- identity on resume

def test_check_identity(tmp_path):
    state, integrator = make_state()
    path = tmp_path / restart_io.RESTART_NAME
    write_full(path, state, integrator)
    stored = restart_io.read_restart(path)
    restart_io.check_identity(stored, identity())
    with pytest.raises(RestartMismatch, match="fastmace_version") as raised:
        restart_io.check_identity(stored, identity(fastmace_version="0.2.0"))
    assert isinstance(raised.value, FastMACEError)
    with pytest.raises(RestartMismatch, match="head.*species|species.*head"):
        restart_io.check_identity(stored, identity(head="omol", species=(11, 17, 35)))


def test_main_py_files_have_no_identity(tmp_path):
    """main.py's adapter writes state_arrays without identity or plan; Simulation.resume refuses them."""
    state, integrator = make_state()
    state.force_field_identity = None
    state.protocol = None
    arrays = restart_io.state_arrays(state, "fastmace", integrator.state_dict(), variable_cell=True)
    assert int(arrays["fastmace_restart"]) == restart_io.FORMAT_VERSION
    assert not {"identity", "identity_digest", "plan", "protocol", "output_chunk"} & set(arrays)
    for key in ("p_eps", "force_eps", "cell_state", "stress", "noise", "noise_next", "rng_state"):
        assert key in arrays, key  # research-engine keys, as main.py's adapter returns them
    path = tmp_path / restart_io.RESTART_NAME
    np.savez(path, **arrays)
    stored = restart_io.read_restart(path)
    assert restart_io.stored_identity(stored) is None
    with pytest.raises(RestartMismatch, match="no force-field identity"):
        restart_io.check_identity(stored, identity())
    fresh, fresh_integrator = make_state(steps=1)
    restart_io.load_state(stored, fresh, fresh_integrator)
    assert torch.equal(fresh.p_eps, state.p_eps) and torch.equal(fresh.force_eps, state.force_eps)


# -------------------------------------------------------------------------- older files

def _version_1_arrays(state):
    """A 1a restart file (FastMACE format version 1, io/restart.py at tag fastmace-1a)."""
    arrays = dict(step=state.step, positions=state.positions.numpy(), momenta=(state.velocities * state.masses).numpy(),
                  velocities=state.velocities.numpy(), masses=state.masses.numpy(), species=state.species.numpy(),
                  cell=state.cell.numpy(), engine="gpu", rng_state=json.dumps(state.rng.generator.bit_generator.state),
                  noise_next=state.rng.cursor, forces=state.forces.numpy(), energy=state.energy.numpy(),
                  fastmace_restart=1, noise=state.rng.batch.numpy())
    if state.stress is not None:
        arrays.update(stress=state.stress.numpy(), cell_state=state.cell.numpy(), p_eps=state.p_eps.numpy(),
                      force_eps=state.force_eps.numpy(), integrator_keys=np.array(["force_eps", "p_eps"], dtype=str))
    else:
        arrays["integrator_keys"] = np.array([], dtype=str)
    return arrays


@pytest.mark.parametrize("npt", [False, True])
def test_a_1a_restart_file_loads(tmp_path, npt):
    state, integrator = make_state(npt=npt)
    path = tmp_path / restart_io.RESTART_NAME
    np.savez(path, **_version_1_arrays(state))
    stored = restart_io.read_restart(path, engine="gpu")
    assert restart_io.format_version(stored) == 1 and restart_io.stored_identity(stored) is None
    fresh, fresh_integrator = make_state(npt=npt, seed=4, steps=1)
    fresh.plan_revision = 5
    restart_io.load_state(stored, fresh, fresh_integrator)
    names = ("positions", "velocities", "cell", "forces", "energy") + (("stress", "p_eps", "force_eps") if npt else ())
    for name in names:
        assert torch.equal(getattr(fresh, name), getattr(state, name)), name
    assert fresh.rng.cursor == state.rng.cursor and fresh.step == state.step
    assert fresh.plan_revision == 5  # not in the file: unchanged
    with pytest.raises(RestartMismatch, match="no force-field identity"):
        restart_io.check_identity(stored, identity())


def test_a_research_engine_npt_restart_file_loads(tmp_path):
    """workflow.write_restart with GPUEngine.state (no integrator_keys, no velocities, engine "gpu")."""
    state, integrator = make_state(cell_mode="full")
    path = tmp_path / restart_io.RESTART_NAME
    np.savez(path, step=state.step, positions=state.positions.numpy(),
             momenta=(state.velocities * state.masses).numpy(), cell=state.cell.numpy(), engine="gpu",
             rng_state=json.dumps(state.rng.generator.bit_generator.state), noise=state.rng.batch.numpy(),
             noise_next=state.rng.cursor, forces=state.forces.numpy(), energy=state.energy.numpy(),
             cell_state=state.cell.numpy(), p_eps=state.p_eps.numpy(), force_eps=state.force_eps.numpy(),
             stress=state.stress.numpy())
    stored = restart_io.read_restart(path, engine="gpu")
    assert restart_io.format_version(stored) == 0
    fresh, fresh_integrator = make_state(cell_mode="full", seed=4, steps=1)
    restart_io.load_state(stored, fresh, fresh_integrator)
    for name in ("positions", "cell", "forces", "energy", "stress", "p_eps", "force_eps"):
        assert torch.equal(getattr(fresh, name), getattr(state, name)), name
    masses = fresh.masses.numpy().reshape(-1)
    np.testing.assert_array_equal(fresh.velocities.numpy(),
                                  (state.velocities * state.masses).numpy() / masses[:, np.newaxis])


# -------------------------------------------------------------------------- the file itself

def test_the_file_is_a_plain_npz(tmp_path):
    """Readable by np.load without pickles; deterministic member order and time stamps."""
    state, integrator = make_state()
    path = tmp_path / restart_io.RESTART_NAME
    write_full(path, state, integrator)
    with np.load(path, allow_pickle=False) as stored:
        assert str(stored["engine"]) == "fastmace"
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
    assert [info.filename for info in infos] == sorted(info.filename for info in infos)
    assert {info.date_time for info in infos} == {(1980, 1, 1, 0, 0, 0)}


def test_restart_write_is_atomic(tmp_path, monkeypatch):
    state, integrator = make_state()
    path = tmp_path / restart_io.RESTART_NAME
    write_full(path, state, integrator)
    before = path.read_bytes()
    state.step += 1

    def failing(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(restart_io.np.lib.format, "write_array", failing)
    with pytest.raises(OSError, match="disk full"):
        write_full(path, state, integrator)
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [restart_io.RESTART_NAME]


def test_external_stress_units():
    """The config's GPa reach the integrator in ASE units (scalar and six Voigt components)."""
    state = SimulationState.from_atoms(nacl(repeat=1), "cpu", SEED, compute_stress=True)
    stressed = LangevinBAOAB(state, IntegratorConfig(ensemble="npt", cell_mode="full", friction=0.01,
                                                     external_stress_GPa=(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)))
    expected = np.array([[1.0, 0.3, 0.2], [0.3, 2.0, 0.1], [0.2, 0.1, 3.0]]) * units.GPa
    np.testing.assert_allclose(stressed.externalstress.numpy(), expected, rtol=1e-15)
