"""A resume under another MACE or FastMACE version warns; every other identity change refuses.

Plan task 2-T09; SPEC §10.6 as amended by owner decision 23: a version change alone warns and is
recorded (a plan revision and the report's warning); the reference record is compared by its
content without the version; any other force-field identity change raises ``RestartMismatch``.
Also 2-T21's G2: the resume reloads the model from the recorded file (path and SHA-256).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace import integrators
from fastmace.config import EngineConfig
from fastmace.engine import constraints
from fastmace.errors import RestartMismatch
from fastmace.io.output import Output
from fastmace.planner import minimal
from fastmace.reference import records
from fastmace.state import VERSION_FIELDS, ForceFieldIdentity

NA_CL = (11, 17)
LANGEVIN = integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01)


def identity(**changes):
    fields = dict(checkpoint_sha256="ab" * 32, head="omat_pbe", species=(11, 17), force_precision="float32",
                  physics={"charge": None, "spin_multiplicity": None}, reference_record_id=None,
                  mace_version="0.3.16", fastmace_version="0.2.0")
    fields.update(changes)
    return ForceFieldIdentity(**fields)


# --------------------------------------------------------------------------- the split (state.py)

def test_the_version_fields():
    assert VERSION_FIELDS == ("mace_version", "fastmace_version")


def test_version_changes_name_both_versions():
    stored = identity()
    now = identity(mace_version="0.3.17", fastmace_version="0.3.0")
    assert stored.version_changes(now) == {"mace_version": ("0.3.16", "0.3.17"),
                                           "fastmace_version": ("0.2.0", "0.3.0")}
    assert stored.version_changes(stored) == {}
    # Everything else is still a difference, and the digest still covers the versions.
    assert stored.differences(now) == ("mace_version", "fastmace_version")
    assert stored.digest != now.digest


def test_with_versions_of_takes_only_the_versions():
    stored = identity()
    now = identity(mace_version="0.3.17", force_precision="float64")
    aligned = now.with_versions_of(stored)
    assert aligned.mace_version == "0.3.16" and aligned.force_precision == "float64"
    assert stored.differences(aligned) == ("force_precision",)
    assert now.mace_version == "0.3.17"  # a new record; the old one is unchanged


def test_a_reference_record_counts_as_a_version_only_when_its_content_is_the_same():
    stored = identity(reference_record_id="ref/0.3.16/a")
    now = identity(reference_record_id="ref/0.3.17/a", mace_version="0.3.17")
    same = {"ref/0.3.16/a": "c1", "ref/0.3.17/a": "c1", "ref/0.3.17/b": "c2"}

    def same_reference(first, second):
        return same.get(first) is not None and same.get(first) == same.get(second)

    assert stored.version_changes(now, same_reference) == {
        "mace_version": ("0.3.16", "0.3.17"), "reference_record_id": ("ref/0.3.16/a", "ref/0.3.17/a")}
    # Without a way to compare the records, another id is another identity.
    assert "reference_record_id" not in stored.version_changes(now)
    other = identity(reference_record_id="ref/0.3.17/b", mace_version="0.3.17")
    assert "reference_record_id" not in stored.version_changes(other, same_reference)
    # A record that appears or disappears is never a version change.
    assert "reference_record_id" not in identity().version_changes(now, same_reference)


# --------------------------------------------------------------------------- the reference records

def _record(mace_version="0.3.16", **changes):
    record = records.record_for("0.3.16", "mh1")
    return dataclasses.replace(record, mace_version=mace_version, **changes)


def test_reference_content_drops_only_the_version():
    from fastmace.simulation import _reference_content_digest

    assert _record().id != _record("0.3.17").id  # the id carries the version
    assert _reference_content_digest(_record()) == _reference_content_digest(_record("0.3.17"))
    other = _record("0.3.17", numerical=())
    assert _reference_content_digest(other) != _reference_content_digest(_record())


def test_the_resume_finds_the_records_this_fastmace_builds():
    from fastmace.simulation import _same_reference

    record = records.record_for("0.3.16", "mh1")
    assert _same_reference(record.id, record.id)
    # 2-T04: another MACE gets the unqualified stock record; with the same content it is the same
    # reference up to the version (decision 23).
    assert _same_reference(record.id, records.record_for("0.3.17", "mh1").id)
    # An id this FastMACE cannot rebuild cannot be compared, so it is not the same record.
    assert not _same_reference(record.id, _record("0.3.17", stock={**record.stock, "backend": "e3nn"}).id)
    assert not _same_reference(record.id, "fastmace.reference/0.3.16/mh1/0000000000000000")
    assert not _same_reference(record.id, "not an id")


def test_resume_identity_warns_on_versions_and_refuses_the_rest(monkeypatch):
    from fastmace import simulation
    from fastmace.io import restart

    stored = identity()
    monkeypatch.setattr(restart, "stored_identity", lambda file: stored)
    assert simulation._resume_identity({}, stored) == {}
    assert simulation._resume_identity({}, identity(fastmace_version="0.3.0")) == {
        "fastmace_version": ("0.2.0", "0.3.0")}
    for change, name in ((dict(checkpoint_sha256="cd" * 32), "checkpoint_sha256"), (dict(head="omol"), "head"),
                         (dict(species=(11, 17, 35)), "species"), (dict(force_precision="float64"), "force_precision"),
                         (dict(physics={"charge": 0, "spin_multiplicity": 1}), "physics"),
                         (dict(reference_record_id="ref-2"), "reference_record_id")):
        with pytest.raises(RestartMismatch, match=name) as raised:
            simulation._resume_identity({}, identity(mace_version="0.3.17", fastmace_version="0.3.0", **change))
        assert "mace_version" not in str(raised.value)  # the versions alone never refuse


def test_resume_identity_compares_the_reference_records_by_content(monkeypatch):
    from fastmace import simulation
    from fastmace.io import restart

    old, new, other = _record(), _record("0.3.17"), _record("0.3.17", numerical=())
    known = {record.id: record for record in (old, new, other)}
    monkeypatch.setattr(simulation, "_known_reference_record", known.get)
    stored = identity(reference_record_id=old.id)
    monkeypatch.setattr(restart, "stored_identity", lambda file: stored)
    changes = simulation._resume_identity({}, identity(reference_record_id=new.id, mace_version="0.3.17"))
    assert changes == {"mace_version": ("0.3.16", "0.3.17"), "reference_record_id": (old.id, new.id)}
    with pytest.raises(RestartMismatch, match="reference_record_id"):
        simulation._resume_identity({}, identity(reference_record_id=other.id, mace_version="0.3.17"))


# --------------------------------------------------------------------------- Simulation.resume

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
    planner = minimal.MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ={},
                                     process=constraints.ProcessRecord())
    monkeypatch.setattr(minimal, "default_planner", lambda force_field: planner)
    return planner


def _simulation(loaded, output=None):
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    return Simulation(atoms, ForceField(loaded, atoms, precision="float64"), integrator=LANGEVIN, output=output,
                      seed=1)


def _upgrade(monkeypatch):
    import fastmace

    monkeypatch.setattr(fastmace, "__version__", "9.9.9")


@pytest.mark.filterwarnings("ignore::fastmace.model.loading.ModelLicenceWarning")
def test_a_version_only_change_warns_records_both_and_continues(loaded, tmp_path, monkeypatch):
    import fastmace
    from fastmace.simulation import Simulation

    before = fastmace.__version__
    whole = _simulation(loaded)
    whole.run(steps=4)
    sim = _simulation(loaded, output=Output(directory=str(tmp_path)))
    sim.run(steps=2)
    sim.save_restart(tmp_path / "restart.npz")
    sim.close()
    _upgrade(monkeypatch)
    with pytest.warns(UserWarning, match=rf"fastmace_version {before!r} in the file, '9\.9\.9' now"):
        resumed = Simulation.resume(tmp_path)
    plan = resumed.plan
    assert plan.revision == 1 and plan.predecessor.revision == 0 and plan.step == 2
    assert before in plan.reason and "9.9.9" in plan.reason
    assert resumed._engine.state.plan_revision == 1
    assert any(before in line and "9.9.9" in line for line in resumed._report.warnings())
    resumed.run(steps=2)
    resumed.close()
    # Same physics, same plan fields: the continuation is the uninterrupted run.
    np.testing.assert_array_equal(resumed.committed().positions, whole.committed().positions)
    metadata = json.loads((Path(tmp_path) / "metadata.json").read_text())
    assert metadata["force_field_identity"]["fastmace_version"] == "9.9.9"
    revisions = json.dumps(metadata["plan_revisions"])
    assert before in revisions and "9.9.9" in revisions
    assert any(before in line and "9.9.9" in line for line in metadata["report"]["warnings"])


def test_the_same_versions_resume_silently(loaded, tmp_path, recwarn):
    from fastmace.simulation import Simulation

    sim = _simulation(loaded)
    sim.run(steps=1)
    sim.save_restart(tmp_path / "restart.npz")
    recwarn.clear()
    resumed = Simulation.resume(tmp_path)
    assert resumed.plan.revision == 0
    assert not [w for w in recwarn if "version" in str(w.message)]


def test_a_version_and_a_physics_change_still_refuse(loaded, tmp_path, monkeypatch):
    from fastmace.simulation import Simulation

    sim = _simulation(loaded)
    sim.run(steps=1)
    sim.save_restart(tmp_path / "restart.npz")
    path = tmp_path / "simulation.json"
    record = json.loads(path.read_text())
    record["force_field"]["precision"] = "float32"
    path.write_text(json.dumps(record))
    _upgrade(monkeypatch)
    with pytest.raises(RestartMismatch, match="force_precision") as raised:
        Simulation.resume(tmp_path)
    assert "fastmace_version" not in str(raised.value)


# --------------------------------------------------------------------------- G2: the recorded file

def test_a_label_reloads_from_its_recorded_file(monkeypatch, tmp_path):
    from fastmace import simulation
    from fastmace.model import loading

    calls = []
    monkeypatch.setattr(loading, "load_pristine", lambda *args, **kwargs: calls.append((args, kwargs)))
    file = tmp_path / "local-copy.model"
    file.write_bytes(b"x")
    source = {"label": "mh1-r2scan", "path": str(file), "head": "r2scan", "sha256": "ab" * 32, "trust": "trusted",
              "charge": None, "spin_multiplicity": None}
    simulation._load_model(source)
    assert calls[-1] == (("mh1-r2scan",), dict(checkpoint=str(file), sha256="ab" * 32, charge=None,
                                                spin_multiplicity=None))
    # A run moved to another machine: the label finds its file there, still checked by the hash.
    file.unlink()
    simulation._load_model(source)
    assert calls[-1] == (("mh1-r2scan",), dict(sha256="ab" * 32, charge=None, spin_multiplicity=None))
    path_source = {**source, "label": None, "head": None}
    simulation._load_model(path_source)
    assert calls[-1] == ((str(file), None), dict(sha256="ab" * 32, charge=None, spin_multiplicity=None))
