"""A protocol run takes a ``steps`` key from its file (fastmace run with [relax] or --steps)."""

import pytest
from ase.build import bulk

from fastmace import quick
from fastmace.model import loading
from fastmace.protocols import runner


class _Stop(Exception):
    pass


@pytest.fixture
def md_steps(monkeypatch):
    seen = {}

    def fake_md(atoms, **kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(loading, "load_pristine", lambda *args, **kwargs: object())
    monkeypatch.setattr(quick, "md", fake_md)
    return seen


@pytest.mark.parametrize(("file_steps", "flag_steps", "expected"), [(40, None, 40), (40, 10, 10), (5, 10, 5)])
def test_a_steps_key_reaches_fm_md_once(tmp_path, md_steps, file_steps, flag_steps, expected):
    options = {"model": "mh1", "temperature_K": 300.0, "steps": file_steps, "output": str(tmp_path / "run")}
    with pytest.raises(_Stop):
        runner.start(bulk("NaCl", "rocksalt", a=5.64), options, steps=flag_steps, log=lambda message: None)
    assert md_steps["steps"] == expected


@pytest.mark.parametrize(("file_steps", "flag_steps", "recorded"), [(40, 10, 40), (None, 10, 10)])
def test_without_a_schedule_the_record_keeps_the_run_length(tmp_path, md_steps, file_steps, flag_steps, recorded):
    options = {"model": "mh1", "temperature_K": 300.0, "output": str(tmp_path / "run")}
    if file_steps is not None:
        options["steps"] = file_steps
    with pytest.raises(_Stop):
        runner.start(bulk("NaCl", "rocksalt", a=5.64), options, steps=flag_steps, log=lambda message: None)
    record = runner.read_record(tmp_path / "run")
    assert record["md_steps"] == recorded
    assert runner._unscheduled_left(record, 10) == recorded - 10
    assert runner._unscheduled_left({"md_steps": None}, 10) == 0
