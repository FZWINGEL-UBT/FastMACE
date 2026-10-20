"""Durable output and restart files (plan task 2-T10; SPEC §10.6).

A fake ``os.fsync`` and ``os.replace`` record the calls: each fsync is named by the file or
directory its descriptor refers to at that moment, so the tests check the order on disk: the closed
chunk files, their directory, the restart's temporary file, the rename, the restart's directory.
"""

from __future__ import annotations

import errno
import os
import sys
from pathlib import Path

import pytest

from _support import chunk_run
from fastmace import provenance
from fastmace.io import durable
from fastmace.io import output as output_io
from fastmace.io import restart as restart_io

POSIX = os.name == "posix"
DIRECTORY = "<dir>"
needs_posix = pytest.mark.skipif(not POSIX, reason="directory fsync is POSIX only")


class Recorder:
    """Records fsync and replace calls on files in ``directory`` (fsyncs elsewhere are named by path)."""

    def __init__(self, monkeypatch, directory):
        self.directory = Path(directory)
        self.events = []
        self.fail = {}  # name -> errno raised by the fsync of that file or directory
        real_fsync, real_replace = os.fsync, os.replace

        def fsync(fd):
            name = self._name(os.fstat(fd))
            self.events.append(("fsync", name))
            if name in self.fail:
                raise OSError(self.fail[name], os.strerror(self.fail[name]))
            real_fsync(fd)

        def replace(source, target):
            self.events.append(("replace", Path(source).name, Path(target).name))
            real_replace(source, target)

        monkeypatch.setattr(os, "fsync", fsync)
        monkeypatch.setattr(os, "replace", replace)

    def _name(self, stat):
        if os.path.samestat(stat, os.stat(self.directory)):
            return DIRECTORY
        for path in self.directory.iterdir():
            if os.path.samestat(stat, os.stat(path)):
                return path.name
        return "?"


def directory_fsync():
    return [("fsync", DIRECTORY)] if POSIX else []


# --------------------------------------------------------------------------- atomic_write


@pytest.mark.parametrize("data", [b"\x00bytes\n", "text é\n"])
def test_atomic_write_fsyncs_the_temporary_file_renames_and_fsyncs_the_directory(tmp_path, monkeypatch, data):
    recorder = Recorder(monkeypatch, tmp_path)
    path = tmp_path / "record.json"
    assert durable.atomic_write(path, data) == path
    temporary = f".record.json.{os.getpid()}.tmp"
    assert recorder.events == [("fsync", temporary), ("replace", temporary, "record.json"), *directory_fsync()]
    assert path.read_bytes() == (data.encode("utf-8") if isinstance(data, str) else data)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["record.json"]


def test_atomic_write_replaces_an_existing_file(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("old")
    durable.atomic_write(path, "new")
    assert path.read_text() == "new"


def test_a_failed_file_fsync_leaves_the_old_file_and_no_temporary(tmp_path, monkeypatch):
    path = tmp_path / "a.txt"
    path.write_text("old")
    recorder = Recorder(monkeypatch, tmp_path)
    recorder.fail[f".a.txt.{os.getpid()}.tmp"] = errno.EIO
    with pytest.raises(OSError) as raised:
        durable.atomic_write(path, "new")
    assert raised.value.errno == errno.EIO
    assert path.read_text() == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.txt"]
    assert not any(event[0] == "replace" for event in recorder.events)


@needs_posix
@pytest.mark.parametrize("code", [errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP])
def test_a_directory_without_fsync_is_skipped(tmp_path, monkeypatch, code):
    recorder = Recorder(monkeypatch, tmp_path)
    recorder.fail[DIRECTORY] = code
    durable.atomic_write(tmp_path / "a.txt", "x")
    assert (tmp_path / "a.txt").read_text() == "x"
    assert recorder.events[-1] == ("fsync", DIRECTORY)


@needs_posix
def test_any_other_directory_fsync_error_raises(tmp_path, monkeypatch):
    recorder = Recorder(monkeypatch, tmp_path)
    recorder.fail[DIRECTORY] = errno.EIO
    with pytest.raises(OSError) as raised:
        durable.atomic_write(tmp_path / "a.txt", "x")
    assert raised.value.errno == errno.EIO


def test_einval_on_a_file_fsync_raises(tmp_path, monkeypatch):
    """Only the directory fsync may be skipped: a file that cannot be synced is an error."""
    recorder = Recorder(monkeypatch, tmp_path)
    path = tmp_path / "a.txt"
    path.write_text("x")
    recorder.fail["a.txt"] = errno.EINVAL
    with pytest.raises(OSError):
        durable.fsync_file(path)


@pytest.mark.skipif(POSIX, reason="Windows only")
def test_the_directory_fsync_is_a_no_op_on_windows(tmp_path, monkeypatch):
    recorder = Recorder(monkeypatch, tmp_path)
    durable.fsync_directory(tmp_path)
    assert recorder.events == []


# --------------------------------------------------------------------------- chunks and restart


def restart_point_events(n):
    """A restart point after chunk ``n``: the chunks, their directory, then the restart file."""
    temporary = restart_io.RESTART_NAME + ".tmp.npz"
    return [("fsync", f"thermo.{n}.csv"), ("fsync", f"trajectory.{n}.traj"), *directory_fsync(),
            ("fsync", temporary), ("replace", temporary, restart_io.RESTART_NAME), *directory_fsync()]


def close_events(n):
    return [("fsync", f"thermo.{n}.csv"), ("fsync", f"trajectory.{n}.traj"), *directory_fsync()]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_the_restart_sequence_fsyncs_chunks_then_the_restart_then_the_directory(tmp_path, monkeypatch, asynchronous):
    recorder = Recorder(monkeypatch, tmp_path)
    chunk_run.run(tmp_path, asynchronous=asynchronous, stop_at=10)
    # Restart points at 4 and 8, then the stop at 10; leaving the run closes chunk 3.
    expected = restart_point_events(0) + restart_point_events(1) + restart_point_events(2) + close_events(3)
    assert recorder.events == expected


def test_nothing_is_fsynced_between_restart_points(tmp_path, monkeypatch):
    """Rows and frames are flushed, not fsynced: the cost is paid once per restart point."""
    recorder = Recorder(monkeypatch, tmp_path)
    chunk_run.run(tmp_path, asynchronous=False, restart_every=None)
    assert recorder.events == close_events(0)


def test_resume_from_a_durable_restart_is_unchanged(tmp_path):
    chunk_run.run(tmp_path / "plain", asynchronous=False)
    chunk_run.run(tmp_path / "resumed", asynchronous=False, stop_at=10)
    chunk_run.run(tmp_path / "resumed", asynchronous=False, resume=True)
    assert output_io.read_thermo(tmp_path / "plain") == output_io.read_thermo(tmp_path / "resumed")


# --------------------------------------------------------------------------- JSON records


def test_write_metadata_is_durable(tmp_path, monkeypatch):
    recorder = Recorder(monkeypatch, tmp_path)
    provenance.write_metadata(tmp_path / "metadata.json", {"schema": provenance.SCHEMA})
    temporary = f".metadata.json.{os.getpid()}.tmp"
    assert recorder.events == [("fsync", temporary), ("replace", temporary, "metadata.json"), *directory_fsync()]


def test_simulation_json_is_durable(tmp_path, monkeypatch):
    from fastmace import simulation

    recorder = Recorder(monkeypatch, tmp_path)
    simulation._write_json(tmp_path / simulation.SIMULATION_NAME, {"b": 1, "a": [1.5]})
    name = simulation.SIMULATION_NAME
    temporary = f".{name}.{os.getpid()}.tmp"
    assert recorder.events == [("fsync", temporary), ("replace", temporary, name), *directory_fsync()]
    assert (tmp_path / name).read_bytes() == b'{\n  "a": [\n    1.5\n  ],\n  "b": 1\n}\n'


def test_durable_imports_nothing_heavy():
    """``fm.Output`` is a light name; ``io.output`` may import ``io.durable`` at import time."""
    import subprocess
    code = "import fastmace.io.durable, fastmace.io.output, sys; print('torch' in sys.modules, 'ase' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.split() == ["False", "False"]
