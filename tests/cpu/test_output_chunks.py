"""Output chunks at restart points (plan task 1b-T06; SPEC §10.6 "Output files at restart points",
§8.1 async output, §12.2 restart row).

Every restart write closes the current ``thermo.<n>.csv`` and ``trajectory.<n>.traj`` and stores
``output_chunk``, ``output_rows`` and ``output_frames`` in the restart file; resume deletes the
chunks after it and opens new ones, and never rewrites a file. The runs are the fake schedule of
``_support.chunk_run`` (1a-T14's), plus one run of the CPU graph engine with a tiny MACE model.
"""

from __future__ import annotations

import os
import signal
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.io import read

from _support import chunk_run
from fastmace.errors import RestartMismatch
from fastmace.io import output as output_io
from fastmace.io import restart as restart_io

ALL_STEPS = [0, *range(2, chunk_run.TOTAL_STEPS + 1, 2)]


def files(directory):
    return {path.name: path.read_bytes() for path in sorted(Path(directory).iterdir())
            if path.name != restart_io.RESTART_NAME}


def steps_of(directory):
    header, rows = output_io.read_thermo(directory)
    assert header == chunk_run.HEADER or header == chunk_run.NPT_HEADER
    return [int(row[0]) for row in rows]


def concatenated(directory):
    """The thermo rows and the trajectory frames over all chunks, in order."""
    _, rows = output_io.read_thermo(directory)
    frames = output_io.read_frames(directory)
    return rows, [(f.positions.tobytes(), f.get_momenta().tobytes(), f.get_potential_energy()) for f in frames]


# --------------------------------------------------------------------------- chunk layout


@pytest.mark.parametrize("variable_cell", [False, True])
def test_synchronous_and_asynchronous_writers_give_the_same_chunks(tmp_path, variable_cell):
    chunk_run.run(tmp_path / "sync", asynchronous=False, variable_cell=variable_cell)
    chunk_run.run(tmp_path / "async", asynchronous=True, variable_cell=variable_cell)
    sync, asynchronous = files(tmp_path / "sync"), files(tmp_path / "async")
    assert sync == asynchronous
    # Restarts at 4, 8, 12 and 16: five chunks, each ending exactly at a restart step.
    assert sorted(sync) == sorted([f"thermo.{n}.csv" for n in range(5)] + [f"trajectory.{n}.traj" for n in range(5)])
    for n, last in enumerate((4, 8, 12, 16, 20)):
        lines = (tmp_path / "sync" / f"thermo.{n}.csv").read_text().splitlines()[1:]
        steps = [int(line.split(",")[0]) for line in lines]
        assert steps[-1] == last and all(last - 4 < s <= last for s in steps if n)
        assert len(read(tmp_path / "sync" / f"trajectory.{n}.traj", index=":")) == len(steps)


def test_every_chunk_has_the_header(tmp_path):
    chunk_run.run(tmp_path, asynchronous=False)
    for n in range(5):
        assert (tmp_path / f"thermo.{n}.csv").read_text().splitlines()[0] == ",".join(chunk_run.HEADER)


def test_ase_read_over_the_chunks_gives_every_frame(tmp_path):
    chunk_run.run(tmp_path, asynchronous=True)
    frames = []
    for n in output_io.chunk_indices(tmp_path):
        frames += read(tmp_path / output_io.chunk_paths(tmp_path, n)[1], index=":")
    assert len(frames) == len(ALL_STEPS)
    assert [f.positions.tobytes() for f in frames] == [f.positions.tobytes() for f in output_io.read_frames(tmp_path)]
    assert steps_of(tmp_path) == ALL_STEPS


def test_restart_indices_are_cumulative_and_stored(tmp_path):
    chunk_run.run(tmp_path, asynchronous=False, stop_at=10)
    stored = restart_io.read_restart(tmp_path / restart_io.RESTART_NAME)
    # Restarts at 4 and 8, then the stop at 10 (a restart point too): chunk 2 holds steps 10 only.
    assert restart_io.output_indices(stored) == {"output_chunk": 2, "output_rows": 6, "output_frames": 6}
    # The next chunk is open and empty: header only, a trajectory without frames.
    assert (tmp_path / "thermo.3.csv").read_text().splitlines() == [",".join(chunk_run.HEADER)]
    assert read(tmp_path / "trajectory.3.traj", index=":") == []


def test_an_empty_chunk_has_both_files(tmp_path):
    """A restart cadence shorter than the output cadence gives chunks without rows or frames."""
    atoms, state = chunk_run.make_state()
    with output_io.ChunkedOutput(tmp_path, atoms, chunk_run.HEADER) as writer:
        writer.write(chunk_run._sync(state), sample=chunk_run.prefix(0), frame=True)
        first = writer.restart_point()
        second = writer.restart_point()
    assert first == {"output_chunk": 0, "output_rows": 1, "output_frames": 1}
    assert second == {"output_chunk": 1, "output_rows": 1, "output_frames": 1}
    assert output_io.chunk_indices(tmp_path) == [0, 1, 2]
    assert (tmp_path / "thermo.1.csv").read_text().splitlines() == [",".join(chunk_run.HEADER)]
    assert read(tmp_path / "trajectory.1.traj", index=":") == []


def test_a_new_run_refuses_a_directory_with_chunks(tmp_path):
    chunk_run.run(tmp_path, asynchronous=False)
    atoms, _ = chunk_run.make_state()
    with pytest.raises(FileExistsError, match="thermo.0.csv"):
        output_io.ChunkedOutput(tmp_path, atoms, chunk_run.HEADER)


def test_output_steps_is_reexported_from_protocols():
    """main.py's adapter imports io.trajectory.output_steps; the canonical copy is fastmace.protocols' (1b-T07)."""
    from fastmace import protocols
    from fastmace.io import trajectory

    assert trajectory.output_steps is protocols.output_steps


# --------------------------------------------------------------------------- resume


@pytest.mark.parametrize("asynchronous", [False, True])
def test_resume_deletes_exactly_the_later_chunks(tmp_path, asynchronous):
    with pytest.raises(chunk_run.Crash):
        chunk_run.run(tmp_path, asynchronous=asynchronous, crash_in=11)
    before = files(tmp_path)
    assert output_io.chunk_indices(tmp_path) == [0, 1, 2]  # chunk 2 holds step 10 (restart at 8)
    atoms, _ = chunk_run.make_state()
    indices = restart_io.output_indices(restart_io.read_restart(tmp_path / restart_io.RESTART_NAME))
    assert indices["output_chunk"] == 1
    with output_io.ChunkedOutput(tmp_path, atoms, chunk_run.HEADER, resume_from=indices):
        after = files(tmp_path)
    # Chunks 0 and 1 untouched (not rewritten), chunk 2 replaced by a new, empty one.
    for name in ("thermo.0.csv", "thermo.1.csv", "trajectory.0.traj", "trajectory.1.traj"):
        assert after[name] == before[name]
    assert after["thermo.2.csv"].decode().splitlines() == [",".join(chunk_run.HEADER)]
    assert read(tmp_path / "trajectory.2.traj", index=":") == []


def test_resume_never_rewrites_a_kept_chunk(tmp_path):
    with pytest.raises(chunk_run.Crash):
        chunk_run.run(tmp_path, asynchronous=True, crash_in=11)
    kept = [tmp_path / name for name in ("thermo.0.csv", "thermo.1.csv", "trajectory.0.traj", "trajectory.1.traj")]
    stamps = {path: (os.stat(path).st_ino, os.stat(path).st_mtime_ns) for path in kept}
    chunk_run.run(tmp_path, asynchronous=True, resume=True)
    assert {path: (os.stat(path).st_ino, os.stat(path).st_mtime_ns) for path in kept} == stamps


@pytest.mark.parametrize("asynchronous", [False, True])
def test_interrupted_run_equals_uninterrupted(tmp_path, asynchronous):
    """Same cadence: the concatenated rows and frames equal the uninterrupted run's, bit for bit."""
    chunk_run.run(tmp_path / "whole", asynchronous=asynchronous)
    with pytest.raises(chunk_run.Crash):
        chunk_run.run(tmp_path / "parts", asynchronous=asynchronous, crash_in=11)
    chunk_run.run(tmp_path / "parts", asynchronous=asynchronous, resume=True)
    assert concatenated(tmp_path / "parts") == concatenated(tmp_path / "whole")
    assert files(tmp_path / "parts") == files(tmp_path / "whole")
    assert steps_of(tmp_path / "parts") == ALL_STEPS


def test_resume_after_a_stop_request(tmp_path):
    chunk_run.run(tmp_path / "whole", asynchronous=True)
    chunk_run.run(tmp_path / "parts", asynchronous=True, stop_at=10)
    chunk_run.run(tmp_path / "parts", asynchronous=True, resume=True)
    assert concatenated(tmp_path / "parts") == concatenated(tmp_path / "whole")
    assert steps_of(tmp_path / "parts") == ALL_STEPS


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="needs SIGKILL")
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("kill_in", [11, 12])
def test_crash_between_restart_and_flush(tmp_path, asynchronous, kill_in):
    """SPEC §12.2: a child killed after the step-8 restart and before the next one resumes without
    duplicate or missing rows and frames. Killed in step 11, the asynchronous run still holds step
    10's snapshot unwritten; killed in step 12, chunk 2 holds step 10 on disk."""
    chunk_run.run(tmp_path / "whole", asynchronous=asynchronous)
    parts = tmp_path / "parts"
    child = subprocess.run([sys.executable, chunk_run.__file__, str(parts), str(kill_in), str(int(asynchronous))],
                           capture_output=True, text=True, timeout=300)
    assert child.returncode == -signal.SIGKILL, child.stderr
    assert (10 in steps_of(parts)) == (kill_in == 12 or not asynchronous)
    chunk_run.run(parts, asynchronous=asynchronous, resume=True)
    steps = steps_of(parts)
    assert steps == ALL_STEPS and len(set(steps)) == len(steps)
    assert len(output_io.read_frames(parts)) == len(ALL_STEPS)
    assert concatenated(parts) == concatenated(tmp_path / "whole")


def test_crash_between_restart_point_and_restart_write(tmp_path):
    """Chunks closed but the restart file not yet replaced: resume from the older restart deletes them."""
    chunk_run.run(tmp_path / "whole", asynchronous=False)
    parts = tmp_path / "parts"
    chunk_run.run(parts, asynchronous=False, stop_at=8)
    atoms, state = chunk_run.make_state()
    # The step-8 restart closed chunk 1 and opened chunk 2; run to 12, close chunk 2, crash before writing.
    stored = restart_io.read_restart(parts / restart_io.RESTART_NAME)
    restart_io.load_state(stored, state)
    with output_io.ChunkedOutput(parts, atoms, chunk_run.HEADER,
                                 resume_from=restart_io.output_indices(stored)) as writer:
        for step in range(9, 13):
            chunk_run._step(state)
            if step % 2 == 0:
                writer.write(chunk_run._sync(state), sample=chunk_run.prefix(step), frame=True)
        writer.restart_point()
    assert output_io.chunk_indices(parts) == [0, 1, 2, 3]
    chunk_run.run(parts, asynchronous=False, resume=True)
    assert concatenated(parts) == concatenated(tmp_path / "whole")


# --------------------------------------------------------------------------- refusals


def crashed(tmp_path):
    with pytest.raises(chunk_run.Crash):
        chunk_run.run(tmp_path, asynchronous=False, crash_in=11)
    return restart_io.output_indices(restart_io.read_restart(tmp_path / restart_io.RESTART_NAME))


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0,
                    reason="needs POSIX permissions and a non-root user")
def test_read_only_directory_refuses_and_touches_nothing(tmp_path):
    indices = crashed(tmp_path)
    before = files(tmp_path)
    atoms, _ = chunk_run.make_state()
    mode = tmp_path.stat().st_mode
    tmp_path.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        with pytest.raises(RestartMismatch, match="cannot delete"):
            output_io.ChunkedOutput(tmp_path, atoms, chunk_run.HEADER, resume_from=indices)
    finally:
        tmp_path.chmod(mode)
    assert files(tmp_path) == before


@pytest.mark.parametrize("missing", ["thermo.0.csv", "trajectory.1.traj"])
def test_missing_chunk_refuses_and_touches_nothing(tmp_path, missing):
    indices = crashed(tmp_path)
    (tmp_path / missing).unlink()
    before = files(tmp_path)
    atoms, _ = chunk_run.make_state()
    with pytest.raises(RestartMismatch, match=missing):
        output_io.ChunkedOutput(tmp_path, atoms, chunk_run.HEADER, resume_from=indices)
    assert files(tmp_path) == before


def test_truncated_chunk_refuses(tmp_path):
    indices = crashed(tmp_path)
    path = tmp_path / "thermo.1.csv"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    before = files(tmp_path)
    atoms, _ = chunk_run.make_state()
    with pytest.raises(RestartMismatch, match="rows"):
        output_io.ChunkedOutput(tmp_path, atoms, chunk_run.HEADER, resume_from=indices)
    assert files(tmp_path) == before


def test_restart_without_chunk_indices_refuses(tmp_path):
    crashed(tmp_path)
    atoms, _ = chunk_run.make_state()
    with pytest.raises(RestartMismatch, match="output chunk"):
        output_io.ChunkedOutput(tmp_path, atoms, chunk_run.HEADER, resume_from={})


# --------------------------------------------------------------------------- the graph engine


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_graph_engine_run_resumes_without_duplicate_or_missing_output(tmp_path, monkeypatch):
    """One run of the CPU graph engine with a tiny MACE model (pipelined, asynchronous snapshots):
    killed in chunk 2 after the step-8 restart, resumed; rows and frames are those of every output step.
    Values are not compared: bitwise continuation is plan task 1b-T14's (canonical restart points)."""
    pytest.importorskip("cuequivariance_torch")
    from _support.structures import nacl
    from _support.tiny_models import FAMILIES, tiny_mace_model
    from fastmace.config import EngineConfig, IntegratorConfig
    from fastmace.engine import constraints
    from fastmace.io.trajectory import AsyncOutput
    from fastmace.model import loading
    from fastmace.planner.fixed import FixedPlanner
    from fastmace.planner.plan import Hardware

    monkeypatch.setattr(loading, "_LICENCES_SHOWN", set())
    checkpoint = tmp_path / "tiny.model"
    torch.save(tiny_mace_model([11, 17], **FAMILIES["mh1"]), checkpoint)
    planner = FixedPlanner(engine=EngineConfig(device="cpu", backend="cueq", compile_mode="none"), environ={})
    loaded = planner.load(str(checkpoint), trusted=True)
    atoms = nacl(rattle=0.05, seed=1)
    plan = planner.plan(loaded, atoms, IntegratorConfig(), Hardware("cpu", (0, 0), 0))
    out = tmp_path / "run"
    out.mkdir()
    restart_path = out / restart_io.RESTART_NAME
    total, every, restart_every = 16, 2, 4

    def run(crash_in=None):
        engine = planner.build(plan, loaded, atoms, temperature_K=300.0, constraints=constraints.ProcessRecord())
        indices = None
        if restart_path.is_file():
            stored = restart_io.read_restart(restart_path, engine="gpu")
            indices = restart_io.output_indices(stored)
            restart_io.load_state(stored, engine.state)
        start = engine.state.step
        with output_io.ChunkedOutput(out, atoms, chunk_run.HEADER, resume_from=indices) as writer:
            queue = AsyncOutput(writer)
            if not start:
                writer.write(engine.sync(), sample=(0, 0.0, "heating", 300.0), frame=True)
            for step in range(start + 1, total + 1):
                if step == crash_in:
                    return
                engine.step()
                queue.drain()
                if step % every == 0:
                    queue.submit(engine.snapshot(), sample=(step, step * 5e-4, "heating", 300.0), frame=True)
                if step % restart_every == 0 and step < total:
                    engine.flush()
                    outputs = queue.restart_point()
                    restart_io.write_restart(restart_path, engine.state, engine="gpu", outputs=outputs)
            queue.drain()

    run(crash_in=12)  # abandoned in step 12: chunk 2 holds step 10
    assert 10 in steps_of(out)
    run()
    steps = steps_of(out)
    assert steps == list(range(0, total + 1, every))
    frames = output_io.read_frames(out)
    assert len(frames) == len(steps) and all(np.isfinite(f.get_potential_energy()) for f in frames)
