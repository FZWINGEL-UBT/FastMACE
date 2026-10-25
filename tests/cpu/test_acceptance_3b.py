"""Acceptance tools of plan task 3b-T10, without a GPU.

- ``tools/ab_fastmace.py --release 3`` and ``tools/trajectory_versions.py --against 3 [--ensemble npt]``: this
  checkout (side 3b, D3 off) against the ``fastmace-3`` tag, item 5. The tag was not set when they were written,
  so the release side needs ``--expected-commit``.
- ``tools/acceptance/resultlog.py`` and ``stoppable_pytest.sh``: a stopped pytest run keeps its per-test results
  and writes its junit file (phase-3.md, open finding 7).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOLS = REPOSITORY_ROOT / "tools"
ACCEPTANCE = TOOLS / "acceptance"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(TOOLS))
ab_fastmace = _load(TOOLS / "ab_fastmace.py", "_tool_ab_fastmace_3bt10")
trajectory_versions = _load(TOOLS / "trajectory_versions.py", "_tool_trajectory_versions_3bt10")


# --- tools/ab_fastmace.py --release 3 -------------------------------------------------------------------------


def _args(tmp_path, *extra):
    return ["--output", str(tmp_path / "out"), "--checkpoint", "m", "--renamed", str(tmp_path / "v"), *extra]


def test_release_3_is_the_fastmace_3_tag_with_side_3b(tmp_path):
    release = ab_fastmace.RELEASES["3"]
    assert (release.tag, release.commit, release.renamed, release.planner, release.current) == (
        "fastmace-3", None, "fastmace_3", "minimal", "3b")
    args = ab_fastmace.parse_args(_args(tmp_path, "--release", "3", "--expected-commit", "abc1234"))
    assert args.sides == ["3", "3b"] and args.current == "3b" and args.expected_commit == "abc1234"
    assert args.integrator_options == {}  # fastmace-3 already has decision 37's projection
    assert ab_fastmace.Version("3b", args, None).minimal and ab_fastmace.Version("3", args, None).minimal


def test_release_3_needs_the_tag_commit_for_its_own_side(tmp_path):
    with pytest.raises(SystemExit):
        ab_fastmace.parse_args(_args(tmp_path, "--release", "3"))
    args = ab_fastmace.parse_args(_args(tmp_path, "--release", "3", "--sides", "3b"))  # this checkout alone
    assert args.sides == ["3b"] and args.expected_commit is None


def test_release_3_takes_only_its_sides(tmp_path):
    for sides in (("3", "p3b"), ("2", "3b"), ("3b", "3b")):
        with pytest.raises(SystemExit):
            ab_fastmace.parse_args(_args(tmp_path, "--release", "3", "--expected-commit", "abc", "--sides", *sides))
    args = ab_fastmace.parse_args(_args(tmp_path, "--release", "3", "--expected-commit", "abc", "--sides", "3b", "3"))
    assert args.sides == ["3b", "3"]


def test_release_3_npt_runs_one_side_per_process(tmp_path):
    with pytest.raises(SystemExit):
        ab_fastmace.parse_args(_args(tmp_path, "--release", "3", "--expected-commit", "abc", "--ensemble", "npt"))
    for side in ("3", "3b"):
        args = ab_fastmace.parse_args(_args(tmp_path, "--release", "3", "--expected-commit", "abc",
                                            "--ensemble", "npt", "--sides", side))
        assert args.ensemble == "npt" and args.sides == [side]


def test_earlier_releases_are_unchanged(tmp_path):
    args = ab_fastmace.parse_args(_args(tmp_path, "--release", "p3b"))
    assert args.sides == ["p3b", "3"] and args.expected_commit == "156ab76"


# --- tools/trajectory_versions.py --against 3 --ensemble npt ------------------------------------------------


def _run_args(tmp_path, *extra):
    return ["run", "--output", str(tmp_path / "out"), "--checkpoint", "m", *extra]


def test_trajectory_run_against_3(tmp_path):
    args = trajectory_versions.parse_args(_run_args(tmp_path, "--against", "3", "--side", "3b", "--device", "cpu",
                                                    "--backend", "auto", "--ensemble", "npt"))
    assert args.ensemble == "npt" and args.expected_commit is None
    args = trajectory_versions.parse_args(_run_args(tmp_path, "--against", "3", "--side", "3", "--renamed", "v",
                                                    "--expected-commit", "abc", "--reproducible", "--backend", "cueq"))
    assert args.ensemble == "nvt" and args.expected_commit == "abc"
    with pytest.raises(SystemExit):  # the release side needs the tag's commit
        trajectory_versions.parse_args(_run_args(tmp_path, "--against", "3", "--side", "3", "--renamed", "v"))
    for side in ("p3b", "2"):
        with pytest.raises(SystemExit):
            trajectory_versions.parse_args(_run_args(tmp_path, "--against", "3", "--side", side, "--renamed", "v",
                                                     "--expected-commit", "abc"))


def test_npt_rows_record_the_cell_and_stress():
    values = {"positions": np.zeros((2, 3)), "cell": np.eye(3) * 5.0, "stress": np.arange(6.0)}
    assert trajectory_versions.npt_fields(values) == {"cell": (np.eye(3) * 5.0).tolist(),
                                                      "stress": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}
    assert trajectory_versions.npt_fields({"positions": np.zeros((2, 3))}) == {}


def _rows(cell=5.0, stress=0.0, npt=True):
    rows = []
    for s in range(10, 101, 10):
        row = {"step": s, "target": 300.0, "energy": -1.0, "temperature": 300.0, "positions": [[0.0, 0.0, s * 1e-3]]}
        if npt:
            row.update(cell=[[cell, 0.0, 0.0], [0.0, cell, 0.0], [0.0, 0.0, cell]], stress=[stress] * 6)
        rows.append(row)
    return rows


def _write(directory, side, rows, ensemble="npt"):
    directory.mkdir()
    meta = dict(status="complete", side=side, fastmace_commit=side, backend="auto", steps=100, every=10, seeds=[1],
                targets=[300.0, 600.0], structure="NaCl", model="mh1-r2scan",
                runs={"1": dict(plan=dict(options={"a": True}), rows=rows)})
    if ensemble is not None:
        meta["ensemble"] = ensemble
    (directory / "trajectory.json").write_text(json.dumps(meta))
    return directory


def test_compare_bitwise_includes_cell_and_stress(tmp_path):
    release = _write(tmp_path / "a", "3", _rows())
    same = _write(tmp_path / "b", "3b", _rows())
    cell = _write(tmp_path / "c", "3b", _rows(cell=5.0 + 1e-15))
    stress = _write(tmp_path / "d", "3b", _rows(stress=1e-18))
    assert trajectory_versions.compare(release, same, exact=True)["passed"]
    for other in (cell, stress):
        assert not trajectory_versions.compare(release, other, exact=True)["passed"]


def test_compare_refuses_mixed_ensembles(tmp_path):
    npt = _write(tmp_path / "a", "3", _rows())
    nvt = _write(tmp_path / "b", "3b", _rows(npt=False), ensemble="nvt")
    with pytest.raises(ValueError, match="ensemble"):
        trajectory_versions.compare(npt, nvt, exact=True)
    old = _write(tmp_path / "c", "3", _rows(npt=False), ensemble=None)  # a run from before 3b-T10: NVT
    assert trajectory_versions.compare(old, nvt, exact=True)["passed"]


# --- tools/acceptance/resultlog.py and stoppable_pytest.sh ----------------------------------------------------

DUMMY_SUITE = '''
import time

def test_fast():
    pass

def test_fails():
    assert 1 == 2

def test_slow():
    time.sleep(900)  # far beyond every timeout below: only the signal can end it

def test_never_reached():
    pass
'''


def _suite(tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "pytest.ini").write_text("[pytest]\naddopts =\n")
    (suite / "test_dummy.py").write_text(DUMMY_SUITE)
    return suite


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _wait_for(path, nodeid_part, timeout_s=300):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if any(nodeid_part in r.get("nodeid", "") for r in _records(path)):
            return
        time.sleep(0.2)
    raise AssertionError(f"{nodeid_part} never reached {path}")


def _junit(path):
    root = ElementTree.parse(path).getroot()
    # pytest writes the interrupted test as an unnamed <testcase/>; the result log names it.
    cases = {case.get("name"): case for case in root.iter("testcase") if case.get("name") is not None}
    return {name: ("failed" if case.find("failure") is not None else "passed") for name, case in cases.items()}


def _check_stopped(out):
    records = _records(out / "results.jsonl")
    tests = {r["nodeid"].split("::")[-1]: r["outcome"] for r in records if r["event"] == "test"}
    assert tests == {"test_fast": "passed", "test_fails": "failed"}
    interrupted = [r for r in records if r["event"] == "interrupted"]
    assert len(interrupted) == 1 and interrupted[0]["nodeid"].endswith("::test_slow")
    assert records[-1] == {**records[-1], "event": "session_finish", "exitstatus": 2}
    assert _junit(out / "junit.xml") == {"test_fast": "passed", "test_fails": "failed"}


# 6-T13: 60 s timed out under the full -n 8 suite (plans/phase-6.md, Known flakes). A slow start or a slow
# interrupt under load is not what these tests check; returncode 2 and the interrupted record still prove the
# signal, not the dummy test, ended the run (test_slow sleeps 900 s).
STOP_TIMEOUT_S = 300


def _environment():
    env = dict(os.environ)
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_XDIST_WORKER", None)
    return env


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals and bash")
@pytest.mark.parametrize("name", ["SIGINT", "SIGTERM", "SIGHUP"])
def test_a_stopped_job_keeps_its_results_and_junit(tmp_path, name):
    suite, out = _suite(tmp_path), tmp_path / "out"
    job = subprocess.Popen(["bash", str(ACCEPTANCE / "stoppable_pytest.sh"), str(out), sys.executable, "-q",
                            "-p", "no:cacheprovider", "-p", "no:xdist", "test_dummy.py"],
                           cwd=suite, env=_environment(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        _wait_for(out / "results.jsonl", "test_fails")
        time.sleep(0.5)  # test_slow is running
        job.send_signal(getattr(signal, name))
        output, _ = job.communicate(timeout=STOP_TIMEOUT_S)
    finally:
        if job.poll() is None:
            job.kill()
    assert job.returncode == 2, output.decode()  # pytest's "interrupted"
    assert int((out / "pytest.pid").read_text()) > 0
    _check_stopped(out)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_the_plugin_rearms_an_ignored_sigint(tmp_path):
    """A non-interactive bash starts background jobs with SIGINT ignored; the plugin makes it stop pytest again."""
    suite, out = _suite(tmp_path), tmp_path / "out"
    env = _environment()
    env["PYTHONPATH"] = str(ACCEPTANCE) + os.pathsep + env.get("PYTHONPATH", "")
    job = subprocess.Popen([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:xdist",
                            "-p", "resultlog", "--result-log", str(out / "results.jsonl"),
                            "--junitxml", str(out / "junit.xml"), "test_dummy.py"],
                           cwd=suite, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
    try:
        _wait_for(out / "results.jsonl", "test_fails")
        time.sleep(0.5)
        job.send_signal(signal.SIGINT)
        output, _ = job.communicate(timeout=STOP_TIMEOUT_S)
    finally:
        if job.poll() is None:
            job.kill()
    assert job.returncode == 2, output.decode()
    _check_stopped(out)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_job_script_parses():
    for script in ("stoppable_pytest.sh", "phase3b_local.sh"):
        subprocess.run(["bash", "-n", str(ACCEPTANCE / script)], check=True)
