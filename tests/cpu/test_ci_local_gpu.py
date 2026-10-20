"""tools/ci/local_gpu.py (plan task 2-T18): the local GPU reservation, with /proc mocked."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("_tool_local_gpu", REPOSITORY_ROOT / "tools" / "ci" / "local_gpu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lg = _load()


class FakeProc:
    """A /proc with processes (pid -> argv, parent pid, start ticks) and /proc/meminfo."""

    def __init__(self, root: Path, available_gib=14.0):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.meminfo(available_gib)

    def meminfo(self, available_gib):
        kib = int(available_gib * 1024**2)
        (self.root / "meminfo").write_text(f"MemTotal:       16000000 kB\nMemAvailable:   {kib} kB\n")

    def add(self, pid, argv, ppid=1, start=1000):
        directory = self.root / str(pid)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in argv) + b"\0")
        fields = ["S", str(ppid)] + ["0"] * 17 + [str(start)] + ["0"] * 10
        (directory / "stat").write_text(f"{pid} ({Path(argv[0]).name}) " + " ".join(fields) + "\n")
        return pid

    def remove(self, pid):
        for child in (self.root / str(pid)).iterdir():
            child.unlink()
        (self.root / str(pid)).rmdir()


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def proc(tmp_path):
    return FakeProc(tmp_path / "proc")


def test_process_table(proc):
    proc.add(10, ["/usr/bin/python3", "-m", "pytest", "-n", "2", "tests/cpu"], ppid=1, start=555)
    assert lg.start_ticks(10, proc.root) == 555
    assert lg.parent_pid(10, proc.root) == 1
    assert lg.start_ticks(99, proc.root) is None
    assert lg.available_gib(proc.root) == pytest.approx(14.0)


@pytest.mark.parametrize("argv, is_pytest", [
    (["/home/u/env/bin/python", "-m", "pytest", "tests/cpu"], True),
    (["/home/u/env/bin/pytest", "tests/gpu"], True),
    (["pytest"], True),
    (["bash", "-lc", "cd x && python -m pytest tests/cpu"], False),  # a shell that mentions pytest is not one
    (["python3", "tools/test_budget.py", "junit.xml"], False),
])
def test_pytest_processes(proc, argv, is_pytest):
    proc.add(20, argv)
    assert (lg.pytest_processes(proc.root, own_pid=1) == [20]) is is_pytest


def test_own_ancestors_are_not_counted(proc):
    proc.add(30, ["python3", "-m", "pytest", "tests/gpu"], ppid=1)
    proc.add(31, ["python3", "tools/ci/local_gpu.py", "run", "--", "x"], ppid=30)
    assert lg.pytest_processes(proc.root, own_pid=31) == []


def test_reserve_creates_the_marker_with_the_holder(proc, tmp_path):
    marker = tmp_path / "gpu"
    proc.add(40, ["bash"], start=777)
    ok, holder = lg.reserve(marker, "2-T05 OOM cases", pid=40, proc_root=proc.root, user="fzwin", now="T")
    assert ok
    assert holder == dict(owner="fzwin", purpose="2-T05 OOM cases", pid=40, start_ticks=777, started="T")
    assert json.loads((marker / "holder.json").read_text()) == holder


def test_a_live_marker_blocks_a_second_reservation(proc, tmp_path):
    marker = tmp_path / "gpu"
    proc.add(40, ["bash"])
    proc.add(41, ["bash"])
    assert lg.reserve(marker, "first", pid=40, proc_root=proc.root)[0]
    ok, holder = lg.reserve(marker, "second", pid=41, proc_root=proc.root)
    assert not ok and holder["purpose"] == "first"


def test_a_stale_marker_is_reported_and_removed(proc, tmp_path, capsys):
    marker = tmp_path / "gpu"
    proc.add(40, ["bash"])
    proc.add(41, ["bash"])
    assert lg.reserve(marker, "crashed run", pid=40, proc_root=proc.root)[0]
    proc.remove(40)
    ok, holder = lg.reserve(marker, "next", pid=41, proc_root=proc.root)
    assert ok and holder["purpose"] == "next"
    assert "stale" in capsys.readouterr().err


def test_a_reused_pid_is_stale(proc, tmp_path):
    marker = tmp_path / "gpu"
    proc.add(40, ["bash"], start=100)
    assert lg.reserve(marker, "old", pid=40, proc_root=proc.root)[0]
    proc.add(40, ["other"], start=999)  # the kernel gave pid 40 to another process
    assert lg.live_holder(marker, proc.root) is None


def test_a_marker_without_holder_file_is_live_for_a_minute(proc, tmp_path):
    marker = tmp_path / "gpu"
    marker.mkdir()  # a reserver between mkdir and writing holder.json
    assert lg.live_holder(marker, proc.root, now=marker.stat().st_mtime + 10)["purpose"] == "being reserved"
    assert lg.live_holder(marker, proc.root, now=marker.stat().st_mtime + 120) is None


def test_release_only_removes_its_own_marker(proc, tmp_path):
    marker = tmp_path / "gpu"
    proc.add(40, ["bash"])
    assert lg.reserve(marker, "mine", pid=40, proc_root=proc.root)[0]
    assert not lg.release(marker, pid=41)
    assert marker.exists()
    assert lg.release(marker, pid=40)
    assert not marker.exists()
    assert not lg.release(marker, pid=40)  # nothing left: fine


def test_check(proc, tmp_path):
    marker = tmp_path / "gpu"
    proc.add(40, ["bash"])
    assert lg.check(marker, proc.root) == (True, [])
    lg.reserve(marker, "2-T05 OOM cases", pid=40, proc_root=proc.root, user="fzwin")
    free, reasons = lg.check(marker, proc.root)
    assert not free and "2-T05 OOM cases" in reasons[0] and "fzwin" in reasons[0]
    # The other rule workers check before a local CPU suite: 6 GB available.
    proc.remove(40)
    proc.meminfo(5.0)
    free, reasons = lg.check(marker, proc.root, min_available_gib=6)
    assert not free and "available" in reasons[0]
    assert not marker.exists()  # the stale marker went on the way


def test_wait_for_the_machine(proc):
    clock = Clock()
    suite = proc.add(50, ["python", "-m", "pytest", "tests/cpu"])
    calls = []

    def sleep(seconds):
        calls.append(seconds)
        clock.sleep(seconds)
        if len(calls) == 2:
            proc.remove(suite)  # the sibling CPU suite ends

    ok, reasons = lg.wait_for_machine(proc.root, own_pid=1, min_available_gib=10, wait_s=600, interval_s=30,
                                      clock=clock, sleep=sleep)
    assert ok and reasons == [] and clock.now == 60


def test_wait_for_the_machine_needs_memory_too(proc):
    clock = Clock()
    proc.meminfo(8.0)
    ok, reasons = lg.wait_for_machine(proc.root, own_pid=1, min_available_gib=10, wait_s=90, interval_s=30,
                                      clock=clock, sleep=clock.sleep)
    assert not ok and "available" in reasons[0] and clock.now == 90


def test_acquire_waits_for_the_holder_then_for_the_machine(proc, tmp_path):
    marker = tmp_path / "gpu"
    clock = Clock()
    proc.add(40, ["bash"])
    proc.add(41, ["bash"])
    lg.reserve(marker, "first", pid=40, proc_root=proc.root)
    calls = []

    def sleep(seconds):
        calls.append(seconds)
        clock.sleep(seconds)
        if len(calls) == 1:
            lg.release(marker, pid=40)

    ok, reasons = lg.acquire(marker, "second", pid=41, proc_root=proc.root, wait_s=600, interval_s=30,
                             min_available_gib=10, clock=clock, sleep=sleep)
    assert ok and lg.live_holder(marker, proc.root)["purpose"] == "second"


def test_acquire_gives_up_and_releases_when_the_machine_stays_busy(proc, tmp_path):
    marker = tmp_path / "gpu"
    clock = Clock()
    proc.add(41, ["bash"])
    proc.add(50, ["python", "-m", "pytest", "tests/cpu"])
    ok, reasons = lg.acquire(marker, "ci", pid=41, proc_root=proc.root, wait_s=60, interval_s=30,
                             min_available_gib=10, clock=clock, sleep=clock.sleep)
    assert not ok and "pytest" in reasons[0]
    assert not marker.exists()  # no reservation is left behind


def test_run_reserves_runs_and_releases(tmp_path, monkeypatch):
    marker = tmp_path / "gpu"
    seen = tmp_path / "seen.json"
    monkeypatch.setattr(lg, "wait_for_machine", lambda *args, **kwargs: (True, []))
    script = ("import json, pathlib, sys; m = pathlib.Path(sys.argv[1]) / 'holder.json'; "
              "pathlib.Path(sys.argv[2]).write_text(m.read_text()); sys.exit(3)")
    status = lg.main(["run", "--marker", str(marker), "--purpose", "unit test", "--wait-minutes", "0", "--",
                      sys.executable, "-c", script, str(marker), str(seen)])
    assert status == 3  # the command's own status
    assert json.loads(seen.read_text())["purpose"] == "unit test"
    assert not marker.exists()


def test_run_reports_busy_with_status_75(tmp_path, monkeypatch, capsys):
    marker = tmp_path / "gpu"
    monkeypatch.setattr(lg, "wait_for_machine", lambda *args, **kwargs: (False, ["2 pytest processes run"]))
    status = lg.main(["run", "--marker", str(marker), "--purpose", "ci", "--wait-minutes", "0", "--", "true"])
    assert status == lg.BUSY == 75
    assert "pytest" in capsys.readouterr().err
    assert not marker.exists()


def test_check_command_status(tmp_path):
    assert lg.main(["check", "--marker", str(tmp_path / "gpu")]) == 0
