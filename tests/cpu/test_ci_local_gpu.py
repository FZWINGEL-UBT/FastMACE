"""tools/ci/local_gpu.py (plan task 2-T18): the local GPU reservation, with /proc mocked."""

import importlib.util
import json
import sys
import time
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

    def add(self, pid, argv, ppid=1, start=1000, env=None):
        directory = self.root / str(pid)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in argv) + b"\0")
        if env is not None:
            (directory / "environ").write_bytes(b"".join(f"{k}={v}".encode() + b"\0" for k, v in env.items()))
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
    # a queued reservation wrapper whose command is pytest is not a running pytest (round-2 deadlock)
    (["python3", "/x/tools/ci/local_gpu.py", "run", "--purpose", "p", "--", "python", "-m", "pytest", "t"], False),
    # ... also behind a prefix such as timeout or env (phase 3 deadlock)
    (["timeout", "900", "python3", "tools/ci/local_gpu.py", "run", "--", "env", "A=1", "python", "-m", "pytest"],
     False),
    (["/home/u/env/bin/python", "-m", "pytest", "tests/cpu/test_ci_local_gpu.py"], True),
])
def test_pytest_processes(proc, argv, is_pytest):
    proc.add(20, argv)
    assert (lg.pytest_processes(proc.root, own_pid=1) == [20]) is is_pytest


def test_cpu_only_pytest_processes_count_only_for_quiet_runs(proc):
    proc.add(60, ["python", "-m", "pytest", "tests/cpu"], env={"HOME": "/h", "CUDA_VISIBLE_DEVICES": ""})
    proc.add(61, ["python", "-m", "pytest", "tests/cpu"], env={"CUDA_VISIBLE_DEVICES": "-1"})
    proc.add(62, ["python", "-m", "pytest", "tests/gpu"], env={"CUDA_VISIBLE_DEVICES": "0"})
    proc.add(63, ["python", "-m", "pytest", "tests/gpu"])  # no readable environment: it counts
    assert lg.pytest_processes(proc.root, own_pid=1) == [60, 61, 62, 63]
    assert lg.pytest_processes(proc.root, own_pid=1, gpu_only=True) == [62, 63]
    reasons = lg.machine_reasons(proc.root, own_pid=1)
    assert "can use the GPU" in reasons[0] and "[62, 63]" in reasons[0]
    proc.remove(62)
    proc.remove(63)
    assert lg.machine_reasons(proc.root, own_pid=1) == []  # CPU-only suites do not block a GPU run
    assert "[60, 61]" in lg.machine_reasons(proc.root, own_pid=1, quiet=True)[0]  # but a timing run waits


def test_a_cpu_only_check_is_blocked_only_by_a_quiet_reservation(proc, tmp_path):
    marker = tmp_path / "gpu"
    proc.add(40, ["bash"])
    lg.reserve(marker, "3-Tnn fast tier", pid=40, proc_root=proc.root)
    assert "quiet" not in json.loads((marker / "holder.json").read_text())
    assert lg.check(marker, proc.root, cpu_only=True) == (True, [])
    assert not lg.check(marker, proc.root)[0]
    assert lg.release(marker, pid=40)
    ok, holder = lg.reserve(marker, "3-Tnn timing", pid=40, proc_root=proc.root, quiet=True)
    assert ok and holder["quiet"] is True and json.loads((marker / "holder.json").read_text())["quiet"] is True
    free, reasons = lg.check(marker, proc.root, cpu_only=True)
    assert not free and "3-Tnn timing" in reasons[0] and "(quiet)" in reasons[0]


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


def test_the_quiet_and_cpu_only_flags(tmp_path, monkeypatch):
    marker = tmp_path / "gpu"
    seen = {}

    def wait(*args, **kwargs):
        seen.update(kwargs)
        return True, []

    monkeypatch.setattr(lg, "wait_for_machine", wait)
    status = lg.main(["run", "--marker", str(marker), "--purpose", "timing", "--quiet", "--wait-minutes", "0",
                      "--", sys.executable, "-c", "pass"])
    assert status == 0 and seen["quiet"] is True and not marker.exists()
    assert lg.main(["check", "--cpu-only", "--marker", str(marker)]) == 0


def test_check_command_status(tmp_path):
    assert lg.main(["check", "--marker", str(tmp_path / "gpu")]) == 0


# The queue (task OPS-T01): waiting reservations take the GPU in arrival order. On 22 October 2026 waiters
# raced for the marker at each release; one worker's back-to-back jobs kept winning for 37 minutes.

def _queued(marker, proc, pid, purpose, arrival_ns):
    proc.add(pid, ["python3", "tools/ci/local_gpu.py", "run"])
    return lg.enqueue(marker, purpose, pid, pid, proc.root, arrival_ns=arrival_ns)


def test_the_queue_is_next_to_the_marker(tmp_path):
    assert lg.queue_dir(tmp_path / ".fastmace-local-gpu") == tmp_path / ".fastmace-local-gpu-queue"


def test_waiters_take_the_gpu_in_arrival_order(proc, tmp_path):
    marker = tmp_path / "gpu"
    proc.add(40, ["bash"])
    assert lg.reserve(marker, "holder", pid=40, proc_root=proc.root)[0]
    a = _queued(marker, proc, 41, "a", 1)
    b = _queued(marker, proc, 42, "b", 2)
    c = _queued(marker, proc, 43, "c", 3)
    assert [t["purpose"] for _, t in lg.live_tickets(marker, proc.root)] == ["a", "b", "c"]
    assert not lg.take_turn(marker, *a, proc_root=proc.root)[0]  # the holder still has it
    lg.release(marker, pid=40)
    ok, reason = lg.take_turn(marker, *c, proc_root=proc.root)
    assert not ok and "'a'" in reason  # c polls first but is behind a and b
    assert not lg.take_turn(marker, *b, proc_root=proc.root)[0]
    ok, holder = lg.take_turn(marker, *a, proc_root=proc.root)
    assert ok and holder["purpose"] == "a" and not a[0].exists()
    lg.release(marker, pid=41)
    assert not lg.take_turn(marker, *c, proc_root=proc.root)[0]
    assert lg.take_turn(marker, *b, proc_root=proc.root)[0]
    lg.release(marker, pid=42)
    assert lg.take_turn(marker, *c, proc_root=proc.root)[0]
    assert lg.live_tickets(marker, proc.root) == []


def test_a_stale_ticket_is_skipped(proc, tmp_path, capsys):
    marker = tmp_path / "gpu"
    _queued(marker, proc, 41, "crashed waiter", 1)
    b = _queued(marker, proc, 42, "b", 2)
    proc.remove(41)
    ok, holder = lg.take_turn(marker, *b, proc_root=proc.root)
    assert ok and holder["purpose"] == "b"
    assert "stale" in capsys.readouterr().err
    assert list(lg.queue_dir(marker).iterdir()) == []


def test_a_ticket_whose_pid_was_reused_is_stale(proc, tmp_path):
    marker = tmp_path / "gpu"
    _queued(marker, proc, 41, "old", 1)
    proc.add(41, ["other"], start=999)
    assert lg.live_tickets(marker, proc.root) == []


def test_a_lost_ticket_is_rewritten_at_its_place(proc, tmp_path):
    marker = tmp_path / "gpu"
    proc.add(40, ["bash"])
    lg.reserve(marker, "holder", pid=40, proc_root=proc.root)
    a = _queued(marker, proc, 41, "a", 1)
    _queued(marker, proc, 42, "b", 2)
    a[0].unlink()
    assert not lg.take_turn(marker, *a, proc_root=proc.root)[0]
    assert [t["purpose"] for _, t in lg.live_tickets(marker, proc.root)] == ["a", "b"]


def test_a_waiter_that_times_out_leaves_the_queue(proc, tmp_path):
    marker = tmp_path / "gpu"
    clock = Clock()
    proc.add(40, ["bash"])
    proc.add(41, ["bash"])
    lg.reserve(marker, "holder", pid=40, proc_root=proc.root)
    ok, reasons = lg.acquire(marker, "patient", pid=41, proc_root=proc.root, wait_s=60, interval_s=30,
                             min_available_gib=10, clock=clock, sleep=clock.sleep)
    assert not ok and "holder" in reasons[0] and clock.now == 60
    assert lg.live_tickets(marker, proc.root) == []


def test_acquire_waits_behind_an_earlier_ticket(proc, tmp_path):
    marker = tmp_path / "gpu"
    clock = Clock()
    proc.add(41, ["bash"])
    first = _queued(marker, proc, 42, "first", 1)
    calls = []

    def sleep(seconds):
        calls.append(seconds)
        clock.sleep(seconds)
        if len(calls) == 2:  # the earlier waiter takes its turn and finishes
            assert lg.take_turn(marker, *first, proc_root=proc.root)[0]
            lg.release(marker, pid=42)

    ok, _ = lg.acquire(marker, "second", pid=41, proc_root=proc.root, wait_s=600, interval_s=30,
                       min_available_gib=10, clock=clock, sleep=sleep)
    assert ok and len(calls) == 2  # the GPU was free all along, but "first" came first


def test_concurrent_waiters_run_in_arrival_order(proc, tmp_path):
    import threading

    marker = tmp_path / "gpu"
    proc.add(40, ["bash"])
    lg.reserve(marker, "holder", pid=40, proc_root=proc.root)
    order, threads = [], []

    def waiter(pid):
        ok, _ = lg.acquire(marker, f"w{pid}", pid=pid, proc_root=proc.root, wait_s=60, interval_s=0.01,
                           min_available_gib=None)
        order.append(pid)
        lg.release(marker, pid)

    for pid in (43, 41, 44, 42):  # arrival order, not pid order
        proc.add(pid, ["bash"])
        threads.append(threading.Thread(target=waiter, args=(pid,)))
        threads[-1].start()
        while len(lg.live_tickets(marker, proc.root)) < len(threads):
            time.sleep(0.005)
    lg.release(marker, pid=40)
    for thread in threads:
        thread.join(30)
    assert order == [43, 41, 44, 42]


def test_a_release_frees_the_marker_even_when_its_directory_cannot_be_removed(proc, tmp_path, monkeypatch):
    """On NFS a holder.json that a waiter has open becomes ``.nfsXXXX`` and the directory survives an rmtree;
    an in-place removal left an empty marker that read as "being reserved" for 60 s (suite-speed)."""
    marker = tmp_path / "gpu"
    proc.add(40, ["bash"])
    proc.add(41, ["bash"])
    assert lg.reserve(marker, "holder", pid=40, proc_root=proc.root)[0]
    rmtree = lg.shutil.rmtree

    def nfs_rmtree(path, ignore_errors=False):
        path = Path(path)
        for child in path.iterdir():
            child.unlink()
        (path / ".nfs0000000000abcd0000000001").write_text("")  # silly-renamed, still open elsewhere

    monkeypatch.setattr(lg.shutil, "rmtree", nfs_rmtree)
    assert lg.release(marker, pid=40)
    assert not marker.exists()
    assert lg.live_holder(marker, proc.root) is None
    ok, holder = lg.reserve(marker, "next", pid=41, proc_root=proc.root)
    assert ok and holder["pid"] == 41
    leftovers = list(tmp_path.glob(".gpu.removed-*"))
    assert len(leftovers) == 1  # the directory renamed aside, waiting for its deletion
    monkeypatch.setattr(lg.shutil, "rmtree", rmtree)
    assert lg.release(marker, pid=41)
    assert not marker.exists() and not list(tmp_path.glob(".gpu.removed-*"))  # later removals clear leftovers


def test_status_lists_the_queue(proc, tmp_path, monkeypatch, capsys):
    marker = tmp_path / "gpu"
    monkeypatch.setattr(lg.time, "time", lambda: 1000.0)
    proc.add(41, ["bash"])
    proc.add(42, ["bash"])
    lg.enqueue(marker, "3-T01 fast tier", 41, 41, proc.root, arrival_ns=int(900e9))
    lg.enqueue(marker, "3-T02 timing", 42, 42, proc.root, arrival_ns=int(990e9))
    lines = lg.queue_lines(marker, proc.root)
    assert "1." in lines[1] and "3-T01 fast tier" in lines[1] and "1m40s" in lines[1]
    assert "2." in lines[2] and "3-T02 timing" in lines[2] and "0m10s" in lines[2]


def test_run_with_a_temporary_home_leaves_no_ticket(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("FASTMACE_LOCAL_GPU_MARKER", raising=False)
    monkeypatch.setattr(lg, "wait_for_machine", lambda *args, **kwargs: (True, []))
    assert lg.main(["run", "--purpose", "unit test", "--wait-minutes", "0", "--", sys.executable, "-c", "pass"]) == 0
    assert not (tmp_path / ".fastmace-local-gpu").exists()
    assert list((tmp_path / ".fastmace-local-gpu-queue").iterdir()) == []
    assert lg.main(["status"]) == 0
    assert "local GPU free" in capsys.readouterr().out
