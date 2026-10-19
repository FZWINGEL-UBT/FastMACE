"""2-T11: stale torch-extension build locks (1b-T27's finding).

A lock left by a killed OEq JIT build made every process that imports MACE
before FastMACE wait 300 s silently in OEq 0.7's ``_wait_for_torch_build_lock``.
Fake lock files, fake ages and fake holder scans; no real waits.
"""

import inspect
import os
import sys
import warnings
from pathlib import Path

import pytest
from test_oeq_runtime import fake_build, fake_runtime

from fastmace.kernels import _oeq_runtime as runtime_module

HOUR = 3600.0
NOW = 2_000_000_000.0


def make_lock(directory, age, now=NOW):
    """A torch ``FileBaton`` lock file written ``age`` seconds before ``now``."""
    lock = Path(directory) / runtime_module.BUILD_LOCK
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_bytes(b"")
    os.utime(lock, (now - age, now - age))
    return lock


def no_holders(path):
    return []


# ------------------------------------------------------------------ finding locks and their holders

def test_finds_locks_in_both_torch_layouts(tmp_path):
    """``<root>/<name>/lock`` (TORCH_EXTENSIONS_DIR set) and ``<root>/<py>_<cu>/<name>/lock`` (default root)."""
    flat = make_lock(tmp_path / "flat" / "libtorch_tp_jit", 2 * HOUR)
    nested = make_lock(tmp_path / "nested" / "py310_cu130" / "libtorch_tp_jit", 2 * HOUR)
    (tmp_path / "nested" / "py310_cu130" / "other").mkdir()
    locks = runtime_module.build_locks([tmp_path / "flat", tmp_path / "nested", tmp_path / "missing"],
                                       now=NOW, holders=no_holders)
    assert sorted(lock.path for lock in locks) == sorted([flat, nested])
    assert all(lock.age == pytest.approx(2 * HOUR) for lock in locks)


def test_an_old_lock_without_a_holder_is_stale(tmp_path):
    lock = make_lock(tmp_path / "py310_cu130" / "libtorch_tp_jit", 8 * 24 * HOUR)
    (stale,) = runtime_module.stale_build_locks([tmp_path], now=NOW, holders=no_holders)
    assert stale.path == lock and stale.holders == () and stale.stale


def test_a_young_lock_is_a_running_build(tmp_path):
    make_lock(tmp_path / "libtorch_tp_jit", 10 * 60)
    assert runtime_module.stale_build_locks([tmp_path], now=NOW, holders=no_holders) == []


def test_the_threshold_is_an_hour_by_default(tmp_path):
    assert runtime_module.STALE_LOCK_AGE == HOUR
    make_lock(tmp_path / "a" / "libtorch_tp_jit", HOUR - 60)
    old = make_lock(tmp_path / "b" / "libtorch_tp_jit", HOUR + 60)
    stale = runtime_module.stale_build_locks([tmp_path / "a", tmp_path / "b"], now=NOW, holders=no_holders)
    assert [lock.path for lock in stale] == [old]
    assert runtime_module.stale_build_locks([tmp_path / "b"], threshold=2 * HOUR, now=NOW, holders=no_holders) == []


def test_an_old_lock_with_a_holder_is_not_stale(tmp_path):
    make_lock(tmp_path / "libtorch_tp_jit", 3 * HOUR)
    assert runtime_module.stale_build_locks([tmp_path], now=NOW, holders=lambda path: [4242]) == []


def test_an_old_lock_with_an_unknown_holder_is_not_called_stale(tmp_path):
    """Without /proc nobody can tell whether a build holds it."""
    make_lock(tmp_path / "libtorch_tp_jit", 3 * HOUR)
    (lock,) = runtime_module.build_locks([tmp_path], now=NOW, holders=lambda path: None)
    assert lock.holders is None and not lock.stale
    assert runtime_module.stale_build_locks([tmp_path], now=NOW, holders=lambda path: None) == []


def fake_proc(tmp_path, links):
    """A /proc with ``{pid: [targets of its fds]}``, plus entries that are not processes."""
    proc = tmp_path / "proc"
    for pid, targets in links.items():
        fd = proc / str(pid) / "fd"
        fd.mkdir(parents=True)
        for number, target in enumerate(targets):
            os.symlink(target, fd / str(number))
    (proc / "self").mkdir(parents=True, exist_ok=True)
    (proc / "meminfo").write_text("")
    return proc


def test_holders_are_the_processes_with_the_lock_open(tmp_path):
    lock = make_lock(tmp_path / "ext" / "libtorch_tp_jit", 0)
    proc = fake_proc(tmp_path, {101: [str(lock), "/dev/null"], 202: ["/dev/null"], 303: [str(lock)]})
    assert runtime_module.lock_holders(lock, proc=proc) == [101, 303]
    assert runtime_module.lock_holders(tmp_path / "other", proc=proc) == []


def test_holders_unknown_without_proc(tmp_path):
    lock = make_lock(tmp_path / "libtorch_tp_jit", 0)
    assert runtime_module.lock_holders(lock, proc=tmp_path / "no-proc") is None


def test_finding_locks_writes_nothing(tmp_path):
    make_lock(tmp_path / "py310_cu130" / "libtorch_tp_jit", 3 * HOUR)
    before = sorted((str(p), p.stat().st_mtime) for p in tmp_path.rglob("*"))
    runtime_module.stale_build_locks([tmp_path], now=NOW, holders=no_holders)
    assert sorted((str(p), p.stat().st_mtime) for p in tmp_path.rglob("*")) == before


def test_torch_default_root(tmp_path):
    root = runtime_module.torch_extensions_root
    assert root({"TORCH_EXTENSIONS_DIR": str(tmp_path / "x")}) == tmp_path / "x"
    assert root({"XDG_CACHE_HOME": str(tmp_path / "c"), "HOME": "/nowhere"}) == tmp_path / "c" / "torch_extensions"
    assert root({"HOME": str(tmp_path)}) == tmp_path / ".cache" / "torch_extensions"


# ------------------------------------------------------------------ the short wait of the OEq preparation

class FakeClock:
    def __init__(self, on_sleep=None):
        self.now, self.slept, self.on_sleep = 0.0, [], on_sleep

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds
        if self.on_sleep:
            self.on_sleep(self)


def test_the_wait_gives_up_after_its_limit(tmp_path):
    lock = make_lock(tmp_path / "libtorch_tp_jit", 0)
    clock = FakeClock()
    left = runtime_module.wait_for_build_locks(tmp_path, wait=5.0, poll=1.0, clock=clock, sleep=clock.sleep)
    assert left == [lock]
    assert clock.now == pytest.approx(5.0) and len(clock.slept) == 5


def test_the_wait_ends_when_the_lock_goes(tmp_path):
    lock = make_lock(tmp_path / "libtorch_tp_jit", 0)
    clock = FakeClock(on_sleep=lambda c: c.now >= 2 and lock.exists() and lock.unlink())
    assert runtime_module.wait_for_build_locks(tmp_path, wait=30.0, poll=1.0, clock=clock, sleep=clock.sleep) == []
    assert clock.now == pytest.approx(2.0)


def test_no_lock_no_wait(tmp_path):
    clock = FakeClock()
    assert runtime_module.wait_for_build_locks(tmp_path, wait=30.0, clock=clock, sleep=clock.sleep) == []
    assert clock.slept == []


def test_the_default_wait_is_short_and_a_keyword():
    assert runtime_module.LOCK_WAIT == 30.0
    parameter = inspect.signature(runtime_module.prepare_in_process).parameters["lock_wait"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.default == runtime_module.LOCK_WAIT


@pytest.fixture
def preparable(tmp_path, monkeypatch):
    """``prepare_in_process`` on a fake runtime, with a seeding step that may leave a lock behind."""
    runtime, driver = fake_runtime(tmp_path)
    monkeypatch.setattr(runtime_module, "find_runtime", lambda: runtime)
    monkeypatch.setattr(runtime_module, "find_driver", lambda environ: driver)
    monkeypatch.setattr(runtime_module, "_compiler_version", lambda command, environ: "g++ 13")
    monkeypatch.delitem(sys.modules, "openequivariance", raising=False)
    monkeypatch.delitem(sys.modules, "torch.utils.cpp_extension", raising=False)
    planted = []
    seed = runtime_module.reuse_extension

    def reuse(cache, extensions_dir, identity, build):
        info = seed(cache, extensions_dir, identity, build)
        if planted:
            planted.append(make_lock(extensions_dir / "libtorch_tp_jit", 0))
        return info

    monkeypatch.setattr(runtime_module, "reuse_extension", reuse)

    def prepare(lock=False, **options):
        if lock:
            planted.append("plant")
        return runtime_module.prepare_in_process(tmp_path / "cache", preload=False, environ={"PATH": "/usr/bin"},
                                                 build=fake_build, **options)
    prepare.planted = planted
    return prepare


def test_prepare_without_a_lock_does_not_warn(preparable):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        runtime_plan, _ = preparable()
    assert runtime_plan.extension["source"] == "built"


def test_prepare_gives_up_on_a_lock_and_names_it(preparable):
    with pytest.warns(UserWarning, match="build lock") as caught:
        with pytest.raises(runtime_module.OEqRuntimeError) as raised:
            preparable(lock=True, lock_wait=0)
    lock = str(preparable.planted[-1])
    assert lock in str(caught[0].message) and "delete" in str(caught[0].message)
    error = raised.value
    assert error.what.startswith("OEq unavailable: stale build lock")
    assert lock in error.what and lock in error.next_step


def test_a_stale_lock_falls_to_plain_cueq_with_the_reason(preparable, monkeypatch):
    """``prepare_hybrid_if_possible`` records the reason; the planner then gives plain cuEq (SPEC §9.4)."""
    from fastmace import _patches
    from fastmace.model import loading

    monkeypatch.setattr(_patches, "_RECORDS", {})
    monkeypatch.setattr(loading, "_oeq_installed", lambda: True)
    monkeypatch.setattr(loading, "_cuda_visible", lambda: True)
    monkeypatch.setattr(loading, "hybrid_blockers", lambda modules=None: [])
    monkeypatch.setattr(_patches, "prepare_oeq_runtime", lambda cache: preparable(lock=True, lock_wait=0))
    with pytest.warns(UserWarning, match="build lock"):
        outcome = loading.prepare_hybrid_if_possible(cache=Path("unused"))
    assert outcome.startswith("failed: ") and "OEq unavailable: stale build lock" in outcome
