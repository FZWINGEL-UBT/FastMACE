"""The capture arena's allocator (fastmace/engine/arena_shim.c) and its Python side, without a GPU.

The shim resolves libcuda only when it is asked for device memory, so its plan bookkeeping, mismatch counting and
the control calls run on any Linux machine with a C compiler.
"""

import ctypes
import shutil
import sys

import pytest
import torch

from fastmace.config import EngineConfig
from fastmace.engine import arena_cuda

linux_cc = pytest.mark.skipif(not sys.platform.startswith("linux") or not (shutil.which("gcc") or shutil.which("cc")),
                              reason="needs Linux and a C compiler")


def test_capture_arena_is_off_by_default():
    assert EngineConfig().capture_arena is False


@pytest.fixture
def lib(tmp_path):
    so = arena_cuda.build(tmp_path)
    lib = ctypes.CDLL(str(so))
    for name in ("arena_mismatches", "arena_fallback_bytes", "arena_planned_served", "arena_refusals",
                 "arena_virtual_live", "arena_log_count", "arena_set_headroom"):
        getattr(lib, name).restype = ctypes.c_int64
    lib.arena_set_headroom.argtypes = [ctypes.c_int64]
    lib.arena_set_plan.argtypes = [ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
                                   ctypes.c_int64]
    lib.arena_malloc.restype = ctypes.c_void_p
    lib.arena_malloc.argtypes = [ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    return lib


def _plan(lib, sizes, offsets, total):
    n = len(sizes)
    return lib.arena_set_plan(n, (ctypes.c_int64 * n)(*sizes), (ctypes.c_int64 * n)(*offsets), total)


@linux_cc
def test_build_is_cached_per_source(tmp_path):
    first = arena_cuda.build(tmp_path)
    assert first.exists() and arena_cuda.build(tmp_path) == first


@linux_cc
def test_planned_mode_counts_requests_off_the_plan(lib):
    assert _plan(lib, [4 << 20, 2 << 20], [0, 4 << 20], 6 << 20) == 0
    lib.arena_mode(1)
    lib.arena_malloc(8 << 20, 0, None)  # the plan's first request is 4 MiB: off the plan
    lib.arena_malloc(2 << 20, 0, None)  # size matches, but no device here: the arena cannot be made
    lib.arena_malloc(2 << 20, 0, None)  # past the plan's end
    lib.arena_mode(0)
    assert lib.arena_planned_served() == 3
    assert lib.arena_mismatches() == 3
    assert lib.arena_fallback_bytes() == (8 + 2 + 2) << 20
    assert lib.arena_log_count() == 3


@linux_cc
def test_a_new_plan_resets_the_counters(lib):
    _plan(lib, [2 << 20], [0], 2 << 20)
    lib.arena_mode(1)
    lib.arena_malloc(4 << 20, 0, None)
    lib.arena_mode(0)
    assert lib.arena_mismatches() == 1
    assert _plan(lib, [2 << 20], [0], 2 << 20) == 0
    assert (lib.arena_mismatches(), lib.arena_fallback_bytes(), lib.arena_planned_served()) == (0, 0, 0)


@linux_cc
def test_virtual_mode_without_a_driver_returns_null_and_holds_nothing(lib):
    lib.arena_mode(2)
    assert not lib.arena_malloc(2 << 20, 0, None)
    lib.arena_mode(0)
    assert lib.arena_virtual_live() == 0


@linux_cc
def test_headroom_is_settable(lib):
    assert lib.arena_set_headroom(3 << 30) == 3 << 30
    assert lib.arena_set_headroom(arena_cuda.HEADROOM_BYTES) == 1 << 30


def test_max_difference_is_zero_only_for_equal_bits():
    a = {"forces": torch.tensor([1.0, 2.0]), "energy": torch.tensor(3.0)}
    assert arena_cuda.max_difference(a, {k: v.clone() for k, v in a.items()}) == {"energy": 0.0, "forces": 0.0}
    b = {"forces": torch.tensor([1.0, 2.5]), "energy": torch.tensor(3.0)}
    assert arena_cuda.max_difference(b, a) == {"energy": 0.0, "forces": 0.5}


def test_max_difference_treats_unequal_nan_patterns_as_infinite():
    a = {"stress": torch.tensor([float("nan"), 1.0])}
    b = {"stress": torch.tensor([0.0, 1.0])}
    assert arena_cuda.max_difference(a, b) == {"stress": float("inf")}
    assert arena_cuda.max_difference(a, {"stress": a["stress"].clone()}) == {"stress": 0.0}


def test_the_arena_is_refused_off_linux_and_on_wsl(monkeypatch):
    arena_cuda._PROCESS.pop("shim", None)
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(arena_cuda.ArenaUnavailable, match="Linux"):
        arena_cuda.shim()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(arena_cuda, "_on_wsl", lambda: True)
    with pytest.raises(arena_cuda.ArenaUnavailable, match="WSL"):
        arena_cuda.shim()


def test_one_engine_holds_the_arena():
    arena_cuda._PROCESS.pop("owner", None)
    arena_cuda.claim(1)
    arena_cuda.claim(1)
    with pytest.raises(arena_cuda.ArenaUnavailable, match="another engine"):
        arena_cuda.claim(2)
    arena_cuda.unclaim(2)  # not the holder: no effect
    with pytest.raises(arena_cuda.ArenaUnavailable):
        arena_cuda.claim(2)
    arena_cuda.unclaim(1)
    arena_cuda.claim(2)
    arena_cuda.unclaim(2)
