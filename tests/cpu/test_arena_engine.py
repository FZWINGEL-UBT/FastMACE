"""The capture-arena build's pool handling in ``GraphEngine`` (fastmace/engine/graph.py), without a GPU.

A pool that keeps segments after its graphs are dropped must end the build in the fallback, never let it go on: on
A100 stage 1 (Alex 4981434/4981437) the recording pool kept every segment because the caller still named its
MemPool. The fakes here stand in for the CUDA calls: a pool "keeps segments" while any reference to it is alive.
"""

import gc
import types
import weakref

import pytest
import torch

from fastmace.engine import arena_cuda
from fastmace.engine.graph import GraphEngine, _ArenaFallback


class FakePool:
    def __init__(self, number):
        self.id = (0, number)


class FakeShim:
    def __init__(self):
        self.modes = []

    def clear_log(self):
        pass

    def alias_fill(self, device):
        pass

    def mode(self, mode):
        self.modes.append(mode)

    def counters(self):
        return {"uncaptured": 0, "virtual_live": 0, "mismatches": 0, "refusals": 0}

    def alias_intact(self):
        return True


@pytest.fixture
def engine(monkeypatch):
    """A stand-in with the engine's arena methods; ``pools`` tracks every pool made, weakly."""
    alive: dict[tuple, weakref.ref] = {}

    def segment_bytes(pool_id):
        ref = alive.get(tuple(pool_id))
        return 2 << 20 if ref is not None and ref() is not None else 0

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda.memory, "_record_memory_history", lambda *a, **k: None)
    monkeypatch.setattr(arena_cuda, "pool_segment_bytes", segment_bytes)
    monkeypatch.setattr(arena_cuda, "pool_blocks", lambda pool_id: {"segment_bytes": segment_bytes(pool_id),
                                                                     "live_bytes": 0, "largest_live": []})
    monkeypatch.setattr(arena_cuda, "start_history", lambda: None)
    monkeypatch.setattr(arena_cuda, "stop_history", lambda pool_id: [("segment_alloc", 2 << 20, 0x1000)])

    stub = types.SimpleNamespace(capture_arena_info={}, _graphs={}, _pool=None, _pool_captures=0,
                                 _captured_kinds=set(), _process_record=types.SimpleNamespace(arena_pools=set()),
                                 kept=[], made=[])

    def make_pool(shim):
        pool = FakePool(len(stub.made) + 1)
        alive[pool.id] = weakref.ref(pool)
        stub.made.append(pool.id)
        stub._process_record.arena_pools.add(pool.id)
        return pool

    stub._arena_pool = make_pool
    stub._arena_capture_keys = lambda keys: None
    for name in ("_arena_free_pool", "_arena_reset_capture_state", "_arena_record"):
        setattr(stub, name, types.MethodType(getattr(GraphEngine, name), stub))
    return stub


def test_a_pool_only_its_holder_names_is_freed(engine):
    holder = {"pool": engine._arena_pool(None)}
    assert engine._arena_free_pool(holder) is True
    assert "pool" not in holder and engine._process_record.arena_pools == set()
    assert "pool_left" not in engine.capture_arena_info


def test_a_pool_still_named_elsewhere_is_reported_and_kept(engine):
    pool = engine._arena_pool(None)
    holder = {"pool": pool}  # the caller's own name keeps the pool alive: the 4981434/4981437 failure
    assert engine._arena_free_pool(holder) is False
    assert engine.capture_arena_info["pool_left"]["segment_bytes"] == 2 << 20
    assert pool.id in engine._process_record.arena_pools  # still counted: its memory is still reserved
    del pool
    gc.collect()


def test_a_recording_pool_that_keeps_segments_ends_the_build(engine, monkeypatch):
    """The recording returns its trace only when its pool is gone; otherwise the build falls back."""
    real_free = engine._arena_free_pool

    def free_but_keep(holder):
        engine.kept.append(holder["pool"])  # some state still holds the recording pool
        return real_free(holder)

    engine._arena_free_pool = free_but_keep
    with pytest.raises(_ArenaFallback, match="kept segments"):
        engine._arena_record(FakeShim(), ["ACP"])
    assert engine.capture_arena_info["pool_left"]["segment_bytes"] == 2 << 20


def test_a_recording_pool_with_no_other_name_returns_the_trace(engine):
    events, mode = engine._arena_record(FakeShim(), ["ACP"])
    assert mode == "virtual" and events == [("segment_alloc", 2 << 20, 0x1000)]
    assert engine._process_record.arena_pools == set()


def test_the_build_falls_back_when_the_recording_pool_is_kept(engine, monkeypatch):
    """End to end through ``_prepare_arena``: status fallback, no arena held, the claim returned."""
    shim = FakeShim()
    monkeypatch.setattr(arena_cuda, "shim", lambda: shim)
    arena_cuda._PROCESS.pop("owner", None)
    real_free = engine._arena_free_pool

    def free_but_keep(holder):
        engine.kept.append(holder["pool"])
        return real_free(holder)

    engine._arena_free_pool = free_but_keep
    engine._arena = None
    engine._arena_building = False
    engine.capture_memory = []
    engine._arena_drop = types.MethodType(GraphEngine._arena_drop, engine)
    GraphEngine._prepare_arena(engine, ["ACP"])
    info = engine.capture_arena_info
    assert info["status"] == "fallback" and "kept segments" in info["reason"], info
    assert engine._arena is None and arena_cuda._PROCESS.get("owner") is None
    assert shim.modes[-1] == arena_cuda.RECORD
