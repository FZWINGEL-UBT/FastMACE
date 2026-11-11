"""The CUDA side of the capture arena (``EngineConfig.capture_arena``; the planner is ``fastmace.engine.arena``).

The graph pool sits on ``torch.cuda.MemPool`` over a small C allocator (``arena_shim.c``, built with the system C
compiler into ``FASTMACE_CACHE/arena`` on first use) that can (0) record its segment requests, (1) serve them at
planned offsets of one arena, or (2) hand out virtual address ranges with no memory behind them, for a recording
capture that is never replayed. ``GraphEngine.prepare`` drives the build (record, plan, planned capture, re-record
and plan check, correctness check) and falls back to the caching pool at the first failure.

One arena per process: the shim's state is the process's.
"""

from __future__ import annotations

import ctypes
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "arena_shim.c"
RECORD, PLANNED, VIRTUAL = 0, 1, 2
HEADROOM_BYTES = 1 << 30  # the shim refuses device memory beyond the free memory minus this
CANARY = 0xFF  # the virtual mode's shared granule: NaN as float, -1 as an integer, so a read of it also shows
POOL_ACTIONS = ("alloc", "free_completed", "segment_alloc", "segment_free", "segment_map", "segment_unmap")
HISTORY_ENTRIES = 2_000_000
_PROCESS: dict[str, Any] = {}  # "shim": the loaded Shim; "owner": the id of the engine holding the arena


class ArenaUnavailable(RuntimeError):
    """The arena cannot be used here (no compiler, not Linux, WSL, another engine holds it); the caching pool is
    used instead."""


def _on_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def build(cache: Path | None = None) -> Path:
    """Compile ``arena_shim.c`` into the cache (once per source text); returns the shared library."""
    from fastmace import _cache

    source = SOURCE.read_bytes()
    digest = hashlib.sha256(source).hexdigest()[:16]
    out = (cache or _cache.cache_dir("arena")) / digest
    so = out / "arena_shim.so"
    if so.exists():
        return so
    compiler = shutil.which("gcc") or shutil.which("cc")
    if compiler is None:
        raise ArenaUnavailable("no C compiler (gcc or cc) for the arena allocator")
    out.mkdir(parents=True, exist_ok=True)
    partial = out / f"arena_shim.{id(out)}.so"
    result = subprocess.run([compiler, "-O2", "-shared", "-fPIC", "-o", str(partial), str(SOURCE), "-ldl"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise ArenaUnavailable(f"building the arena allocator failed: {result.stderr.strip()[:500]}")
    partial.replace(so)
    return so


class Shim:
    """ctypes view of the loaded allocator."""

    def __init__(self, so: Path):
        self.so = str(so)
        lib = ctypes.CDLL(self.so)
        for name in ("arena_log_count", "arena_mismatches", "arena_fallback_bytes", "arena_planned_served",
                     "arena_refusals", "arena_virtual_live", "arena_set_headroom", "arena_uncaptured"):
            getattr(lib, name).restype = ctypes.c_int64
        lib.arena_set_headroom.argtypes = [ctypes.c_int64]
        lib.arena_alias_fill.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.arena_alias_intact.argtypes = [ctypes.c_int]
        lib.arena_base.restype = ctypes.c_uint64
        lib.arena_log_get.argtypes = [ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
                                      ctypes.POINTER(ctypes.c_uint64)]
        lib.arena_set_plan.argtypes = [ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
                                       ctypes.c_int64]
        self.lib = lib
        lib.arena_set_headroom(HEADROOM_BYTES)
        self.allocator = torch.cuda.memory.CUDAPluggableAllocator(self.so, "arena_malloc", "arena_free")

    def mode(self, mode: int) -> None:
        self.lib.arena_mode(int(mode))

    def clear_log(self) -> None:
        self.lib.arena_log_clear()

    def log(self) -> list[tuple[str, int, int]]:
        kind, size, ptr = ctypes.c_int64(), ctypes.c_int64(), ctypes.c_uint64()
        out = []
        for i in range(self.lib.arena_log_count()):
            self.lib.arena_log_get(i, ctypes.byref(kind), ctypes.byref(size), ctypes.byref(ptr))
            out.append(("malloc" if kind.value == 1 else "free", int(size.value), int(ptr.value)))
        return out

    def set_plan(self, sizes, offsets, total: int) -> None:
        n = len(sizes)
        code = self.lib.arena_set_plan(n, (ctypes.c_int64 * max(n, 1))(*sizes),
                                       (ctypes.c_int64 * max(n, 1))(*offsets), int(total))
        if code != 0:
            raise RuntimeError(f"arena_set_plan returned {code} (-2: the previous arena is still held)")

    def counters(self) -> dict[str, int]:
        lib = self.lib
        return {"mismatches": int(lib.arena_mismatches()), "fallback_bytes": int(lib.arena_fallback_bytes()),
                "planned_served": int(lib.arena_planned_served()), "refusals": int(lib.arena_refusals()),
                "virtual_live": int(lib.arena_virtual_live()), "uncaptured": int(lib.arena_uncaptured()),
                "base": int(lib.arena_base())}

    def alias_fill(self, device: int, value: int = CANARY) -> None:
        """Fill the shared granule of the virtual mode with ``value`` and reset the uncaptured count."""
        if self.lib.arena_alias_fill(int(device), int(value)) != 0:
            raise RuntimeError("the arena allocator could not map or fill its shared granule")

    def alias_intact(self, value: int = CANARY) -> bool:
        """The shared granule still holds ``value`` everywhere: no kernel wrote to the recording pool."""
        return self.lib.arena_alias_intact(int(value)) == 1

    def release(self) -> None:
        self.lib.arena_release()


def shim(cache: Path | None = None) -> Shim:
    """The process's allocator, built and loaded on first use. Refused off Linux and on WSL (bugchecks of the
    local 5060 Ti with the stage-0 shim, results/capture-arena-20261110; lab runs there use the guarded lab
    script)."""
    loaded = _PROCESS.get("shim")
    if loaded is not None:
        return loaded
    import sys

    if not sys.platform.startswith("linux"):
        raise ArenaUnavailable("the capture arena needs Linux")
    if _on_wsl():
        raise ArenaUnavailable("the capture arena is refused on WSL (HYPERVISOR_ERROR bugchecks, "
                               "results/capture-arena-20261110)")
    loaded = Shim(build(cache))
    _PROCESS["shim"] = loaded
    return loaded


def claim(owner: int) -> None:
    holder = _PROCESS.get("owner")
    if holder is not None and holder != owner:
        raise ArenaUnavailable("another engine of this process holds the capture arena")
    _PROCESS["owner"] = owner


def unclaim(owner: int) -> None:
    if _PROCESS.get("owner") == owner:
        _PROCESS.pop("owner", None)


def start_history() -> None:
    torch.cuda.memory._record_memory_history(enabled="all", context=None, stacks="python",
                                             max_entries=HISTORY_ENTRIES)


def stop_history(pool_id) -> list[tuple[str, int, int]]:
    """Stop the allocator's history and return the pool's block-level trace as (action, size, addr)."""
    torch.cuda.synchronize()
    snapshot = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    device = torch.cuda.current_device()
    traces = snapshot.get("device_traces") or []
    events = []
    for event in traces[device] if device < len(traces) else ():
        if tuple(event.get("pool_id") or ()) != tuple(pool_id):
            continue
        if event["action"] in POOL_ACTIONS:
            events.append((event["action"], int(event["size"]), int(event["addr"])))
    if len(events) >= HISTORY_ENTRIES:
        raise RuntimeError("the allocator history overflowed; the pool's trace is incomplete")
    return events


def pool_segment_bytes(pool_id) -> int:
    pool = tuple(pool_id)
    return sum(segment["total_size"] for segment in torch.cuda.memory_snapshot()
               if tuple(segment.get("segment_pool_id") or ()) == pool)


def max_difference(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, float]:
    """Per-tensor maximum absolute difference (0.0: bitwise equal; NaN-aware: unequal NaN patterns are inf)."""
    out = {}
    for name in sorted(a):
        x, y = a[name], b[name]
        if torch.equal(x, y):
            out[name] = 0.0
        elif x.shape != y.shape or not torch.equal(torch.isnan(x), torch.isnan(y)):
            out[name] = float("inf")
        else:
            d = (x.double() - y.double()).abs()
            d = torch.nan_to_num(d, nan=0.0)
            out[name] = float(d.max())
    return out


__all__ = ["ArenaUnavailable", "PLANNED", "RECORD", "Shim", "VIRTUAL", "build", "claim", "max_difference",
           "pool_segment_bytes", "shim", "start_history", "stop_history", "unclaim"]
