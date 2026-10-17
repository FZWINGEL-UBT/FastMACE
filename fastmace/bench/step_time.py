"""MD step time (SPEC §12.5): median of 4 blocks of 300 steps after 300 warm-up steps.

Copied from the research engine's ``model_step_time.time_engine`` and refactored: the engine is anything
with the four methods of ``TimedEngine``, the clock can be replaced (tests), and the statistics are a
record of their own. Each block starts and ends on a synchronised GPU, so a block measures exactly its
own steps; ``finish_block`` drains work an engine keeps in flight (pipelined or speculative steps).
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol, Sequence

# Version of the JSON records written by the bench tools; raise it when a field changes meaning.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TimingProtocol:
    """Steps before timing, steps per timed block, and the number of blocks."""

    warmup: int = 300
    block: int = 300
    blocks: int = 4

    def __post_init__(self):
        if self.warmup < 0 or self.block < 1 or self.blocks < 1:
            raise ValueError(f"warm-up must be >= 0 and block and blocks >= 1, not {self}")

    def as_dict(self):
        return dict(warmup=self.warmup, block=self.block, blocks=self.blocks)


class TimedEngine(Protocol):
    """What the timing loop needs from an engine."""

    def step(self) -> None:
        """One MD step at the run's target temperature."""

    def prepare_block(self) -> None:
        """Called before each timed block (the research engine rebinds its streaming counter here)."""

    def finish_block(self) -> None:
        """Complete every step launched so far (pipelined and speculative work)."""

    def synchronize(self) -> None:
        """Wait for the device."""


@dataclass(frozen=True)
class BlockStatistics:
    """Per-step times of the timed blocks, in ms, and their summary. ``spread`` is (max - min) / median."""

    block_ms: tuple[float, ...]
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float
    spread: float

    @classmethod
    def from_blocks(cls, block_ms: Sequence[float]) -> BlockStatistics:
        blocks = tuple(float(value) for value in block_ms)
        if not blocks:
            raise ValueError("no timed blocks")
        median = statistics.median(blocks)
        return cls(block_ms=blocks, median_ms=median, mean_ms=statistics.fmean(blocks), min_ms=min(blocks),
                   max_ms=max(blocks), stdev_ms=statistics.stdev(blocks) if len(blocks) > 1 else 0.0,
                   spread=(max(blocks) - min(blocks)) / median if median else 0.0)

    def as_dict(self):
        return dict(median_ms=self.median_ms, mean_ms=self.mean_ms, min_ms=self.min_ms, max_ms=self.max_ms,
                    stdev_ms=self.stdev_ms, spread=self.spread, block_ms=list(self.block_ms))


def warm_up(engine: TimedEngine, steps: int) -> None:
    """Run ``steps`` untimed steps and leave the device idle."""
    for _ in range(steps):
        engine.step()
    engine.finish_block()
    engine.synchronize()


def time_block(engine: TimedEngine, steps: int, clock: Callable[[], float] = time.perf_counter) -> float:
    """Milliseconds per step of one block of ``steps`` steps."""
    engine.prepare_block()
    engine.synchronize()
    started = clock()
    for _ in range(steps):
        engine.step()
    engine.finish_block()
    engine.synchronize()
    return (clock() - started) / steps * 1e3


def time_engine(engine: TimedEngine, protocol: TimingProtocol = TimingProtocol(),
                clock: Callable[[], float] = time.perf_counter) -> BlockStatistics:
    """Warm up, then time ``protocol.blocks`` blocks one after another."""
    warm_up(engine, protocol.warmup)
    return BlockStatistics.from_blocks([time_block(engine, protocol.block, clock) for _ in range(protocol.blocks)])


def result_record(stats: BlockStatistics, *, setup_s: float | None = None, peak_memory_bytes: int | None = None,
                  **extra) -> dict:
    """A versioned JSON record of one timed configuration: step time, setup time and peak GPU memory."""
    return dict(schema_version=SCHEMA_VERSION, created_utc=datetime.now(timezone.utc).isoformat(),
                **stats.as_dict(), setup_s=setup_s, peak_memory_bytes=peak_memory_bytes, **extra)
