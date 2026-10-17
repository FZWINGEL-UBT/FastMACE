"""Alternating A/B step-time comparison in one process (SPEC §12.5) and the baseline export it compares with.

Cross-process step times scatter by about 3 %, so small effects are compared in one process: both engines
are built and warmed up first, then their timed blocks alternate (A B A B ... or A B B A ...), and the
ratio of the medians is the result. The per-pair ratios show how far one block pair can stray.

The other side of a comparison with the research engine is a frozen revision of it, exported from
ResearchModule with ``git archive`` into a directory of its own (AGENTS.md, "Baseline for A/B"). That
directory must record the commit it was exported from in a file ``COMMIT``; ``load_baseline`` refuses it
otherwise, so every A/B result names the revision it was measured against. This module never imports the
research engine: ``activate`` only puts the export on ``sys.path``, and the tool that builds the research
engine (``tools/ab_step_time.py``) imports it from there at run time and checks with ``check_origin`` that
the modules came from the export.
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, MutableSequence, Sequence

from fastmace.bench.step_time import BlockStatistics, TimedEngine, TimingProtocol, time_block, warm_up
from fastmace.errors import FastMACEError

COMMIT_FILE = "COMMIT"
ENGINE_DIRECTORY = Path("Experiments", "Langevin_Thermostat")
BENCHMARK_SCRIPTS_DIRECTORY = Path("Experiments", "Benchmark_MACE_GPU_OPTIMIZATIONS", "scripts")
ENGINE_PACKAGE = "langevin"
_COMMIT = re.compile(r"[0-9a-f]{7,40}")
PATTERNS = ("abab", "abba")


class BaselineExportError(FastMACEError):
    """A baseline directory is not a usable export of the research engine."""


@dataclass(frozen=True)
class BaselineExport:
    """An exported research-engine revision: its directory and the commit it was exported from."""

    root: Path
    commit: str

    @property
    def engine_root(self) -> Path:
        return self.root / ENGINE_DIRECTORY

    @property
    def benchmark_scripts(self) -> Path:
        return self.root / BENCHMARK_SCRIPTS_DIRECTORY

    def as_dict(self):
        return dict(root=str(self.root), commit=self.commit)


def load_baseline(path) -> BaselineExport:
    """The export at ``path``; refuses a directory without the engine or without a recorded commit."""
    root = Path(path).expanduser()
    if not root.is_dir():
        raise BaselineExportError(f"baseline export {root} is not a directory")
    root = root.resolve()
    commit_file = root / COMMIT_FILE
    if not commit_file.is_file():
        raise BaselineExportError(
            f"baseline export {root} has no {COMMIT_FILE} file: write the commit it was exported from into it")
    commit = commit_file.read_text().strip()
    if not _COMMIT.fullmatch(commit):
        raise BaselineExportError(
            f"{commit_file} must hold a commit hash (7 to 40 lowercase hex digits), not {commit!r}")
    if not (root / ENGINE_DIRECTORY / ENGINE_PACKAGE / "__init__.py").is_file():
        raise BaselineExportError(f"baseline export {root} has no {ENGINE_DIRECTORY / ENGINE_PACKAGE} package")
    return BaselineExport(root=root, commit=commit)


def activate(export: BaselineExport, path: MutableSequence[str] = sys.path) -> None:
    """Put the export's engine and benchmark-script directories first on ``path`` (once)."""
    for entry in reversed((str(export.engine_root), str(export.benchmark_scripts))):
        while entry in path:
            path.remove(entry)
        path.insert(0, entry)


def check_origin(export: BaselineExport, name: str, file: str | None) -> None:
    """Refuse a module ``name`` (with ``__file__`` ``file``) that was not imported from the export."""
    if file is None or export.root not in Path(file).resolve().parents:
        raise BaselineExportError(f"module {name} was imported from {file}, outside the baseline export "
                                  f"{export.root}; import it only after activate() and before anything else")


def block_order(names: Sequence[str], blocks: int, pattern: str = "abab") -> list[str]:
    """The order of the timed blocks of two sides: ``abab`` alternates, ``abba`` mirrors every pair."""
    if len(names) != 2 or names[0] == names[1]:
        raise ValueError(f"A/B needs two distinct sides, not {tuple(names)}")
    if blocks < 1:
        raise ValueError("at least one block per side")
    if pattern not in PATTERNS:
        raise ValueError(f"pattern must be one of {PATTERNS}, not {pattern!r}")
    a, b = names
    if pattern == "abab":
        return [a, b] * blocks
    return [name for index in range(blocks) for name in ((a, b) if index % 2 == 0 else (b, a))]


@dataclass(frozen=True)
class ABResult:
    """Block statistics of both sides, the block order, and the ratio of B to A."""

    names: tuple[str, str]
    order: tuple[str, ...]
    statistics: Mapping[str, BlockStatistics]
    pattern: str

    @property
    def ratio(self) -> float:
        """Median of B over median of A."""
        a, b = self.names
        return self.statistics[b].median_ms / self.statistics[a].median_ms

    @property
    def pair_ratios(self) -> tuple[float, ...]:
        """B over A for each pair of blocks (the i-th block of each side)."""
        a, b = self.names
        return tuple(y / x for x, y in zip(self.statistics[a].block_ms, self.statistics[b].block_ms))

    def as_dict(self):
        a, b = self.names
        return dict(names=list(self.names), pattern=self.pattern, order=list(self.order),
                    sides={name: self.statistics[name].as_dict() for name in self.names},
                    ratio_b_over_a=self.ratio, pair_ratios_b_over_a=list(self.pair_ratios))


def run_ab(sides: Mapping[str, TimedEngine], protocol: TimingProtocol = TimingProtocol(), pattern: str = "abab",
           clock: Callable[[], float] = time.perf_counter) -> ABResult:
    """Warm up both sides, then time ``protocol.blocks`` blocks of each, alternating by ``pattern``."""
    names = tuple(sides)
    order = block_order(names, protocol.blocks, pattern)
    for name in names:
        warm_up(sides[name], protocol.warmup)
    blocks = {name: [] for name in names}
    for name in order:
        blocks[name].append(time_block(sides[name], protocol.block, clock))
    return ABResult(names=names, order=tuple(order), pattern=pattern,
                    statistics={name: BlockStatistics.from_blocks(blocks[name]) for name in names})
