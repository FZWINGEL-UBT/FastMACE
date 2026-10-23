"""Runtime adaptation (SPEC §9.6, §13; plan tasks 5-T06 and 5-T07).

Plan task 5-T06 adds the monitor. ``Monitor`` reads the engine's cumulative counters (the graph engine's
``runtime_counters()``, or any engine's ``info()``, plus the D3 term's ``counters()`` under ``"d3"``) at
each canonical point and at the end of each ``run()`` call, and keeps the difference since the
previous reading as a ``Segment``:

- re-captures (counted against the budget) and the host time they took, all captures and theirs;
- overflows redone, and the step the re-capture budget was first exceeded at;
- per neighbour list the rebuilds and the build time: MACE's list is rebuilt on a side stream
  (``overlap_rebuild``), so its host time is not its build time and stays ``None``; D3's list keeps the
  host time its builds measure.

``Simulation`` takes the readings only at moments that are synchronised anyway (a restart write, the end
of an MC block, a host write's re-evaluation, the end of ``run()``), so the step path gains no
synchronisation. A segment's wall time is the run's loop clock between two readings.

The monitor keeps the last ``SEGMENT_LIMIT`` segments (the totals keep counting), the window since the
last ``mark`` for 5-T07's trigger, and ``decide``, which returns ``None`` until 5-T07 adds the trigger.
It starts empty in every process: a resume starts a new monitor, and the restart format does not
change (phase 5 conductor's choice 5). Under the subprocess runner the child's monitor ``export``s the
segments of each call, and the parent's ``absorb``s them.
"""

from __future__ import annotations

import dataclasses
from collections import deque
from collections.abc import Mapping
from typing import Any

from fastmace.config import AdaptationConfig

# Where a segment ends: a restart write, an MC block, a host write (canonical points, §10.6), or the end
# of a run() call (not canonical: its counters are read, nothing is canonicalised).
SEGMENT_REASONS = ("restart", "mc-block", "host-write", "run-end")
# Segments kept (``counters.segments``); the totals keep counting past it.
SEGMENT_LIMIT = 1000
# The neighbour lists a segment reports: MACE's always, D3's when the force field has the term.
LISTS = ("mace", "d3")
# Segment fields that add up, and the engine counters they come from.
_SUMS = (("recaptures", "recaptures"), ("recapture_s", "recapture_seconds"), ("captures", "captures"),
         ("capture_s", "capture_seconds"), ("overflows_redone", "overflows_redone"))


@dataclasses.dataclass(frozen=True)
class Segment:
    """The counters between two readings: steps ``first_step`` (exclusive) to ``last_step``.

    ``reason`` and ``canonical`` say where it ended; a sum of segments (``Monitor.window``) keeps the last
    one's. ``lists`` maps ``"mace"`` (and ``"d3"``) to ``{rebuilds, build_time_s}``.
    """

    first_step: int
    last_step: int
    reason: str
    canonical: bool
    wall_s: float
    recaptures: int = 0
    recapture_s: float = 0.0
    captures: int = 0
    capture_s: float = 0.0
    overflows_redone: int = 0
    budget_exceeded_at: int | None = None
    lists: Mapping[str, Mapping[str, Any]] = dataclasses.field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        """``counters.segments``' entry (``provenance.SEGMENT_KEYS``)."""
        return {"first_step": self.first_step, "last_step": self.last_step, "reason": self.reason,
                "canonical": self.canonical, "wall_s": self.wall_s, "recaptures": self.recaptures,
                "recapture_s": self.recapture_s, "captures": self.captures, "capture_s": self.capture_s,
                "overflows_redone": self.overflows_redone, "budget_exceeded_at": self.budget_exceeded_at,
                "lists": {name: dict(counts) for name, counts in self.lists.items()}}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Segment:
        fields = dict(record)
        fields["lists"] = {name: dict(counts) for name, counts in fields["lists"].items()}
        return cls(**fields)


def _reading(info: Mapping[str, Any]) -> dict[str, Any]:
    """The cumulative counters a segment is the difference of; a counter the engine lacks reads as 0."""
    out = {field: info.get(key) or 0 for field, key in _SUMS}
    lists = {"mace": {"rebuilds": int(info.get("neighbour_rebuilds") or 0), "build_time_s": None}}
    d3 = info.get("d3")
    if d3:
        lists["d3"] = {"rebuilds": int(d3.get("rebuilds") or 0),
                       "build_time_s": float(d3.get("build_seconds") or 0.0)}
    out["lists"] = lists
    out["budget_exceeded_at"] = info.get("recapture_budget_exceeded_at")
    return out


def _lists(after, before, sign=-1):
    out = {}
    for name in LISTS:
        if name not in after:
            continue
        counts = after[name]
        other = before.get(name, {"rebuilds": 0, "build_time_s": None if counts["build_time_s"] is None else 0.0})
        time_s = None
        if counts["build_time_s"] is not None and other["build_time_s"] is not None:
            time_s = counts["build_time_s"] + sign * other["build_time_s"]
        out[name] = {"rebuilds": counts["rebuilds"] + sign * other["rebuilds"], "build_time_s": time_s}
    return out


def combine(first: Segment | None, second: Segment) -> Segment:
    """``first`` followed by ``second``: the sums, ``first``'s start, ``second``'s end and reason."""
    if first is None:
        return second
    sums = {field: getattr(first, field) + getattr(second, field) for field, _ in _SUMS}
    lists = _lists(second.lists, first.lists, sign=1)
    for name, counts in first.lists.items():
        lists.setdefault(name, dict(counts))
    exceeded = first.budget_exceeded_at if first.budget_exceeded_at is not None else second.budget_exceeded_at
    return Segment(first.first_step, second.last_step, second.reason, second.canonical,
                   first.wall_s + second.wall_s, **sums, budget_exceeded_at=exceeded, lists=lists)


def _minus(after: Segment, before: Segment | None) -> Segment | None:
    """The segments ``after`` sums beyond ``before`` (both sums from the same start)."""
    if before is None:
        return after
    if after.last_step == before.last_step:
        return None
    sums = {field: getattr(after, field) - getattr(before, field) for field, _ in _SUMS}
    exceeded = after.budget_exceeded_at if before.budget_exceeded_at is None else None
    return Segment(before.last_step, after.last_step, after.reason, after.canonical, after.wall_s - before.wall_s,
                   **sums, budget_exceeded_at=exceeded, lists=_lists(after.lists, before.lists))


class Monitor:
    """The runtime monitor of one process's engine (module docstring). ``adaptation`` holds the limits and
    thresholds 5-T07's trigger reads; ``"off"`` still records."""

    def __init__(self, adaptation: AdaptationConfig | None = None):
        self.adaptation = AdaptationConfig() if adaptation is None else adaptation
        self.closed = 0  # segments closed (or absorbed), the dropped ones included
        self._segments = deque(maxlen=SEGMENT_LIMIT)
        self._totals = None
        self._previous = None  # (step, wall_s, reading) of the last reading
        self._mark = None  # the window's start (step), or None: the whole run

    @property
    def started(self) -> bool:
        return self._previous is not None

    @property
    def segments(self) -> tuple[Segment, ...]:
        return tuple(self._segments)

    def start(self, info: Mapping[str, Any], *, step: int, wall_s: float) -> None:
        """Take the baseline: the counters at ``step`` (the first run's start, or a new engine's) count from
        here, the setup's captures not included."""
        self._previous = (int(step), float(wall_s), _reading(info))

    def observe(self, info: Mapping[str, Any], *, step: int, wall_s: float, reason: str,
                canonical: bool = True) -> Segment | None:
        """Close the segment since the previous reading at ``step``; ``None`` if no step passed (its counters
        carry into the next segment) or if this is the first reading (the baseline)."""
        if reason not in SEGMENT_REASONS:
            raise ValueError(f"segment reason must be one of {SEGMENT_REASONS}, got {reason!r}")
        if self._previous is None:
            self.start(info, step=step, wall_s=wall_s)
            return None
        first, started, before = self._previous
        if int(step) == first:
            return None
        after = _reading(info)
        sums = {field: after[field] - before[field] for field, _ in _SUMS}
        exceeded = after["budget_exceeded_at"]
        segment = Segment(first, int(step), reason, bool(canonical), float(wall_s) - started, **sums,
                          budget_exceeded_at=exceeded, lists=_lists(after["lists"], before["lists"]))
        self._previous = (int(step), float(wall_s), after)
        self._close(segment)
        return segment

    def _close(self, segment):
        self._segments.append(segment)
        self._totals = combine(self._totals, segment)
        self.closed += 1

    def decide(self, window: Segment | None) -> None:
        """What to do at this point; 5-T06 decides nothing (5-T07 adds the trigger, SPEC §9.6)."""
        return None

    def mark(self, step: int) -> None:
        """Start the window at ``step`` (5-T07 marks a tier switch)."""
        self._mark = int(step)

    def window(self, since_step: int | None = None) -> Segment | None:
        """The sum of the kept segments that start at or after ``since_step`` (default: the mark, or the
        whole run); ``None`` if there is none."""
        since = self._mark if since_step is None else int(since_step)
        out = None
        for segment in self._segments:
            if since is None or segment.first_step >= since:
                out = combine(out, segment)
        return out

    def totals(self) -> Segment | None:
        """Every segment closed (or absorbed) in this process, the dropped ones included."""
        return self._totals

    def records(self) -> list[dict[str, Any]]:
        """``counters.segments``: the kept segments' records."""
        return [segment.record() for segment in self._segments]

    def neighbour_lists(self, *, mace_skin_A: Any = None, d3_skin_A: Any = None) -> dict[str, Any] | None:
        """``counters.neighbour_lists``: per list the total rebuilds and build time, and its skin in Å;
        ``None`` before the first segment."""
        if self._totals is None:
            return None
        skins = {"mace": mace_skin_A, "d3": d3_skin_A}
        return {name: {**counts, "skin_A": skins[name]} for name, counts in self._totals.lists.items()}

    # ------------------------------------------------------------------ the subprocess runner (§8.7)

    def cursor(self) -> tuple[int, Segment | None]:
        """Where a child's call starts: ``export(cursor)`` returns what it closed since."""
        return self.closed, self._totals

    def export(self, cursor: tuple[int, Segment | None]) -> dict[str, Any]:
        """The segments closed since ``cursor`` (the kept ones) and their sum, JSON-safe for the reply."""
        closed, totals = cursor
        count = min(self.closed - closed, len(self._segments))
        segments = list(self._segments)[len(self._segments) - count:] if count else []
        delta = None if self._totals is None else _minus(self._totals, totals)
        return {"closed": self.closed - closed, "segments": [segment.record() for segment in segments],
                "totals": None if delta is None else delta.record()}

    def absorb(self, exported: Mapping[str, Any] | None) -> None:
        """Add a child's ``export``: its segments to the kept ones, its sum to the totals."""
        if not exported or not exported.get("closed"):
            return
        self._segments.extend(Segment.from_record(record) for record in exported["segments"])
        self._totals = combine(self._totals, Segment.from_record(exported["totals"]))
        self.closed += int(exported["closed"])


def report_line(totals: Segment | None, budget: int | None = None) -> str | None:
    """The report's line (§7.7, measured values only), or ``None`` when no re-capture and no overflow
    happened."""
    if totals is None or not (totals.recaptures or totals.overflows_redone):
        return None
    parts = []
    if totals.recaptures:
        details = [f"{totals.recapture_s:.3g} s"]
        if totals.wall_s > 0:
            details[0] += f", {100.0 * totals.recapture_s / totals.wall_s:.1f} % of the loop"
        if totals.budget_exceeded_at is not None:
            budget_text = f"budget {budget}" if budget is not None else "budget"
            details.append(f"{budget_text} exceeded at step {totals.budget_exceeded_at:,}")
        parts.append(f"re-captures {totals.recaptures} ({'; '.join(details)})")
    if totals.overflows_redone:
        parts.append(f"overflows redone {totals.overflows_redone}")
    return "; ".join(parts)


__all__ = ["LISTS", "SEGMENT_LIMIT", "SEGMENT_REASONS", "Monitor", "Segment", "combine", "report_line"]
