"""Temperature schedules: ramps and holds, whole steps, the target per step (SPEC §7.2, §7.6 "Schedules").

A ``Schedule`` is timestep-free: segments in ps. ``Schedule.resolve(timestep_fs)`` rounds each segment
to whole steps (half up) and records the boundaries, the total and the rounded durations and rates.
The target is piecewise linear in the step. Step ``n`` of a segment uses the target at the step's end
point, as the research engine does ("update the target every timestep using that step's endpoint",
``docs/protocol.md``), so the target changes every step on a ramp and never on a hold. It reaches the
integrator through ``Integrator.set_temperature`` before the step (``TargetFill``), never as an event.

Each ramp interpolates from its colder end, ``T_low + (T_high - T_low) * w``. This reproduces the
research engine's float expressions bit for bit: ``peak * n / H`` when heating from 0 K and
``final + (peak - final) * (H + C - n) / C`` when cooling.

``ScheduleState`` is the schedule position: the restart record of ``SimulationState.protocol``.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

KINDS = ("ramp", "hold")


def _temperature(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number of kelvin, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and at least 0 K, got {value}")
    return value


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite, got {value}")
    return value


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {value!r}")
    return value


@dataclass(frozen=True)
class Segment:
    """One ramp or hold as requested: ``ps`` or (ramps only) ``rate_K_per_ps``, never both."""

    kind: str
    start_K: float
    end_K: float
    ps: float | None = None
    rate_K_per_ps: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"segment kind must be one of {KINDS}, got {self.kind!r}")
        object.__setattr__(self, "start_K", _temperature(self.start_K, "start temperature"))
        object.__setattr__(self, "end_K", _temperature(self.end_K, "end temperature"))
        if (self.ps is None) == (self.rate_K_per_ps is None):
            raise ValueError(f"a {self.kind} needs exactly one of ps= and rate_K_per_ps=")
        if self.ps is not None:
            object.__setattr__(self, "ps", _positive(self.ps, "ps"))
            return
        if self.kind == "hold":
            raise ValueError("a hold needs ps=")
        rate = self.rate_K_per_ps
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate):
            raise ValueError(f"rate_K_per_ps must be a finite number, got {rate!r}")
        change = self.end_K - self.start_K
        if change == 0 or rate == 0 or (rate > 0) != (change > 0):
            raise ValueError(f"rate_K_per_ps={rate} does not lead from {self.start_K} K to {self.end_K} K "
                             "(the sign must follow the ramp's direction)")
        object.__setattr__(self, "rate_K_per_ps", float(rate))

    @property
    def duration_ps(self) -> float:
        """The requested duration (before rounding to whole steps)."""
        if self.ps is not None:
            return self.ps
        return (self.end_K - self.start_K) / self.rate_K_per_ps

    @property
    def phase(self) -> str:
        """``heating``, ``cooling`` or ``hold``: the thermo file's phase column (workflow.schedule_phase)."""
        if self.end_K > self.start_K:
            return "heating"
        return "cooling" if self.end_K < self.start_K else "hold"

    def to_data(self) -> dict[str, Any]:
        """The segment as profile data (``segment_from_data`` reads it back)."""
        if self.kind == "hold":
            return {"kind": "hold", "temperature_K": self.start_K, "ps": self.ps}
        data: dict[str, Any] = {"kind": "ramp", "start_K": self.start_K, "end_K": self.end_K}
        if self.ps is not None:
            data["ps"] = self.ps
        else:
            data["rate_K_per_ps"] = self.rate_K_per_ps
        return data


_SEGMENT_KEYS = {"ramp": ({"kind", "start_K", "end_K"}, {"ps", "rate_K_per_ps"}),
                 "hold": ({"kind", "temperature_K", "ps"}, set())}


def segment_from_data(data: Mapping[str, Any]) -> Segment:
    """A segment from profile or restart data: ``{kind: ramp, start_K, end_K, ps | rate_K_per_ps}`` or
    ``{kind: hold, temperature_K, ps}``. Unknown keys raise ``ValueError``."""
    if not isinstance(data, Mapping):
        raise ValueError(f"a segment must be a mapping, got {data!r}")
    kind = data.get("kind")
    if kind not in _SEGMENT_KEYS:
        raise ValueError(f"segment kind must be one of {KINDS}, got {kind!r}")
    required, optional = _SEGMENT_KEYS[kind]
    missing, unknown = required - set(data), set(data) - required - optional
    if missing or unknown:
        raise ValueError(f"{kind} segment: missing {sorted(missing)}, unknown {sorted(unknown)}")
    if kind == "hold":
        return Segment("hold", data["temperature_K"], data["temperature_K"], data["ps"])
    return Segment("ramp", data["start_K"], data["end_K"], data.get("ps"), data.get("rate_K_per_ps"))


@dataclass(frozen=True)
class Schedule:
    """A sequence of segments; ``+`` concatenates and refuses a temperature jump."""

    segments: tuple[Segment, ...]

    def __post_init__(self) -> None:
        segments = tuple(self.segments)
        if not segments or not all(isinstance(s, Segment) for s in segments):
            raise ValueError("a schedule needs at least one segment")
        for before, after in zip(segments, segments[1:]):
            if before.end_K != after.start_K:
                raise ValueError(f"temperature jump: a segment ends at {before.end_K} K and the next starts "
                                 f"at {after.start_K} K")
        object.__setattr__(self, "segments", segments)

    def __add__(self, other: object) -> Schedule:
        if not isinstance(other, Schedule):
            return NotImplemented
        return Schedule(self.segments + other.segments)

    @property
    def duration_ps(self) -> float:
        """The requested duration (before rounding)."""
        return sum(s.duration_ps for s in self.segments)

    def resolve(self, timestep_fs: float) -> ResolvedSchedule:
        """Whole steps at ``timestep_fs``."""
        return ResolvedSchedule(self, timestep_fs)

    def to_data(self) -> list[dict[str, Any]]:
        return [s.to_data() for s in self.segments]

    @classmethod
    def from_data(cls, segments: Iterable[Mapping[str, Any]]) -> Schedule:
        if isinstance(segments, (str, bytes, Mapping)) or not isinstance(segments, Iterable):
            raise ValueError(f"segments must be a list, got {segments!r}")
        return cls(tuple(segment_from_data(s) for s in segments))


def ramp(start_K: float, end_K: float, *, ps: float | None = None,
         rate_K_per_ps: float | None = None) -> Schedule:
    """A linear ramp from ``start_K`` to ``end_K``, over ``ps`` or at ``rate_K_per_ps`` (signed)."""
    return Schedule((Segment("ramp", start_K, end_K, ps, rate_K_per_ps),))


def hold(temperature_K: float, *, ps: float) -> Schedule:
    """``temperature_K`` for ``ps``."""
    return Schedule((Segment("hold", temperature_K, temperature_K, ps),))


def _whole_steps(duration_ps: float, timestep_fs: float) -> int:
    """Half up (``floor(x + 0.5)``), not Python's half to even. ``duration_ps * 1000 / timestep_fs``
    as the research engine (workflow.ramp_steps), so whole-step durations give its counts."""
    steps = math.floor(duration_ps * 1000 / timestep_fs + 0.5)
    if steps < 1:
        raise ValueError(f"a segment of {duration_ps:g} ps rounds to 0 steps at {timestep_fs:g} fs")
    return steps


@dataclass(frozen=True)
class ResolvedSchedule:
    """A schedule at a timestep: steps per segment, boundaries ``(0, end_1, ..., total)``, targets."""

    schedule: Schedule
    timestep_fs: float
    segment_steps: tuple[int, ...] = field(init=False)
    boundaries: tuple[int, ...] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        timestep = _positive(self.timestep_fs, "timestep_fs")
        object.__setattr__(self, "timestep_fs", timestep)
        steps = tuple(_whole_steps(s.duration_ps, timestep) for s in self.schedule.segments)
        object.__setattr__(self, "segment_steps", steps)
        boundaries = [0]
        for n in steps:
            boundaries.append(boundaries[-1] + n)
        object.__setattr__(self, "boundaries", tuple(boundaries))

    @property
    def total_steps(self) -> int:
        return self.boundaries[-1]

    @property
    def durations_ps(self) -> tuple[float, ...]:
        """The rounded durations."""
        return tuple(n * self.timestep_fs / 1000 for n in self.segment_steps)

    @property
    def rates_K_per_ps(self) -> tuple[float, ...]:
        """The rates after rounding (0 for holds)."""
        return tuple((s.end_K - s.start_K) / d for s, d in zip(self.schedule.segments, self.durations_ps))

    def _check(self, step: Any) -> int:
        step = _count(step, "step")
        if not 0 <= step <= self.total_steps:
            raise ValueError(f"step {step} is outside the schedule (0 to {self.total_steps})")
        return step

    def _segment(self, step: int) -> int:
        """Index of the segment whose steps (start, end] contain ``step``; step 0 is in the first."""
        return max(bisect.bisect_left(self.boundaries, step) - 1, 0)

    def target(self, step: int) -> float:
        """The thermostat target for step ``step`` (its end point); step 0 is the start temperature."""
        step = self._check(step)
        index = self._segment(step)
        segment = self.schedule.segments[index]
        k, n = step - self.boundaries[index], self.segment_steps[index]
        t0, t1 = segment.start_K, segment.end_K
        if t0 <= t1:
            return t0 + (t1 - t0) * k / n
        return t1 + (t0 - t1) * (n - k) / n

    def phase(self, step: int) -> str:
        """``heating``, ``hold`` or ``cooling``; a boundary step belongs to the segment it ends."""
        return self.schedule.segments[self._segment(self._check(step))].phase

    def targets(self, first: int, last: int) -> list[float]:
        """The targets of steps ``first`` to ``last`` inclusive."""
        return [self.target(s) for s in range(self._check(first), self._check(last) + 1)]

    def _extreme(self, first: int, last: int, pick: Callable[[Iterable[float]], float]) -> float:
        first, last = self._check(first), self._check(last)
        if last < first:
            raise ValueError(f"empty step range {first} to {last}")
        # The per-step target is monotonic within a segment, so the extremes are at the range's ends or
        # at segment boundaries inside it.
        inner = (b for b in self.boundaries if first < b < last)
        return pick(self.target(s) for s in (first, last, *inner))

    def max_target(self, first: int, last: int) -> float:
        """The highest target of steps ``first`` to ``last`` inclusive."""
        return self._extreme(first, last, max)

    def min_target(self, first: int, last: int) -> float:
        """The lowest target of steps ``first`` to ``last`` inclusive."""
        return self._extreme(first, last, min)

    def output_steps(self, interval: int) -> list[int]:
        """Step 0, every ``interval`` steps and every segment boundary (``output_steps`` for any schedule)."""
        interval = _count(interval, "interval")
        if interval < 1:
            raise ValueError(f"interval must be positive, got {interval}")
        steps = set(self.boundaries)
        steps.update(range(interval, self.total_steps + 1, interval))
        return sorted(steps)

    def record(self) -> dict[str, Any]:
        """For provenance and the run report: requested and rounded durations and rates, boundaries."""
        segments = []
        for segment, steps, ps, rate in zip(self.schedule.segments, self.segment_steps, self.durations_ps,
                                            self.rates_K_per_ps):
            segments.append({"kind": segment.kind, "phase": segment.phase, "start_K": segment.start_K,
                             "end_K": segment.end_K, "requested_ps": segment.ps,
                             "requested_rate_K_per_ps": segment.rate_K_per_ps, "steps": steps, "ps": ps,
                             "rate_K_per_ps": rate})
        return {"timestep_fs": self.timestep_fs, "total_steps": self.total_steps,
                "boundaries": list(self.boundaries), "target_update": "every step on ramps, end-point target",
                "rounding": "half up to whole steps", "segments": segments}


def output_steps(total_steps: int, interval: int, heat_steps: int, hold_steps: int = 0) -> list[int]:
    """Steps at which a sample (or trajectory frame) is written: 0, every interval, the phase ends.

    1a's function (``io/trajectory.output_steps``, workflow.output_steps) for heat, hold, cool;
    ``ResolvedSchedule.output_steps`` is the same for any schedule.
    """
    steps = {0, heat_steps, heat_steps + hold_steps, total_steps}
    steps.update(range(interval, total_steps + 1, interval))
    return sorted(steps)


class ScheduleState:
    """The schedule position: what ``SimulationState.protocol`` holds and the restart stores (§10.6).

    ``step`` counts the schedule's steps done (0 to ``total_steps``), not the simulation's step.
    """

    def __init__(self, schedule: Schedule, timestep_fs: float, step: int = 0) -> None:
        if not isinstance(schedule, Schedule):
            raise TypeError(f"schedule must be a Schedule, got {type(schedule).__name__}")
        self.schedule = schedule
        self.resolved = schedule.resolve(timestep_fs)
        self.timestep_fs = self.resolved.timestep_fs
        self.step = self.resolved._check(step)

    @property
    def total_steps(self) -> int:
        return self.resolved.total_steps

    @property
    def remaining(self) -> int:
        return self.total_steps - self.step

    @property
    def finished(self) -> bool:
        return self.remaining == 0

    @property
    def target_K(self) -> float:
        """The target of the current (last completed) step."""
        return self.resolved.target(self.step)

    def steps_to_run(self, steps: int | None = None) -> int:
        """The steps a ``run(steps=...)`` takes: all that remain for ``None``. Raises before any step
        for a finished schedule or ``steps`` beyond the remainder (SPEC §7.2); never truncates."""
        if self.finished:
            raise ValueError("the schedule is finished and has no remaining steps; pass a new schedule=")
        if steps is None:
            return self.remaining
        steps = _count(steps, "steps")
        if steps < 1:
            raise ValueError(f"steps must be positive, got {steps}")
        if steps > self.remaining:
            raise ValueError(f"steps={steps} is beyond the schedule: {self.remaining} of its "
                             f"{self.total_steps} steps remain")
        return steps

    def next_target(self) -> float:
        """The target of the next step, to fill before it."""
        self.steps_to_run(1)
        return self.resolved.target(self.step + 1)

    def max_target(self, steps: int | None = None) -> float:
        """The highest target of the next ``steps`` steps (default: all that remain)."""
        steps = self.steps_to_run(steps)
        return self.resolved.max_target(self.step + 1, self.step + steps)

    def min_target(self, steps: int | None = None) -> float:
        """The lowest target of the next ``steps`` steps (default: all that remain)."""
        steps = self.steps_to_run(steps)
        return self.resolved.min_target(self.step + 1, self.step + steps)

    def advance(self, steps: int) -> None:
        """Record ``steps`` completed steps."""
        self.step += self.steps_to_run(steps)

    def to_restart(self) -> dict[str, Any]:
        """JSON-safe: the segments (with their rounded steps), the timestep, the position, the target."""
        segments = [{**s.to_data(), "steps": n}
                    for s, n in zip(self.schedule.segments, self.resolved.segment_steps)]
        return {"segments": segments, "timestep_fs": self.timestep_fs, "step": self.step,
                "target_K": self.target_K}

    @classmethod
    def from_restart(cls, data: Mapping[str, Any]) -> ScheduleState:
        """The inverse of ``to_restart``; refuses a record whose steps or target do not match."""
        segments = [dict(s) for s in data["segments"]]
        stored_steps = [s.pop("steps", None) for s in segments]
        state = cls(Schedule.from_data(segments), data["timestep_fs"], data["step"])
        if stored_steps != list(state.resolved.segment_steps):
            raise ValueError(f"restart schedule: stored segment steps {stored_steps} differ from the "
                             f"resolved {list(state.resolved.segment_steps)}")
        if data["target_K"] != state.target_K:
            raise ValueError(f"restart schedule: stored target {data['target_K']} K differs from "
                             f"{state.target_K} K at step {state.step}")
        return state

    def __repr__(self) -> str:
        return (f"ScheduleState(step={self.step}/{self.total_steps}, target_K={self.target_K:g}, "
                f"timestep_fs={self.timestep_fs:g})")


class TargetFill:
    """Fills ``integrator.set_temperature`` only where the target changes.

    The first call always fills: the device target is derived from the protocol position, also after
    a restart. ``reset()`` forces the next fill (after something else set the target).
    """

    def __init__(self, integrator: Any) -> None:
        self.integrator = integrator
        self._last: float | None = None

    def __call__(self, target_K: float) -> None:
        if self._last is None or target_K != self._last:
            self.integrator.set_temperature(target_K)
            self._last = target_K

    def reset(self) -> None:
        self._last = None
