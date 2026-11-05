"""The near-limit memory probe (``EngineConfig.memory_probe``; docs/validation/capacity/estimate-probe.md).

Rule 12's estimate stays at or above the measured peak at every stage of the memory rescue, but by up to 1.25
times: the caching allocator's reserve above the live peak (the graph pool's packing) differs between crystals by
that much at one stage, and no term predicts it (docs/validation/capacity/estimate-band.md). Within one crystal and stage it
is stable across sizes, so a small replica of the structure measures it:

- **when** (``stages_to_probe``): ``"near-limit"`` (``"auto"``'s resolution for MD on CUDA) probes only where rule
  12's choice is ambiguous, that is, where the chosen stage's estimate is within ``PROBE_SPREAD`` of decision 51's
  limit, or where the next-cheaper stage (the chosen one without its last rung) might fit. ``"always"`` probes
  the chosen and the next-cheaper stage of every plan with a memory choice. Most plans pay nothing;
- **the replica** (``replica``): the structure's own repeat unit (its translational period, found from the
  positions), repeated to ``REPLICA_ATOMS``; the cell itself when it is no larger. A structure without a periodic
  repeat unit (a thermalised start) is not probed;
- **the measurement**: each probed stage in a fresh child of the subprocess runner (``SubprocessRunner.
  memory_probe``), with the stage's rescue records set and nothing else changed, ``PROBE_STEPS`` MD steps; the
  child returns rule 12's estimate of the replica and its peak reserved and allocated memory over build and steps;
- **the correction**: rule 12 re-plans the run with each probed stage's estimate corrected (``memory.probed``):
  the fixed part (``memory.fixed_bytes``) stays, the rest is scaled by the replica's measured over estimated peak
  (both less the fixed part) and kept ``memory.PROBE_HEADROOM`` above. A stage it did not probe keeps its estimate;
- **the record** (``Plan.memory_probe``, ``MemoryProbeRecord``): every probed stage, the choice before and after,
  the cost; in the plan text, the report and provenance. The scales are kept in ``simulation.json``, so a resume
  plans with them and never probes.

A plan that runs in a child process (SPEC §8.7), a relaxation, a resume and a probe's own child never probe. A
probe that fails ends the probing: the rules' plan stays and the report says why.

The constants are this module's and ``planner.memory``'s: no per-GPU or per-structure value, nothing learnt.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastmace.planner.memory import LadderStage, MemoryLadder
    from fastmace.planner.plan import MemoryProbeRecord, Plan

# The replica's size: at 15,000 atoms and more the reserve's ratio to the live peak was stable across sizes in the
# rescue matrix; below it the build's fixed transient sets the peak (docs/validation/capacity/estimate-band.md).
REPLICA_ATOMS = (15_000, 20_000)
# MD steps a probe runs after its build: the captures and the first padded-size growth are in them.
PROBE_STEPS = 20
# How far rule 12's estimate may lie above the measured peak at one stage (the rescue matrix's highest stage band,
# s4, 1.24): a choice whose estimate is within this factor of the limit is ambiguous.
PROBE_SPREAD = 1.25
# Positions that coincide after a translation of the repeat unit, Å.
POSITION_TOLERANCE_A = 1e-3


@dataclass(frozen=True)
class Outcome:
    """A probe's result: its record (``None``: it did not run), the planner the run plans with (a copy with the
    measured scales when it ran), the report's note when it did not, and a warning when a child failed."""

    record: MemoryProbeRecord | None
    planner: Any
    note: str | None = None
    warning: str | None = None

    @property
    def scales(self) -> tuple[tuple[str, float], ...]:
        return tuple(getattr(self.planner, "probe_scales", ()))


def stages_to_probe(ladder: MemoryLadder | None, setting: str) -> tuple[LadderStage, ...]:
    """The stages to measure for rule 12's memory choice ``ladder`` under ``EngineConfig.memory_probe``'s resolved
    ``setting`` ("off", "near-limit", "always"): the chosen stage first, then the next-cheaper one where it might
    fit (always with "always"). Nothing without a ladder (off CUDA) or free memory."""
    if setting == "off" or ladder is None or not ladder.free_bytes:
        return ()
    cheaper = ladder.cheaper
    if setting == "always":
        return (ladder.chosen,) + (() if cheaper is None else (cheaper,))
    cheaper_might_fit = cheaper is not None and cheaper.estimate / PROBE_SPREAD <= ladder.limit
    near = ladder.chosen.estimate * PROBE_SPREAD >= ladder.limit
    if not (near or cheaper_might_fit):
        return ()
    return (ladder.chosen,) + ((cheaper,) if cheaper_might_fit else ())


def repeat_unit(atoms: Any, tolerance_A: float = POSITION_TOLERANCE_A) -> tuple[Any, tuple[int, int, int]] | None:
    """``(unit, repeats)``: the smallest cell whose ``repeats`` along the three cell vectors make ``atoms`` (the
    structure's translational period along each vector, from the positions), or ``None`` when ``atoms`` is not
    periodic in all three directions."""
    import numpy as np
    from scipy.spatial import cKDTree

    if not all(atoms.pbc) or len(atoms) == 0:
        return None
    def torus(values):  # fractional coordinates in [0, 1), as cKDTree's periodic box needs them
        values = values % 1.0
        values[values >= 1.0] = 0.0
        return values

    frac = torus(atoms.get_scaled_positions(wrap=True))
    numbers = np.asarray(atoms.numbers)
    lengths = atoms.cell.lengths()
    tolerance = tolerance_A / lengths  # per axis, fractional
    tree = cKDTree(frac, boxsize=1.0)

    def coincides(shift):
        moved = torus(frac + shift)
        _, index = tree.query(moved)
        close = np.all(np.abs(((moved - frac[index]) + 0.5) % 1.0 - 0.5) <= tolerance + 1e-12, axis=1)
        return bool(np.all(close) and np.all(numbers[index] == numbers))

    repeats = []
    for axis in range(3):
        others = [other for other in range(3) if other != axis]
        offset = ((frac - frac[0]) + 0.5) % 1.0 - 0.5
        same = (numbers == numbers[0]) & np.all(np.abs(offset[:, others]) <= tolerance[others], axis=1)
        steps = frac[same, axis] - frac[0, axis]
        steps = steps % 1.0
        steps = steps[steps > tolerance[axis]]
        found = 1
        for step in np.sort(steps):
            count = int(round(1.0 / step))
            if count > 1 and abs(count * step - 1.0) <= count * tolerance[axis] and len(atoms) % count == 0:
                shift = np.zeros(3)
                shift[axis] = 1.0 / count
                if coincides(shift):
                    found = count
                    break
        repeats.append(found)
    repeats = tuple(repeats)
    unit = atoms[np.flatnonzero(_in_first_cell(frac, repeats, tolerance))]
    unit.set_cell(atoms.cell.array / np.asarray(repeats)[:, None], scale_atoms=False)
    if len(unit) * math.prod(repeats) != len(atoms):
        return None
    return unit, repeats


def _in_first_cell(frac, repeats, tolerance):
    """The atoms whose positions lie in the first of the ``repeats`` sub-cells (one image of each)."""
    import numpy as np

    counts = np.asarray(repeats)
    # an atom at a sub-cell's far face (within the tolerance) belongs to the next sub-cell, periodically
    index = np.floor(frac * counts + tolerance * counts).astype(int) % counts
    return np.all(index == 0, axis=1)


def replica(atoms: Any, size: tuple[int, int] = REPLICA_ATOMS,
            tolerance_A: float = POSITION_TOLERANCE_A) -> Any | None:
    """A replica of ``atoms`` of ``size`` atoms (``REPLICA_ATOMS``): the structure's repeat unit (``repeat_unit``)
    repeated along each cell vector no more often than the structure itself, the most isotropic of those in the
    range. ``atoms`` itself (a copy) when it is no larger than the range; ``None`` when it has no periodic repeat
    unit or no repeat lands in the range."""
    import itertools

    low, high = size
    if len(atoms) <= high:
        return atoms.copy()
    found = repeat_unit(atoms, tolerance_A)
    if found is None:
        return None
    unit, repeats = found
    lengths = unit.cell.lengths()
    best = None
    for counts in itertools.product(*(range(1, n + 1) for n in repeats)):
        n = len(unit) * math.prod(counts)
        if not low <= n <= high:
            continue
        sides = [lengths[axis] * counts[axis] for axis in range(3)]
        key = (max(sides) / min(sides), -n)
        if best is None or key < best[0]:
            best = (key, counts)
    if best is None:
        return None
    out = unit.repeat(best[1])
    out.set_pbc(atoms.pbc)
    return out


def with_scales(planner: Any, scales) -> Any:
    """A copy of ``planner`` whose rule 12 corrects the estimates of the stages in ``scales`` (``(stage, scale)``),
    keeping its process record (the run's planner claims as usual)."""
    copied = copy.copy(planner)
    copied.probe_scales = tuple((str(stage), float(scale)) for stage, scale in scales)
    return copied


def probe(planner: Any, ladder: MemoryLadder, stages, launch: Callable[[LadderStage], dict], *,
          plan_with: Callable[[Any], Plan], clock: Callable[[], float] = time.perf_counter) -> Outcome:
    """Measure ``stages`` of rule 12's choice ``ladder`` with ``launch(stage)`` (one fresh child: ``replica_atoms``,
    ``estimate``, ``reserved``, ``allocated``, ``seconds``) and re-plan with ``plan_with`` on a planner with the
    measured scales. A launch that raises ends the probing: the rules' plan stays, the warning says what failed."""
    from fastmace.planner import memory
    from fastmace.planner.plan import MemoryProbeRecord, MemoryProbeStage

    started = clock()
    measured = []
    for stage in stages:
        try:
            result = launch(stage)
        except Exception as error:  # an optional measurement: the rules' plan stays (the child's log is kept)
            from fastmace.planner.fallback import error_text

            return Outcome(None, planner, warning=f"memory probe stopped: the {stage.stage or 'default'} stage's "
                                                  f"child failed ({error_text(error)}); the rules' plan stays")
        variable = result["estimate"] - stage.fixed
        if variable <= 0 or result["reserved"] <= stage.fixed:
            return Outcome(None, planner, note=f"not used: the {stage.stage or 'default'} stage's replica measured "
                                               f"{result['reserved'] / 2**30:.2f} GiB, below its fixed part")
        scale = (result["reserved"] - stage.fixed) / variable
        corrected = stage.fixed + (stage.estimate - stage.fixed) * scale * memory.PROBE_HEADROOM
        measured.append(MemoryProbeStage(
            stage=stage.stage, rewrites=stage.rewrites, replica_atoms=int(result["replica_atoms"]),
            replica_estimate=int(result["estimate"]), replica_reserved=int(result["reserved"]),
            replica_allocated=int(result["allocated"]), scale=float(scale), estimate=int(stage.estimate),
            corrected=int(math.ceil(corrected)), seconds=float(result["seconds"])))
    scaled = with_scales(planner, [(stage.stage, stage.scale) for stage in measured])
    plan = plan_with(scaled)
    ladder_after = plan.memory_ladder
    chosen = ladder_after.chosen.stage if ladder_after is not None else ladder.chosen.stage
    record = MemoryProbeRecord(stages=tuple(measured), before=ladder.chosen.stage, chosen=chosen,
                               changed_plan=chosen != ladder.chosen.stage, headroom=memory.PROBE_HEADROOM,
                               setup_s=float(clock() - started))
    return Outcome(record, scaled)


def apply(plan: Plan, outcome: Outcome | None) -> Plan:
    """``plan`` with the probe's record (unchanged without one)."""
    if outcome is None or outcome.record is None:
        return plan
    return dataclasses.replace(plan, memory_probe=outcome.record)


def report_line(record: MemoryProbeRecord | None, note: str | None = None) -> str | None:
    """The report's ``memory:`` line of the probe (measured numbers only), or ``None`` without one."""
    if record is None:
        return None if note is None else f"memory probe {note}"

    def name(stage):
        return stage or "default"

    parts = [f"{name(s.stage)}: replica {s.replica_atoms} atoms measured {s.replica_reserved / 2**30:.2f} GiB "
             f"against {s.replica_estimate / 2**30:.2f} estimated, so {s.corrected / 2**30:.2f} GiB here "
             f"(estimate {s.estimate / 2**30:.2f})" for s in record.stages]
    verdict = (f"memory probe chose the {name(record.chosen)} stage, not {name(record.before)}" if record.changed_plan
               else f"memory probe kept the {name(record.chosen)} stage")
    return f"{verdict}; " + "; ".join(parts) + f" (cost {record.setup_s:.1f} s, in setup)"


class ChildProbe:
    """``launch(stage)`` for ``probe``: the stage's rescue records set on ``simulation``'s force field, its run on
    ``replica_atoms`` in a fresh child of the subprocess runner (``SubprocessRunner.memory_probe``). ``close()``
    removes the hand-off directory."""

    def __init__(self, simulation: Any, replica_atoms: Any, steps: int = PROBE_STEPS):
        from fastmace.engine.subprocess import SubprocessRunner

        self.simulation, self.replica_atoms, self.steps = simulation, replica_atoms, steps
        self.runner = SubprocessRunner(dataclasses.replace(simulation.options.runner, persistent=False))

    def __call__(self, stage: LadderStage) -> dict:
        from fastmace.config import AdaptationConfig, CalibrationConfig

        engine = self.simulation._planner.engine  # the planner's (the caller's options may hold none)
        options = dataclasses.replace(self.simulation.options, engine=dataclasses.replace(engine, memory_probe="off"),
                                      adaptation=AdaptationConfig(adapt="off"), calibration=CalibrationConfig())
        started = time.perf_counter()
        result = self.runner.memory_probe(self.simulation, options, self.replica_atoms, dict(stage.rewrites),
                                          steps=self.steps)
        return {**result, "replica_atoms": len(self.replica_atoms), "seconds": time.perf_counter() - started}

    def close(self) -> None:
        self.runner.close()


__all__ = ("ChildProbe", "Outcome", "POSITION_TOLERANCE_A", "PROBE_SPREAD", "PROBE_STEPS", "REPLICA_ATOMS", "apply",
           "probe", "repeat_unit", "replica", "report_line", "stages_to_probe", "with_scales")
