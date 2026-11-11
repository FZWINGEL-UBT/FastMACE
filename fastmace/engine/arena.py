"""Segment-level capture arena: the offset plan and its match check (results/capture-arena-20261110, option A).

The engine's graph pool can sit on a pluggable allocator (``torch.cuda.MemPool``). The caching layer still splits
and reuses blocks inside that pool, so the allocator sees only segment requests, in order, and never their frees.
A recording capture gives the block-level trace of the pool (the allocator's memory history: segment allocations,
block allocations and block frees, with addresses). From it, each segment's busy intervals are known: the event spans
over which one of its blocks is live. Two segments whose busy intervals never overlap may share bytes.

``plan_segments`` places the segments at offsets of one arena (first fit, largest first); ``check_plan`` compares a
planned capture's own trace with the plan. Overlap is only safe when the caching layer made the same choices in the
planned capture as in the recording (its free-block order breaks ties by address, and arena addresses differ from the
recorded ones), so the engine re-records the planned capture and falls back to the caching pool on any difference.

Pure Python on (action, size, addr) tuples: no CUDA here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SEGMENT_ACTIONS = ("segment_alloc", "segment_map")


@dataclass(frozen=True)
class SegmentUse:
    """One segment request of the recorded capture: its size and its busy intervals (merged, half-open, in event
    indices of the trace)."""

    size: int
    busy: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ArenaPlan:
    """Offsets of the segments in request order, the arena's size and what the plan was made from."""

    sizes: tuple[int, ...]
    offsets: tuple[int, ...]
    total: int
    uses: tuple[SegmentUse, ...] = field(repr=False)
    # The block-level signature the plan rests on: each block's (size, segment index, alloc event, free event).
    blocks: tuple[tuple[int, int, int, int], ...] = field(repr=False)

    @property
    def reserved(self) -> int:
        """What the caching pool asks for without the arena: every segment's bytes."""
        return sum(self.sizes)


def _merged(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return tuple((a, b) for a, b in out)


def segment_uses(events):
    """(segments, blocks) of a pool's trace: ``SegmentUse`` per segment in request order, and each block as
    (size, segment index, alloc event, free event); a block still live at the end is busy to ``len(events)``.

    A block outside every recorded segment raises ``ValueError``: the trace is incomplete."""
    segs, blocks = [], []
    live = {}
    busy = []

    def owner(addr):
        for k in range(len(segs) - 1, -1, -1):
            start, size = segs[k]
            if start <= addr < start + size:
                return k
        raise ValueError(f"block at {addr:#x} lies in no recorded segment")

    for i, (action, size, addr) in enumerate(events):
        if action in SEGMENT_ACTIONS:
            segs.append((addr, size))
            busy.append([])
        elif action == "alloc":
            live[addr] = (i, size, owner(addr))
        elif action == "free_completed" and addr in live:
            start, size_, k = live.pop(addr)
            busy[k].append((start, i))
            blocks.append((size_, k, start, i))
    end = len(events)
    for addr, (start, size_, k) in live.items():
        busy[k].append((start, end))
        blocks.append((size_, k, start, end))
    uses = tuple(SegmentUse(size, _merged(b)) for (_, size), b in zip(segs, busy))
    return uses, tuple(sorted(blocks, key=lambda b: (b[2], b[3])))


def _clash(a, b):
    return any(x0 < y1 and y0 < x1 for x0, x1 in a for y0, y1 in b)


def plan_segments(events, alignment: int = 2 << 20) -> ArenaPlan:
    """First-fit offsets, largest segment first, each segment clear of every placed segment whose busy intervals
    meet its own. Offsets are multiples of ``alignment`` (the caching allocator's 2 MiB segment granularity)."""
    uses, blocks = segment_uses(events)
    order = sorted(range(len(uses)), key=lambda k: (-uses[k].size, k))
    offsets = [0] * len(uses)
    placed = []
    total = 0
    for k in order:
        size = uses[k].size
        spans = sorted((offsets[j], offsets[j] + uses[j].size) for j in placed if _clash(uses[k].busy, uses[j].busy))
        offset = 0
        for lo, hi in spans:
            if offset + size <= lo:
                break
            offset = max(offset, -(-hi // alignment) * alignment)
        offsets[k] = offset
        placed.append(k)
        total = max(total, offset + size)
    return ArenaPlan(tuple(u.size for u in uses), tuple(offsets), total, uses, blocks)


def overlaps_are_safe(plan: ArenaPlan) -> bool:
    """No two segments that share bytes are ever busy at the same time (the plan's own invariant)."""
    n = len(plan.sizes)
    for i in range(n):
        for j in range(i + 1, n):
            a0, a1 = plan.offsets[i], plan.offsets[i] + plan.sizes[i]
            b0, b1 = plan.offsets[j], plan.offsets[j] + plan.sizes[j]
            share = a0 < b1 and b0 < a1
            if share and _clash(plan.uses[i].busy, plan.uses[j].busy):
                return False
    return True


@dataclass(frozen=True)
class PlanCheck:
    ok: bool
    reason: str = ""


def check_plan(plan: ArenaPlan, events) -> PlanCheck:
    """Does a planned capture's own trace match the plan? The same segment requests (sizes, order), and every block
    in the same segment over the same event span. Then the busy intervals, and so the overlaps' safety, are the
    plan's. Anything else: not ok, with the first difference."""
    try:
        uses, blocks = segment_uses(events)
    except ValueError as error:
        return PlanCheck(False, str(error))
    sizes = tuple(u.size for u in uses)
    if sizes != plan.sizes:
        first = next((k for k, (a, b) in enumerate(zip(sizes, plan.sizes)) if a != b),
                     min(len(sizes), len(plan.sizes)))
        return PlanCheck(False, f"segment requests differ at {first}: {len(sizes)} segments against {len(plan.sizes)}")
    if blocks != plan.blocks:
        first = next((k for k, (a, b) in enumerate(zip(blocks, plan.blocks)) if a != b),
                     min(len(blocks), len(plan.blocks)))
        return PlanCheck(False, f"block placement differs at block {first} of {len(plan.blocks)}")
    return PlanCheck(True)


__all__ = ["ArenaPlan", "PlanCheck", "SegmentUse", "check_plan", "overlaps_are_safe", "plan_segments", "segment_uses"]
