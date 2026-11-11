"""The segment-level capture arena's offset plan and plan-match check (fastmace.engine.arena; CPU only)."""

import random

import pytest

from fastmace.engine import arena

MB = 1 << 20


def trace(segments, blocks, base=0x7F0000000000):
    """A pool trace: ``segments`` [(size, request event)], ``blocks`` [(segment, offset, size, alloc, free|None)].
    Events are emitted in event-index order; addresses are base + 1 GiB per segment."""
    addr = {k: base + k * (1 << 30) for k in range(len(segments))}
    timed = [(t, 0, ("segment_alloc", size, addr[k])) for k, (size, t) in enumerate(segments)]
    for k, offset, size, a, f in blocks:
        timed.append((a, 2, ("alloc", size, addr[k] + offset)))
        if f is not None:  # a free sorts before an allocation at the same time (address reuse)
            timed.append((f, 1, ("free_completed", size, addr[k] + offset)))
    return [event for _, _, event in sorted(timed)]


def test_idle_segments_share_bytes_and_busy_ones_do_not():
    events = trace([(10 * MB, 0), (10 * MB, 1)], [(0, 0, 8 * MB, 2, 3), (1, 0, 8 * MB, 4, 5)])
    plan = arena.plan_segments(events)
    assert plan.sizes == (10 * MB, 10 * MB) and plan.offsets == (0, 0) and plan.total == 10 * MB
    assert plan.reserved == 20 * MB and arena.overlaps_are_safe(plan)
    events = trace([(10 * MB, 0), (10 * MB, 1)], [(0, 0, 8 * MB, 2, 5), (1, 0, 8 * MB, 3, 4)])  # nested lifetimes
    plan = arena.plan_segments(events)
    assert plan.offsets == (0, 10 * MB) and plan.total == 20 * MB and arena.overlaps_are_safe(plan)


def test_a_segment_is_busy_over_each_block_and_a_live_block_to_the_end():
    events = trace([(10 * MB, 0)], [(0, 0, 2 * MB, 1, 2), (0, 2 * MB, 2 * MB, 3, None)])
    uses, blocks = arena.segment_uses(events)
    n = len(events)
    assert len(uses) == 1 and uses[0].busy[-1][1] == n
    assert [b[1] for b in blocks] == [0, 0] and blocks[-1][3] == n


def test_offsets_are_aligned_and_largest_first():
    events = trace([(4 * MB, 0), (6 * MB, 1), (2 * MB, 2)],
                   [(0, 0, MB, 3, 9), (1, 0, MB, 4, 9), (2, 0, MB, 5, 9)])  # all busy together
    plan = arena.plan_segments(events)
    assert sorted(plan.offsets) == [0, 6 * MB, 10 * MB] and plan.offsets[1] == 0
    assert all(o % (2 * MB) == 0 for o in plan.offsets) and plan.total == 12 * MB


@pytest.mark.parametrize("seed", range(20))
def test_random_plans_are_safe_and_no_larger_than_the_pool(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 12)
    segments = [(rng.choice([2, 4, 20, 64]) * MB, k) for k in range(n)]
    blocks = []
    t = n
    for _ in range(rng.randint(n, 4 * n)):
        k = rng.randrange(n)
        a = t + rng.randint(0, 3)
        f = a + rng.randint(1, 20)
        blocks.append((k, rng.randrange(segments[k][0] // MB) * MB, MB, a, f))
        t += 1
    blocks = [(k, o, s, a, f) for i, (k, o, s, a, f) in enumerate(blocks)
              if not any(k == k2 and o == o2 and a < f2 and a2 < f for j, (k2, o2, _, a2, f2) in enumerate(blocks)
                         if j < i)]  # one live block per address at a time, as a pool has
    plan = arena.plan_segments(trace(segments, blocks))
    assert arena.overlaps_are_safe(plan)
    assert plan.total <= plan.reserved
    peak = max((sum(u.size for u in plan.uses if any(a <= e < b for a, b in u.busy)) for e in range(t + 30)),
               default=0)
    assert plan.total >= peak  # never below the busy segments' sum


def test_check_plan_accepts_the_same_trace_at_other_addresses():
    segments, blocks = [(10 * MB, 0), (10 * MB, 1)], [(0, 0, 8 * MB, 2, 3), (1, 0, 8 * MB, 4, 5)]
    plan = arena.plan_segments(trace(segments, blocks))
    assert arena.check_plan(plan, trace(segments, blocks, base=0x100000000)).ok


def test_check_plan_refuses_other_choices():
    segments, blocks = [(10 * MB, 0), (10 * MB, 1)], [(0, 0, 8 * MB, 2, 3), (1, 0, 8 * MB, 4, 5)]
    plan = arena.plan_segments(trace(segments, blocks))
    moved = [(0, 0, 8 * MB, 2, 3), (0, 0, 8 * MB, 4, 5)]  # the second block reuses segment 0
    check = arena.check_plan(plan, trace(segments, moved))
    assert not check.ok and "block placement" in check.reason
    longer = [(0, 0, 8 * MB, 2, 3), (1, 0, 8 * MB, 4, 6)]  # a lifetime changed
    assert not arena.check_plan(plan, trace(segments, longer)).ok
    check = arena.check_plan(plan, trace([(10 * MB, 0), (12 * MB, 1)], blocks))
    assert not check.ok and "segment requests" in check.reason
    check = arena.check_plan(plan, trace(segments + [(2 * MB, 6)], blocks))
    assert not check.ok and "3 segments against 2" in check.reason


def test_a_block_outside_every_segment_is_refused():
    events = [("segment_alloc", 10 * MB, 0x1000_0000), ("alloc", MB, 0x9000_0000)]
    with pytest.raises(ValueError, match="no recorded segment"):
        arena.plan_segments(events)
    plan = arena.plan_segments([("segment_alloc", 10 * MB, 0x1000_0000)])
    check = arena.check_plan(plan, events)
    assert not check.ok and "no recorded segment" in check.reason
