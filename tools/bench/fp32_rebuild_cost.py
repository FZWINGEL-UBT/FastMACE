"""What a neighbour-list rebuild costs the step, from step_breakdown's ``--pass nsys-steps`` export (fp32 profile).

    python tools/bench/fp32_rebuild_cost.py RUN_DIR/NaCl RUN_DIR/NbCl5 ...

Per NVTX range of the host thread (``stage B``, ``neighbour-list rebuild``, ``md step``), the CUDA runtime calls
inside it give, by correlation id, the kernels it launched (graph replays included). For every force-graph replay
it prints the GPU span of its kernels, split by whether a rebuild's kernels ran during it, and the span from one
step's first kernel to the next step's first kernel, split by whether the step launched a rebuild. The
difference of the medians is the rebuild's cost to the step (side-stream contention included). Standard library
only.
"""

from __future__ import annotations

import bisect
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def ranges(db, names):
    """{name: [(start, end, tid)]} of NVTX push/pop ranges."""
    out = defaultdict(list)
    for start, end, text, tid in db.execute(
            "select start, end, coalesce(text, (select value from StringIds where id = textId)), globalTid "
            "from NVTX_EVENTS where end is not null"):
        if text in names:
            out[text].append((start, end, tid))
    return out


def analyse(directory):
    db = sqlite3.connect(Path(directory) / "nsys_steps.sqlite")
    runtime = db.execute("select start, correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME").fetchall()
    by_tid = defaultdict(list)
    for start, corr, tid in runtime:
        by_tid[tid].append((start, corr))
    for calls in by_tid.values():
        calls.sort()
    kernels = defaultdict(list)
    for start, end, corr in db.execute("select start, end, correlationId from CUPTI_ACTIVITY_KIND_KERNEL"):
        kernels[corr].append((start, end))

    def launched(rng):
        start, end, tid = rng
        calls = by_tid.get(tid, [])
        i = bisect.bisect_left(calls, (start, -1))
        spans = []
        while i < len(calls) and calls[i][0] <= end:
            spans += kernels.get(calls[i][1], [])
            i += 1
        return spans

    named = ranges(db, {"stage B", "stage ACP", "neighbour-list rebuild", "md step"})
    rebuild_kernels = sorted(k for r in named["neighbour-list rebuild"] for k in launched(r))
    starts = [k[0] for k in rebuild_kernels]

    def overlaps(lo, hi):
        i = bisect.bisect_left(starts, lo)
        j = i - 1
        if j >= 0 and rebuild_kernels[j][1] > lo:
            return True
        return i < len(starts) and starts[i] < hi

    with_rb, without = [], []
    for rng in named["stage B"]:
        spans = launched(rng)
        if not spans:
            continue
        lo, hi = min(s for s, _ in spans), max(e for _, e in spans)
        (with_rb if overlaps(lo, hi) else without).append((hi - lo) / 1e6)
    # Step intervals: from one integrator stage (ACP) to the next, by their first kernels. In pipelined mode a
    # rebuild is launched before the step's ACP, beside the previous force graph, so an interval holds a rebuild
    # when a rebuild range lies between the two ACP ranges on the host.
    acps = sorted(r for r in named["stage ACP"])
    rebuild_hosts = sorted(r[0] for r in named["neighbour-list rebuild"])
    firsts = []
    for rng in acps:
        spans = launched(rng)
        if spans:
            firsts.append((rng[0], min(s for s, _ in spans)))
    step_with, step_without = [], []
    for (host0, gpu0), (host1, gpu1) in zip(firsts, firsts[1:]):
        i = bisect.bisect_left(rebuild_hosts, host0)
        has = i < len(rebuild_hosts) and rebuild_hosts[i] < host1
        (step_with if has else step_without).append((gpu1 - gpu0) / 1e6)
    med = statistics.median
    print(f"== {directory}")
    print(f"   B replays: {len(without)} without rebuild kernels, median {med(without):.3f} ms; "
          f"{len(with_rb)} with, median {med(with_rb) if with_rb else float('nan'):.3f} ms")
    print(f"   steps (ACP first kernel to the next ACP's): {len(step_without)} without a rebuild, median "
          f"{med(step_without):.3f} ms; {len(step_with)} with, median "
          f"{med(step_with) if step_with else float('nan'):.3f} ms")
    if with_rb and step_with:
        extra = med(step_with) - med(step_without)
        share = len(step_with) / (len(step_with) + len(step_without))
        print(f"   cost per rebuild {extra:.3f} ms; per step {extra * share:.4f} ms "
              f"({100 * extra * share / med(step_without):.2f} % of a step), rebuild share {share:.3f}")


if __name__ == "__main__":
    for directory in sys.argv[1:]:
        analyse(directory)
