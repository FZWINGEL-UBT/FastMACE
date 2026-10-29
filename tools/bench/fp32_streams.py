"""Stream concurrency and GPU idle holes of whole MD steps (fp32 profile, 28 October 2026).

    python tools/bench/fp32_streams.py RUN_DIR/NaCl RUN_DIR/NbCl5 ... [--steps 300]

Reads step_breakdown's ``--pass nsys-steps`` export (``nsys_steps.sqlite``, from ``nsys stats``) and prints, per
step: the span, the time any stream is busy, each stream's busy time and its largest kernels, the overlap of the
two busiest streams, and the idle holes of at least 10 µs (no kernel or memset on any stream) with the kernels on
either side. Standard library only.
"""

from __future__ import annotations

import argparse
import collections
import sqlite3
from pathlib import Path


def union(intervals):
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def length(u):
    return sum(e - s for s, e in u)


def overlap(a, b):
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            total += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyse(directory, steps):
    db = sqlite3.connect(Path(directory) / "nsys_steps.sqlite")
    names = dict(db.execute("select id, value from StringIds"))
    kernels = [(a, b, st, names.get(n, "?")) for a, b, st, n in
               db.execute("select start, end, streamId, shortName from CUPTI_ACTIVITY_KIND_KERNEL")]
    memsets = [(a, b, st, "memset") for a, b, st in db.execute("select start, end, streamId from CUPTI_ACTIVITY_KIND_MEMSET")]
    per_stream = collections.defaultdict(list)
    for a, b, st, n in kernels:
        per_stream[st].append((a, b, n))
    order = sorted(per_stream, key=lambda st: -sum(b - a for a, b, _ in per_stream[st]))
    busy = {st: union([(a, b) for a, b, _ in per_stream[st]]) for st in order}
    span = max(k[1] for k in kernels) - min(k[0] for k in kernels)
    ms = 1e6 * steps
    print(f"== {directory}: per step span {span / ms:.3f} ms, any stream busy "
          f"{length(union([(a, b) for a, b, _, _ in kernels])) / ms:.3f} ms")
    for st in order[:3]:
        top = collections.Counter()
        for a, b, n in per_stream[st]:
            top[n] += b - a
        print(f"   stream {st}: busy {length(busy[st]) / ms:.3f} ms; " +
              ", ".join(f"{n[:40]} {t / ms:.3f}" for n, t in top.most_common(6)))
    if len(order) > 1:
        print(f"   overlap of the two busiest streams {overlap(busy[order[0]], busy[order[1]]) / ms:.3f} ms")
    events = sorted(kernels + memsets)
    holes, hole_ns = collections.Counter(), collections.Counter()
    end, previous = events[0][1], events[0]
    for event in events[1:]:
        if event[0] > end and event[0] - end >= 10_000:
            key = (previous[3][:40], event[3][:40])
            holes[key] += 1
            hole_ns[key] += event[0] - end
        if event[1] > end:
            end, previous = event[1], event
    print(f"   idle holes >= 10 us: {sum(hole_ns.values()) / 1e3 / steps:.1f} us per step")
    for key, ns in hole_ns.most_common(6):
        print(f"     {ns / 1e3 / steps:7.1f} us/step  {holes[key] / steps:.2f}/step  {key[0]} -> {key[1]}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directories", nargs="+")
    p.add_argument("--steps", type=int, default=300)
    args = p.parse_args(argv)
    for directory in args.directories:
        analyse(directory, args.steps)


if __name__ == "__main__":
    main()
