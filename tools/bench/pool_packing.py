"""Where the graph pool's packing slack comes from, and how much a placement change could recover (offline).

Reads a ``memory_ledger.py`` record (its ``recapture.pool.event_list``: every allocator event inside the graph pool's
segments during one capture of the stages, in order) and replays it:

1. **Growth events.** Each allocation that ends above the highest address used so far grows the pool. For each,
   the tool reports the bytes free below that mark at the moment (holes) and the largest single hole. Holes too
   small for the allocation are the slack.
2. **The bound.** The same blocks with the same lifetimes are packed offline, largest first, each at the lowest
   offset that overlaps no block alive at the same time. The whole sequence is known at capture time, so this is
   what a planned placement could reach. It is compared with the live peak (no placement can go below it) and the
   allocator's footprint.

Pure Python, no GPU::

    python tools/bench/pool_packing.py results/<run>/ledger/ledger-NbCl5-float32-51840.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GIB = 2 ** 30


def lifetimes(events):
    """[(start, end, size, site, address)] per allocation, in event-index time; frees by their request."""
    open_blocks, out = {}, []
    for index, (action, address, size, site) in enumerate(events):
        if action == "alloc":
            open_blocks[address] = (index, size, site)
        elif action == "free_requested" and address in open_blocks:
            start, size, site = open_blocks.pop(address)
            out.append((start, index, size, site, address))
    end = len(events)
    out += [(start, end, size, site, address) for address, (start, size, site) in open_blocks.items()]
    return sorted(out)


def replay(events, ranges):
    """The allocator's footprint (the highest address used in each segment, summed) and every allocation that grew
    it, with the holes below the segment's mark then."""
    def segment(address):
        return next(i for i, (lo, hi) in enumerate(ranges) if lo <= address < hi)

    live, tops, growth, peak_live, now = {}, [0] * len(ranges), [], 0, 0
    for action, address, size, site in events:
        if action == "alloc":
            index = segment(address)
            base, top = ranges[index][0], tops[index]
            offset = address - base
            if offset + size > top:
                holes = gaps(sorted((a - base, s) for a, (s, _, i) in live.items() if i == index), top)
                growth.append(dict(site=site, size=size, segment=index, offset=offset, top_before=top,
                                   grew_by=offset + size - top, free_below=sum(holes),
                                   largest_hole=max(holes, default=0), holes=len(holes)))
                tops[index] = offset + size
            live[address] = (size, site, index)
            now += size
            peak_live = max(peak_live, now)
        elif action == "free_requested" and address in live:
            now -= live.pop(address)[0]
    return sum(tops), peak_live, growth


def gaps(blocks, top):
    """Sizes of the free gaps between sorted (offset, size) blocks below ``top``."""
    out, cursor = [], 0
    for offset, size in blocks:
        if offset > cursor:
            out.append(offset - cursor)
        cursor = max(cursor, offset + size)
    if top > cursor:
        out.append(top - cursor)
    return out


def pack_offline(blocks):
    """Greedy offline packing: largest first, lowest offset free over the block's lifetime. Returns the footprint
    and the placement [(offset, start, end, size, site)]."""
    placed = []
    for start, end, size, site, _ in sorted(blocks, key=lambda b: (-b[2], b[0])):
        busy = sorted((o, o + s) for o, a, b, s, _ in placed if a < end and start < b)
        offset = 0
        for lo, hi in busy:
            if offset + size <= lo:
                break
            offset = max(offset, hi)
        placed.append((offset, start, end, size, site))
    return max((o + s for o, _, _, s, _ in placed), default=0), placed


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ledger", type=Path)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--output", type=Path, help="write the analysis as JSON")
    args = p.parse_args(argv)
    record = json.loads(args.ledger.read_text())
    pool = record["recapture"]["pool"]
    events = pool["event_list"]
    events = [e for e in events if e[0] in ("alloc", "free_requested", "free_completed")]
    top, peak_live, growth = replay(events, pool["segment_ranges"])
    blocks = lifetimes(events)
    packed, placement = pack_offline(blocks)
    big = sorted(blocks, key=lambda b: -b[2])[:args.top]
    print(f"{record['structure']} {record['natoms']} atoms: pool reserved {pool['reserved'] / GIB:.3f} GiB; "
          f"replayed footprint {top / GIB:.3f} GiB; live peak {peak_live / GIB:.3f} GiB; "
          f"offline largest-first packing {packed / GIB:.3f} GiB")
    print(f"recoverable in principle: {(top - peak_live) / GIB:.3f} GiB (to the live peak), "
          f"{(top - packed) / GIB:.3f} GiB (by the offline packing)")
    print(f"{len(growth)} growth events; the largest:")
    for g in sorted(growth, key=lambda g: -g["grew_by"])[:args.top]:
        print(f"  +{g['grew_by'] / GIB:6.3f} GiB for {g['size'] / GIB:6.3f} GiB at {g['site'][-70:]}; free below "
              f"{g['free_below'] / GIB:.3f} GiB in {g['holes']} holes, largest {g['largest_hole'] / GIB:.3f} GiB")
    print("largest blocks (lifetime in event index):")
    for start, end, size, site, _ in big:
        print(f"  {size / GIB:6.3f} GiB  [{start:4d}, {end:4d})  {site[-80:]}")
    if args.output:
        args.output.write_text(json.dumps(dict(
            ledger=str(args.ledger), reserved=pool["reserved"], footprint=top, live_peak=peak_live, packed=packed,
            growth=growth, largest=[dict(size=s, start=a, end=b, site=site) for a, b, s, site, _ in big],
            placement=[dict(offset=o, start=a, end=b, size=s, site=site) for o, a, b, s, site in placement]),
            indent=1) + "\n")
    return dict(footprint=top, live_peak=peak_live, packed=packed)


if __name__ == "__main__":
    main()
