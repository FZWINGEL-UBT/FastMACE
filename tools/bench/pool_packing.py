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


def simulate_best_fit(blocks, outside=frozenset()):
    """The caching allocator's large-block policy on one expandable segment, for the blocks' lifetimes: the
    smallest free block that fits (lowest address on ties), split; a freed block merges with free neighbours; with
    no fit, the segment extends at its end (from a free block there). Blocks whose site is in ``outside`` live in a
    separate arena (counted at their concurrent peak). Returns (segment footprint, arena peak)."""
    starts = sorted((b[0], i) for i, b in enumerate(blocks))
    ends = sorted((b[1], i) for i, b in enumerate(blocks))
    timeline = sorted([(t, 1, i) for t, i in starts] + [(t, 0, i) for t, i in ends])  # frees before allocs
    free = []  # sorted [offset, size]
    placed, top, arena, arena_peak = {}, 0, 0, 0
    for _, is_alloc, i in timeline:
        size, site = blocks[i][2], blocks[i][3]
        if site in outside:
            arena += size if is_alloc else -size
            arena_peak = max(arena_peak, arena)
            continue
        if is_alloc:
            fits = [(s, o) for o, s in free if s >= size]
            if fits:
                s, o = min(fits)
                free.remove([o, s])
                if s > size:
                    free.append([o + size, s - size])
                    free.sort()
                placed[i] = o
            else:
                if free and free[-1][0] + free[-1][1] == top:
                    o = free.pop()[0]
                else:
                    o = top
                top = o + size
                placed[i] = o
        elif i in placed:
            o = placed.pop(i)
            free.append([o, size])
            free.sort()
            merged = []
            for block in free:
                if merged and merged[-1][0] + merged[-1][1] == block[0]:
                    merged[-1][1] += block[1]
                else:
                    merged.append(block)
            free = merged
    return top, arena_peak


LARGE_BUFFER = 20 * 2 ** 20  # the caching allocator's kLargeBuffer


def simulate_segments(blocks, max_split=None):
    """The caching allocator without expandable segments, for large blocks: a request takes the smallest free block
    that fits (split if it is below ``max_split``; an oversize block only for a request within ``LARGE_BUFFER`` of
    its size), else a new segment of its own size. Freed blocks merge within their segment. Returns the segments'
    total size."""
    timeline = sorted([(b[0], 1, i) for i, b in enumerate(blocks)] + [(b[1], 0, i) for i, b in enumerate(blocks)])
    free, placed, total = [], {}, 0  # free: [size, segment, offset]
    segment_count = 0
    for _, is_alloc, i in timeline:
        size = blocks[i][2]
        if is_alloc:
            def usable(block):
                s = block[0]
                if s < size:
                    return False
                if max_split is not None and s >= max_split:
                    return size >= max_split and s - size < LARGE_BUFFER
                return True
            fits = sorted(b for b in free if usable(b))
            if fits:
                s, seg, off = fits[0]
                free.remove([s, seg, off])
                if s > size and (max_split is None or s < max_split):
                    free.append([s - size, seg, off + size])
                    placed[i] = (seg, off, size)
                else:
                    placed[i] = (seg, off, s)
            else:
                placed[i] = (segment_count, 0, size)
                segment_count += 1
                total += size
        elif i in placed:
            seg, off, s = placed.pop(i)
            free.append([s, seg, off])
            same = sorted((b for b in free if b[1] == seg), key=lambda b: b[2])
            free = [b for b in free if b[1] != seg]
            for b in same:
                if free and free[-1][1] == seg and free[-1][2] + free[-1][0] == b[2]:
                    free[-1][0] += b[0]
                else:
                    free.append(list(b))
    return total


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
    simulated, _ = simulate_best_fit(blocks)
    print(f"best-fit simulation of the allocator: {simulated / GIB:.3f} GiB (validates the model of the allocator)")
    for label, max_split in (("no expandable segments", None), ("no expandable segments, max_split 256 MiB", 2 ** 28)):
        print(f"  {label}: {simulate_segments(blocks, max_split) / GIB:.3f} GiB (simulated)")
    sites = sorted({b[3] for b in blocks if b[2] >= 2 ** 28})
    best_single = min((sum(simulate_best_fit(blocks, frozenset([site]))), site) for site in sites)
    print(f"  best single site in an arena of its own: {best_single[0] / GIB:.3f} GiB ({best_single[1][-60:]})")
    # A shared workspace for FastMACE's own kernels: their blocks packed offline into one persistent buffer, the
    # rest left to the allocator.
    ours = {s for s in sites if "/fastmace/kernels/" in s}
    groups = {"all FastMACE kernels": ours,
              "segment_gemm": {s for s in ours if "segment_gemm" in s},
              "segment_gemm + fused_forward": {s for s in ours if "segment_gemm" in s or "fused_forward" in s},
              "every block of 256 MiB or more": set(sites)}
    for label, group in groups.items():
        inside = [b for b in blocks if b[3] in group]
        workspace, _ = pack_offline(inside)
        segment, _ = simulate_best_fit([b for b in blocks if b[3] not in group])
        second, _ = simulate_best_fit(inside)
        print(f"  shared workspace for {label}: pool {segment / GIB:.3f} + workspace {workspace / GIB:.3f} = "
              f"{(segment + workspace) / GIB:.3f} GiB (a second pool, best fit: {(segment + second) / GIB:.3f})")
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
