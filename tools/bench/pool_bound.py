"""Replay a CUDA memory snapshot's trace per pool: live bytes over time, the peak, the tensors live at it, and the
peak with selected allocations (by frame substring) shrunk to a fraction (the chunked-backward bound)."""
import collections
import gzip
import pickle
import sys

path = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else "tf32x3_gemm"
s = pickle.load(gzip.open(path))
t = s["device_traces"][0]
GiB = 2**30


def where(e):
    fr = e.get("frames") or []
    names = [f"{f['filename'].split('/')[-1]}:{f['line']}:{f['name']}" for f in fr[:40]]
    return names


def is_tagged(e):
    return any(tag in f["filename"] or tag in f["name"] for f in (e.get("frames") or []))


pools = collections.Counter(e["pool_id"] for e in t if e["action"] == "alloc")
print("allocs per pool:", dict(pools))
for pool in pools:
    if pool == (0, 0):
        continue
    live, tagged = {}, {}
    cur = curt = 0
    peak = (0, None)
    series = []
    for i, e in enumerate(t):
        if e["pool_id"] != pool:
            continue
        if e["action"] == "alloc":
            live[e["addr"]] = e
            cur += e["size"]
            if is_tagged(e):
                tagged[e["addr"]] = e["size"]
                curt += e["size"]
            series.append((i, cur, curt))
            if cur > peak[0]:
                peak = (cur, i, dict(live))
        elif e["action"] == "free_completed" and e["addr"] in live:
            cur -= live.pop(e["addr"])["size"]
            if e["addr"] in tagged:
                curt -= tagged.pop(e["addr"])
            series.append((i, cur, curt))
    if not series:
        continue
    print(f"\npool {pool}: {len(series)} events, live peak {peak[0] / GiB:.2f} GiB at event {peak[1]}")
    big = sorted(peak[2].values(), key=lambda e: -e["size"])[:14]
    for e in big:
        fr = where(e)
        keep = [f for f in fr if not f.startswith(("<", "__init__", "graphs.py", "module.py"))]
        print(f"  {e['size'] / GiB:6.2f} GiB  tagged={is_tagged(e)!s:5}  {' < '.join(keep[:4])}")
    for k in (1, 4, 8, 16):
        best = max(c - ct + ct / k for _, c, ct in series)
        print(f"  peak with the tagged allocations at 1/{k}: {best / GiB:.2f} GiB")


# The new peak with the tagged ones at 1/8 and the layer region's saved output (modules.py:415) dropped.
def dropped(e):
    return any(f["filename"].endswith("modules.py") and f["line"] == 415 for f in (e.get("frames") or []))


for pool in pools:
    if pool == (0, 0):
        continue
    live = {}
    best = (0, None, None)
    for i, e in enumerate(t):
        if e["pool_id"] != pool:
            continue
        if e["action"] == "alloc":
            live[e["addr"]] = e
        elif e["action"] == "free_completed" and e["addr"] in live:
            live.pop(e["addr"])
        else:
            continue
        val = sum(0 if dropped(x) else x["size"] / (8 if is_tagged(x) else 1) for x in live.values())
        if val > best[0]:
            best = (val, i, dict(live))
    print(f"\npool {pool}: tagged at 1/8, layer output dropped: peak {best[0] / GiB:.2f} GiB at event {best[1]}")
    for e in sorted(best[2].values(), key=lambda e: -e["size"])[:12]:
        skip = ("<", "__init__", "graphs.py", "module.py", "custom_ops", "eval_frame", "_compile")
        keep = [f for f in where(e) if not f.startswith(skip)]
        print(f"  {e['size'] / GiB:6.2f} GiB tagged={is_tagged(e)!s:5} dropped={dropped(e)!s:5} {' < '.join(keep[:4])}")


# The capture arena's bound: the pool's allocations packed offline with their lifetimes known (greedy by size, each
# at the lowest offset free over its lifetime; 512-byte alignment as the caching allocator's), against the live peak
# and the pool's mapped peak.
def arena(pool, drop=lambda e: False, order=lambda b: -b[2]):
    start, blocks = {}, []
    for i, e in enumerate(t):
        if e["pool_id"] != pool:
            continue
        if e["action"] == "alloc":
            start[e["addr"]] = (i, e)
        elif e["action"] == "free_completed" and e["addr"] in start:
            i0, a = start.pop(e["addr"])
            blocks.append((i0, i, -(-a["size"] // 512) * 512, a))
    end = max((i for i, _ in start.values()), default=0) + 1
    blocks += [(i0, end + 1, -(-a["size"] // 512) * 512, a) for i0, a in start.values()]
    blocks = [b for b in blocks if not drop(b[3])]
    placed, top = [], 0
    for i0, i1, size, _ in sorted(blocks, key=order):
        busy = sorted((o, o + s) for j0, j1, o, s in placed if j0 < i1 and i0 < j1)
        offset = 0
        for lo, hi in busy:
            if offset + size <= lo:
                break
            offset = max(offset, hi)
        placed.append((i0, i1, offset, size))
        top = max(top, offset + size)
    events = sorted([(i0, s) for i0, _, s, _ in blocks] + [(i1, -s) for _, i1, s, _ in blocks])
    cur = live = 0
    for _, d in events:
        cur += d
        live = max(live, cur)
    return top, live


for pool in pools:
    if pool == (0, 0):
        continue
    orders = {"size": lambda b: -b[2], "lifetime": lambda b: -(b[1] - b[0]),
              "size*lifetime": lambda b: -b[2] * (b[1] - b[0])}
    for label, drop in (("as profiled", lambda e: False), ("layer output dropped", dropped)):
        for name, order in orders.items():
            top, live = arena(pool, drop, order)
            print(f"arena {pool} {label}, by {name}: packed {top / GiB:.2f} GiB, live peak {live / GiB:.2f} GiB, "
                  f"loss {(top - live) / GiB:.2f} GiB")
