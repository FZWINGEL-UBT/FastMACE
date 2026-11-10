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
