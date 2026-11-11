"""The GPU's busy and idle time in an Nsight Systems capture (sqlite export), without the torch profiler (11 November
2026). Busy: the union of kernel, memcpy and memset intervals within the host's ``captured_call`` NVTX range. Each
idle gap of 20 us or more is charged to the innermost NVTX range open on the calling thread at its midpoint, and to
the range of the launch that ended it (its runtime call's thread, else the calling thread). Usage:
nsysan.py CAPTURE.sqlite OUT.json"""
import bisect
import json
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
out = sys.argv[2]
tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def strings():
    if "StringIds" not in tables:
        return {}
    return dict(db.execute("SELECT id, value FROM StringIds"))


names = strings()
ranges = []
for start, end, text, text_id, tid in db.execute(
        "SELECT start, end, text, textId, globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL AND end > start"):
    label = text if text is not None else names.get(text_id, "?")
    ranges.append((start, end, label, tid))
call = [r for r in ranges if r[2] == "captured_call"]
if not call:
    raise SystemExit("no captured_call range")
c_start, c_end, _, main = call[0]
host_wall = (c_end - c_start) / 1e9
by_tid = {}
for r in ranges:
    if r[2] != "captured_call":
        by_tid.setdefault(r[3], []).append(r)


def label_at(tid, ts):
    found = [(e - s, label) for s, e, label, _ in by_tid.get(tid, ()) if s <= ts < e]
    if not found and tid != main:
        found = [(e - s, label) for s, e, label, _ in by_tid.get(main, ()) if s <= ts < e]
    return min(found)[1] if found else "(none)"


device, kinds = [], {}
for table, kind in (("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"), ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
                    ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
    if table not in tables:
        continue
    for s, e, corr in db.execute(f"SELECT start, end, correlationId FROM {table} WHERE end > {c_start} AND "
                                 f"start < {c_end}"):
        s, e = max(s, c_start), min(e, c_end)
        device.append((s, e, corr))
        kinds[kind] = kinds.get(kind, 0) + (e - s)
launches = {}
if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
    for corr, tid, s in db.execute("SELECT correlationId, globalTid, start FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
        launches[corr] = (tid, s)
device.sort()
busy, gaps, cur = 0, [], None
first_start = device[0][0] if device else c_start
if device and device[0][0] > c_start:
    gaps.append((c_start, device[0][0], device[0][2]))
for s, e, corr in device:
    if cur is None or s > cur[1]:
        if cur is not None:
            busy += cur[1] - cur[0]
            gaps.append((cur[1], s, corr))
        cur = [s, e]
    else:
        cur[1] = max(cur[1], e)
if cur is not None:
    busy += cur[1] - cur[0]
    if cur[1] < c_end:
        gaps.append((cur[1], c_end, None))
during, ended = {}, {}
for s, e, corr in gaps:
    length = e - s
    small = length < 20_000
    key = "(gaps under 20 us)" if small else label_at(main, (s + e) // 2)
    g = during.setdefault(key, [0, 0])
    g[0] += 1
    g[1] += length
    launch = launches.get(corr) if corr is not None else None
    key = "(gaps under 20 us)" if small else (label_at(*launch) if launch else "(call end)")
    g = ended.setdefault(key, [0, 0])
    g[0] += 1
    g[1] += length


def table(d, top=30):
    return {k: dict(count=v[0], seconds=round(v[1] / 1e9, 4))
            for k, v in sorted(d.items(), key=lambda kv: -kv[1][1])[:top]}


rec = dict(host_wall_s=host_wall, device_busy_s=busy / 1e9, idle_s=host_wall - busy / 1e9,
           idle_fraction=1 - busy / 1e9 / host_wall, device_by_kind_s={k: v / 1e9 for k, v in kinds.items()},
           device_events=len(device), gaps_during=table(during), gaps_ended_by=table(ended))
json.dump(rec, open(out, "w"), indent=1)
print(json.dumps(dict(host_wall_s=round(host_wall, 4), busy_s=round(busy / 1e9, 4),
                      idle_fraction=round(rec["idle_fraction"], 4), during=dict(list(rec["gaps_during"].items())[:8]))))
