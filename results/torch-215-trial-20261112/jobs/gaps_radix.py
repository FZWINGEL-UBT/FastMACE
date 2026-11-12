"""Host-side gaps and the radix sort, from an attribution run (trace + pystacks of profile_point.py PYSTACKS=1).

  python gaps_radix.py TRACE.json.gz PYSTACKS.json STEPS OUT.json

- gaps: after the first SKIP_US of the window (the profiler start's first copy and graph launch, listed apart),
  every idle stretch of the GPU over GAP_US (20) between consecutive GPU events. For each gap: the GPU event
  before and after it (name, kind, bytes, copy kind, graph id, launch site) and the host events that run during it
  (CUDA runtime calls and CPU ops overlapping the gap, longest first). Summed per (before, after) pattern per step.
- radix: every radixSort / cub sort kernel: per-step count, graph id, mean time, and its launch stack (own frames).
"""
import collections
import gzip
import json
import re
import sys
from pathlib import Path

trace_path, stacks_path, steps, out_path = sys.argv[1:5]
STEPS = int(steps)
GAP_US = 20.0
SKIP_US = 1000.0  # gaps in the first ms of the window are the profiler start (first memcpy / graph launch under CUPTI)
SKIP_PREFIX = ("torch/", "triton/", "functorch/", "<string>", "contextlib.py", "cuequivariance_ops_torch/")

events = json.loads(gzip.open(trace_path).read())
events = events["traceEvents"] if isinstance(events, dict) else events
gpu = sorted((e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
             key=lambda e: float(e["ts"]))
host = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cpu_op", "cuda_driver")]
stacks = json.loads(Path(stacks_path).read_text())
runtime_by_corr = {e["args"].get("correlation"): e for e in host if e.get("cat") in ("cuda_runtime", "cuda_driver")
                   and e.get("args", {}).get("correlation") is not None}


def own_frames(e, n=4):
    a = e.get("args", {})
    if a.get("graph node id") is None or a.get("graph id") is None:
        r = runtime_by_corr.get(a.get("correlation"))
        return ["(eager)" + (f" via {r['name']}" if r else "")]
    s = stacks.get(str((int(a["graph id"]) << 32) | int(a["graph node id"]))) or stacks.get(str(a["graph node id"]))
    if s is None:
        return ["(graph node without stack)"]
    s = s if isinstance(s, str) else "\n".join(map(str, s))
    fr = [re.sub(r".*/(site-packages|src-[0-9a-f]+|src)/", "", f.strip()) for f in s.splitlines() if f.strip()]
    own = [f for f in fr if not f.startswith(SKIP_PREFIX)]
    return own[:n] or fr[:n]


def describe(e):
    a = e.get("args", {})
    return {"name": e["name"][:90], "cat": e["cat"], "dur_us": round(float(e["dur"]), 2), "bytes": a.get("bytes"),
            "copy_kind": a.get("copy kind"), "graph_id": a.get("graph id"), "site": own_frames(e, 2)}


gaps = []
t0 = float(gpu[0]["ts"])
start_artefacts = []
for a, b in zip(gpu, gpu[1:]):
    end = float(a["ts"]) + float(a["dur"])
    gap = float(b["ts"]) - end
    if gap < GAP_US:
        continue
    if float(a["ts"]) - t0 < SKIP_US:
        start_artefacts.append({"gap_us": round(gap, 1), "before": describe(a), "after": describe(b)})
        continue
    during = [h for h in host if float(h["ts"]) < float(b["ts"]) and float(h["ts"]) + float(h["dur"]) > end]
    during.sort(key=lambda h: -min(float(h["ts"]) + float(h["dur"]), float(b["ts"])) + max(float(h["ts"]), end))
    gaps.append({"gap_us": round(gap, 1), "before": describe(a), "after": describe(b),
                 "host_during": [{"name": h["name"][:80], "cat": h["cat"], "dur_us": round(float(h["dur"]), 1)}
                                 for h in during[:6]]})
pattern = collections.defaultdict(lambda: [0.0, 0])
for g in gaps:
    k = (f"{g['before']['name'][:40]} ({g['before']['copy_kind'] or g['before']['cat']}, {g['before']['bytes']} B, "
         f"graph {g['before']['graph_id']}) -> {g['after']['name'][:40]} (graph {g['after']['graph_id']})")
    pattern[k][0] += g["gap_us"] / STEPS
    pattern[k][1] += 1
gap_rows = sorted(({"pattern": k, "us_per_step": round(v[0], 1), "count": v[1]} for k, v in pattern.items()),
                  key=lambda r: -r["us_per_step"])

sorts = [e for e in gpu if re.search(r"radix|RadixSort|DeviceSegmentedSort|DeviceSort", e["name"])]
by = collections.defaultdict(lambda: {"n": 0, "us": 0.0, "graph_ids": collections.Counter()})
for e in sorts:
    key = (e["name"][:80], " <- ".join(own_frames(e, 3)))
    by[key]["n"] += 1
    by[key]["us"] += float(e["dur"])
    by[key]["graph_ids"][str(e["args"].get("graph id"))] += 1
radix = [{"kernel": k[0], "stack": k[1], "per_step": v["n"] / STEPS, "us_per_step": round(v["us"] / STEPS, 1),
          "graph_ids": dict(v["graph_ids"])} for k, v in sorted(by.items(), key=lambda kv: -kv[1]["us"])]

out = {"gap_threshold_us": GAP_US, "gaps_over_threshold_us_per_step": round(sum(g["gap_us"] for g in gaps) / STEPS, 1),
       "window_start_artefacts": start_artefacts, "gap_patterns": gap_rows, "gaps": sorted(gaps, key=lambda g: -g["gap_us"])[:25], "radix": radix}
Path(out_path).write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: out[k] for k in ("gaps_over_threshold_us_per_step",)}), "start artefacts", [g["gap_us"] for g in start_artefacts])
for r in gap_rows[:8]:
    print(f"  gap {r['us_per_step']:7.1f} us/step  x{r['count']}  {r['pattern']}")
for g in out["gaps"][:4]:
    print("  ", g["gap_us"], "host:", [(h["name"][:40], h["dur_us"]) for h in g["host_during"][:3]])
for r in radix:
    print(f"  sort {r['us_per_step']:6.1f} us/step  {r['per_step']}/step  graphs {r['graph_ids']}  {r['stack'][:200]}")
