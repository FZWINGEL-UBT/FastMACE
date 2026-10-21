"""OPT-T09 analysis of tools/step_breakdown.py output directories (main, ncu and nsys passes).

Usage: python3 t09_analyse.py ROOT STRUCTURE... > tables.md
Reads per structure: summary.json, stages.csv, categories.csv, kernel_sequence.csv, ncu_kernels.csv.
Prints markdown tables: totals, stages, categories with forward/backward split, top kernels with roofline data.
"""
import csv
import json
import sys
from collections import defaultdict

ROOT = sys.argv[1]
STRUCTS = sys.argv[2:]
DRAM_PEAK_GBS = 448.0  # RTX 5060 Ti: 128-bit GDDR7 at 28 Gbps

GROUP = [  # (report group, predicate on the step_breakdown category)
    ("Neighbour list and edge prep (edge geometry, radial basis, gathers, scatters)", lambda c: c.startswith("Edge elementwise")),
    ("Radial networks and embeddings", lambda c: c.startswith("Radial networks")),
    ("Convolution (interaction 1 reassociated sums, interaction 2 streaming OEq)", lambda c: c.startswith("Interaction")),
    ("Symmetric contraction (cuEq)", lambda c: "cuEq" in c),
    ("Atom-level linears (cuBLAS GEMMs and FastMACE Triton GEMMs)", lambda c: c.startswith("Atom-level linear") or c == "Other"),
    ("Atom-level elementwise (gates, products, residuals, readouts, species tables)", lambda c: c.startswith("Atom-level elementwise")),
    ("Copies and fills", lambda c: c.startswith("Copies")),
]


def group_of(category):
    for name, pred in GROUP:
        if pred(category):
            return name
    return "Unassigned: " + category


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def boundary(seq):
    for i, r in enumerate(seq):
        if "ones_like" in r["kernel"]:
            return i
    return None


for s in STRUCTS:
    d = f"{ROOT}/{s}"
    summ = json.load(open(f"{d}/summary.json"))
    print(f"\n## {s} ({summ['natoms']} atoms, {summ.get('padded_edges')} padded edges)\n")
    rp = summ["force_graph_replay_profile"]
    ns = summ.get("nsys_force_graph_gaps") or {}
    print("| Quantity | Value | Source |\n| --- | ---: | --- |")
    print(f"| Wall time per MD step | {summ['wall_ms_per_step']:.3f} ms | un-profiled window (1000 steps) |")
    print(f"| GPU span per step (stage events) | {summ['gpu_span_ms_per_step']:.3f} ms | CUDA events around stages |")
    print(f"| GPU idle between stages | {summ['gpu_idle_ms_per_step']:.3f} ms | CUDA events |")
    print(f"| B graph replay (forces + 2nd half kick) | {summ['force_graph_ms_per_replay']:.3f} ms | CUDA events, un-profiled |")
    print(f"| Force-graph replay span / kernel time | {rp['span_ms_per_replay']:.3f} / {rp['kernel_ms_per_replay']:.3f} ms | profiler, 10 replays |")
    print(f"| Kernels per replay | {rp['kernels_per_replay']:.0f} | profiler |")
    g = rp["gaps_inside_replays"]
    print(f"| Gaps inside a replay | {g['total_gap_us_per_replay']:.0f} µs ({g['gaps_per_replay']:.0f} gaps) | profiler |")
    if ns:
        print(f"| Gaps between graph nodes (nsys) | {ns.get('total_gap_us_per_replay', float('nan')):.0f} µs ({ns.get('gaps_per_replay', 0):.0f} gaps) | nsys, every node traced |")
    print(f"| Eager force call kernel sum | {summ['eager_force_kernel_ms_per_call']:.3f} ms ({summ['eager_kernels_per_call']:.0f} kernels) | profiler, eager calls |")

    print("\nStages (ms per step):\n\n| Stage | ms | % of GPU span |\n| --- | ---: | ---: |")
    span = summ["gpu_span_ms_per_step"]
    for r in rows(f"{d}/stages.csv"):
        print(f"| {r['stage']} | {float(r['ms_per_step']):.3f} | {100 * float(r['ms_per_step']) / span:.1f} |")

    # Force call by group (profiler, eager), with the forward/backward split from ncu's per-kernel durations.
    cats = rows(f"{d}/categories.csv")
    total = sum(float(r["ms_per_force_call"]) for r in cats)
    seq = rows(f"{d}/kernel_sequence.csv")
    b = boundary(seq)
    import os
    has_ncu = os.path.exists(f"{d}/ncu_kernels.csv")
    ncu = ({int(r["index"]): r for r in rows(f"{d}/ncu_kernels.csv") if r["index"]}
           if b is not None and has_ncu else {})
    fwd, bwd = defaultdict(float), defaultdict(float)
    for i, r in enumerate(seq):
        k = ncu.get(i)
        if not k or not k["duration_us"]:
            continue
        (fwd if i < b else bwd)[group_of(r["category"])] += float(k["duration_us"])
    agg = defaultdict(float)
    kern = defaultdict(float)
    for r in cats:
        agg[group_of(r["category"])] += float(r["ms_per_force_call"])
        kern[group_of(r["category"])] += float(r["kernels_per_call"])
    print(f"\nForce call by part (profiler, eager calls: {total:.3f} ms of kernels; forward/backward split from ncu's "
          f"per-kernel durations, boundary at kernel {b} of {len(seq)}):\n")
    print("| Part | ms | % of force kernels | kernels | forward / backward |\n| --- | ---: | ---: | ---: | --- |")
    for name, ms in sorted(agg.items(), key=lambda kv: -kv[1]):
        f_, b_ = fwd.get(name, 0.0), bwd.get(name, 0.0)
        split = f"{100 * f_ / (f_ + b_):.0f} / {100 * b_ / (f_ + b_):.0f} %" if f_ + b_ else "–"
        print(f"| {name} | {ms:.3f} | {100 * ms / total:.1f} | {kern[name]:.0f} | {split} |")
    ftot, btot = sum(fwd.values()), sum(bwd.values())
    if ftot + btot:
        print(f"| **All (ncu durations)** | | | | **{100 * ftot / (ftot + btot):.0f} / {100 * btot / (ftot + btot):.0f} %** "
              f"({ftot / 1e3:.2f} / {btot / 1e3:.2f} ms) |")

    if not has_ncu:
        continue
    # Top kernels by summed ncu duration, grouped by (kernel, op, input dims).
    peak = summ.get("fp32_peak_tflops")
    top = defaultdict(lambda: dict(us=0.0, n=0, dram=0.0, flop=0.0, sm=[], mem=[], occ=[], th=[], reg=None, grid=[]))
    for r in rows(f"{d}/ncu_kernels.csv"):
        if not r["duration_us"]:
            continue
        key = (r["kernel"][:70], r["op"][:40], r["input_dims"][:60], r["category"])
        t = top[key]
        t["us"] += float(r["duration_us"]); t["n"] += 1
        t["dram"] += float(r["dram_read_bytes"] or 0) + float(r["dram_write_bytes"] or 0)
        t["flop"] += float(r["fp32_flops"] or 0)
        t["sm"].append(float(r["sm_throughput_pct"] or 0)); t["mem"].append(float(r["memory_throughput_pct"] or 0))
        t["occ"].append(float(r["achieved_occupancy_pct"] or 0)); t["th"].append(float(r["theoretical_occupancy_pct"] or 0))
        t["reg"] = r["registers"]; t["grid"].append(r["grid_size"])
    allus = sum(t["us"] for t in top.values())
    print(f"\nTop 15 kernels by ncu duration (one eager force call, {allus / 1e3:.2f} ms of ncu kernel time; DRAM peak "
          f"{DRAM_PEAK_GBS:.0f} GB/s, FP32 peak {peak:.1f} TFLOPS):\n")
    print("| # | Kernel (op; inputs) | Part | Launches | µs | % | DRAM GB/s (% peak) | TFLOPS (% peak) | SM / memory SOL % | Occupancy achieved / theoretical % | Bound |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |")
    for i, (key, t) in enumerate(sorted(top.items(), key=lambda kv: -kv[1]["us"])[:15], 1):
        sec = t["us"] * 1e-6
        bw = t["dram"] / sec / 1e9
        tf = t["flop"] / sec / 1e12
        sm, mem = max(t["sm"]), max(t["mem"])
        smm, memm = sum(t["sm"]) / len(t["sm"]), sum(t["mem"]) / len(t["mem"])
        if max(smm, memm) < 60:
            bound = "latency / occupancy"
        elif memm > smm:
            bound = "memory"
        else:
            bound = "compute"
        occ, th = sum(t["occ"]) / len(t["occ"]), sum(t["th"]) / len(t["th"])
        kname, op, dims, cat = key
        label = kname if op in ("", kname[:40]) else f"{kname} ({op})"
        print(f"| {i} | `{label}`; {dims} | {group_of(cat).split(' (')[0]} | {t['n']} | {t['us']:.0f} | {100 * t['us'] / allus:.1f} | "
              f"{bw:.0f} ({100 * bw / DRAM_PEAK_GBS:.0f}) | {tf:.2f} ({100 * tf / peak:.0f}) | {smm:.0f} / {memm:.0f} | {occ:.0f} / {th:.0f} | {bound} |")
