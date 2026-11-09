"""What limits each kernel of one force call, and each part of the step (offline, from tools/modal/step_limiters.sh).

Reads a job directory (``main/`` from step_breakdown's main pass, ``ncu_raw.csv`` from the ncu pass), joins ncu's
kernels with the kernel sequence (operator, category) and classifies each kernel by one rule:

- **tiny**: under 5 µs (launch latency and ramp dominate; fusing or removing it is the only lever);
- **bound by a unit**: the highest of DRAM, L2, L1/shared, issue, and every pipe's utilisation is at least 60% of
  its peak: that unit (``dram``, ``l2``, ``l1``, ``issue``, ``pipe:<name>``);
- **underfilled**: SMs active under 70% of the elapsed cycles (too few blocks, or a tail);
- **latency**: none of the above; labelled by the largest warp stall reason per issued instruction
  (``long_scoreboard`` global memory, ``short_scoreboard`` shared memory or MUFU, ``mio_throttle``,
  ``wait`` fixed latency, ``math_pipe_throttle``, ``barrier``, ``no_instruction``, ...).

Writes limiters.csv (one row per kernel), limiters_by_category.csv and limiters.md (the step's stages, then
categories and the 25 largest kernels with their limiter and evidence).

Usage: python tools/step_limiters.py JOB_DIR [JOB_DIR ...]
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step_breakdown as sb  # noqa: E402

TINY_US = 5.0
BOUND_PCT = 60.0
FILL_PCT = 70.0
STALL = "smsp__average_warps_issue_stalled_"
PIPE_INST = "sm__inst_executed_pipe_"
PIPE_CYC = "sm__pipe_"


def number(text):
    try:
        return float(str(text).replace(",", ""))
    except (TypeError, ValueError):
        return None


def raw_rows(path):
    with Path(path).open(newline="") as handle:
        lines = [line for line in handle if not line.startswith("==")]
    rows = list(csv.reader(lines))
    header, body = rows[0], rows[1:]
    if body and not str(body[0][0]).strip().isdigit():
        body = body[1:]
    out = []
    for cells in body:
        record = dict(zip(header, cells))
        for key in list(record):
            if "__" in key:
                record[key] = number(record[key])
        out.append(record)
    return out


def units(record):
    """{unit: percent of peak} for the throughput units and every pipe."""
    u = {"dram": record.get("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
         "l2": record.get("lts__throughput.avg.pct_of_peak_sustained_elapsed"),
         "l1": record.get("l1tex__throughput.avg.pct_of_peak_sustained_elapsed"),
         "issue": record.get("smsp__issue_active.avg.pct_of_peak_sustained_active")}
    for key, value in record.items():
        if value is None or not isinstance(value, float):
            continue
        if key.startswith(PIPE_INST) and key.endswith(".avg.pct_of_peak_sustained_active"):
            name = key[len(PIPE_INST):-len(".avg.pct_of_peak_sustained_active")]
            u[f"pipe:{name}"] = max(value, u.get(f"pipe:{name}") or 0.0)
        elif key.startswith(PIPE_CYC) and key.endswith("_cycles_active.avg.pct_of_peak_sustained_active"):
            name = key[len(PIPE_CYC):-len("_cycles_active.avg.pct_of_peak_sustained_active")]
            u[f"pipe:{name}"] = max(value, u.get(f"pipe:{name}") or 0.0)
    return {k: v for k, v in u.items() if v is not None}


def stalls(record):
    out = {}
    for key, value in record.items():
        if key.startswith(STALL) and key.endswith("_per_issue_active.ratio") and isinstance(value, float):
            name = key[len(STALL):-len("_per_issue_active.ratio")]
            if name not in ("selected", "not_selected"):
                out[name] = value
    return out


def classify(record):
    us = (record.get("gpu__time_duration.sum") or 0.0) / 1e3
    u = units(record)
    top_unit, top_pct = max(u.items(), key=lambda kv: kv[1]) if u else ("", 0.0)
    active = record.get("sm__cycles_active.avg")
    elapsed = record.get("sm__cycles_elapsed.avg")
    fill = 100.0 * active / elapsed if active and elapsed else None
    s = stalls(record)
    ranked = sorted(s.items(), key=lambda kv: -kv[1])
    if us < TINY_US:
        limiter = "tiny"
    elif top_pct >= BOUND_PCT:
        limiter = top_unit
    elif fill is not None and fill < FILL_PCT:
        limiter = "underfilled"
    else:
        limiter = f"latency:{ranked[0][0]}" if ranked else "latency"
    pipes = sorted(((k, v) for k, v in u.items() if k.startswith("pipe:")), key=lambda kv: -kv[1])[:3]
    sectors = record.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
    requests = record.get("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum")
    wavefronts = record.get("l1tex__data_pipe_lsu_wavefronts_mem_shared.sum")
    conflicts = record.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum")
    return dict(
        duration_us=us, limiter=limiter, top_unit=top_unit, top_unit_pct=top_pct,
        sm_fill_pct=fill,
        sm_pct=record.get("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        mem_pct=record.get("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"),
        dram_pct=u.get("dram"), l2_pct=u.get("l2"), l1_pct=u.get("l1"), issue_pct=u.get("issue"),
        top_pipes=" ".join(f"{k[5:]}={v:.0f}" for k, v in pipes),
        top_stalls=" ".join(f"{k}={v:.2f}" for k, v in ranked[:3]),
        warps_active_pct=record.get("sm__warps_active.avg.pct_of_peak_sustained_active"),
        theoretical_occupancy_pct=record.get("sm__maximum_warps_per_active_cycle_pct"),
        registers=record.get("launch__registers_per_thread"),
        occupancy_limits="/".join(str(int(record[k])) if record.get(k) is not None else "?" for k in (
            "launch__occupancy_limit_registers", "launch__occupancy_limit_shared_mem",
            "launch__occupancy_limit_warps", "launch__occupancy_limit_blocks")),
        grid=record.get("launch__grid_size"), block=record.get("launch__block_size"),
        waves=record.get("launch__waves_per_multiprocessor"),
        sectors_per_ld_request=sectors / requests if sectors and requests else None,
        smem_conflict_share=conflicts / wavefronts if conflicts is not None and wavefronts else None,
        threads_per_inst=record.get("smsp__thread_inst_executed_per_inst_executed.ratio"),
        l2_hit_pct=record.get("lts__t_sector_hit_rate.pct"),
    )


def fmt(v, digits=1):
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def analyse(job: Path):
    raw = raw_rows(job / "ncu_raw.csv")
    with (job / "main" / "kernel_sequence.csv").open(newline="") as handle:
        sequence = list(csv.DictReader(handle))
    joined, matched = sb.join_ncu([dict(r) for r in raw], sequence, None)
    rows = []
    for position, (record, j) in enumerate(zip(raw, joined)):
        c = classify(record)
        rows.append(dict(position=position, category=j["category"], op=j["op"],
                         kernel=str(record.get("Kernel Name", ""))[:90], **c))
    total = sum(r["duration_us"] for r in rows) or 1.0
    fields = list(rows[0])
    with (job / "limiters.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(v, 3) for k, v in r.items()})
    by_cat = defaultdict(lambda: defaultdict(float))
    for r in rows:
        by_cat[r["category"]]["_total"] += r["duration_us"]
        by_cat[r["category"]][r["limiter"]] += r["duration_us"]
    by_lim = defaultdict(float)
    for r in rows:
        by_lim[r["limiter"].split(":")[0] if r["limiter"].startswith("latency") else r["limiter"]] += r["duration_us"]
    with (job / "limiters_by_category.csv").open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["category", "us", "share", "limiters (us)"])
        for cat, d in sorted(by_cat.items(), key=lambda kv: -kv[1]["_total"]):
            mix = " ".join(f"{k}={v:.0f}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]) if k != "_total")
            w.writerow([cat, f"{d['_total']:.1f}", f"{d['_total'] / total:.3f}", mix])
    gpu = (job / "gpu.txt").read_text().strip() if (job / "gpu.txt").is_file() else job.name
    summary = json.loads((job / "main" / "summary.json").read_text()) if (job / "main" / "summary.json").is_file() else {}
    md = [f"# Limiters: {gpu}", "",
          f"ncu kernels: {len(rows)} ({matched} aligned with the profiler's sequence); kernel time {total:.1f} µs "
          "per force call (ncu, running clocks, warm caches).", ""]
    stages = job / "main" / "stages.csv"
    if stages.is_file():
        md += ["## Step stages (main pass)", "", "```", stages.read_text().strip(), "```", ""]
    md += ["## Kernel time by limiter", "", "| Limiter | µs | Share |", "| --- | ---: | ---: |"]
    for k, v in sorted(by_lim.items(), key=lambda kv: -kv[1]):
        md.append(f"| {k} | {v:.1f} | {v / total:.1%} |")
    md += ["", "## By category", "", "| Category | µs | Share | Limiters (µs) |", "| --- | ---: | ---: | --- |"]
    for cat, d in sorted(by_cat.items(), key=lambda kv: -kv[1]["_total"]):
        mix = ", ".join(f"{k} {v:.0f}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]) if k != "_total")
        md.append(f"| {cat} | {d['_total']:.1f} | {d['_total'] / total:.1%} | {mix} |")
    md += ["", "## 25 largest kernels", "",
           "| # | Kernel | Op | µs | Limiter | SM fill | DRAM | L2 | L1 | Issue | Pipes | Stalls | Warps act/theo | Regs | Occ. limits r/s/w/b | Sect/ld | Smem confl | Thr/inst |",
           "| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: |"]
    for r in sorted(rows, key=lambda r: -r["duration_us"])[:25]:
        md.append("| " + " | ".join([
            str(r["position"]), r["kernel"][:50].replace("|", "/"), str(r["op"])[:40].replace("|", "/"),
            fmt(r["duration_us"]), r["limiter"], fmt(r["sm_fill_pct"], 0), fmt(r["dram_pct"], 0), fmt(r["l2_pct"], 0),
            fmt(r["l1_pct"], 0), fmt(r["issue_pct"], 0), r["top_pipes"], r["top_stalls"],
            f"{fmt(r['warps_active_pct'], 0)}/{fmt(r['theoretical_occupancy_pct'], 0)}", fmt(r["registers"], 0),
            r["occupancy_limits"], fmt(r["sectors_per_ld_request"]), fmt(r["smem_conflict_share"], 2),
            fmt(r["threads_per_inst"])]) + " |")
    if summary:
        md += ["", "## Main-pass summary keys", "", "```", json.dumps({k: summary[k] for k in list(summary)[:20]},
                                                                    indent=1, default=str)[:4000], "```"]
    (job / "limiters.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{job}: {len(rows)} kernels, {total:.1f} us; " +
          ", ".join(f"{k} {v / total:.0%}" for k, v in sorted(by_lim.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        analyse(Path(arg))
