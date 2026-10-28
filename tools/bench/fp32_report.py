"""Markdown tables from an fp32 profile run directory (fp32 profile, 28 October 2026).

    python tools/bench/fp32_report.py RUN_DIR [--label "RTX 5060 Ti, local"] > tables.md

RUN_DIR holds one directory per structure (step_breakdown's main pass, with ``parts/`` from fp32_parts and,
locally, the ncu and nsys exports) and ``setup-*`` directories from fp32_setup. Missing pieces are skipped.
Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

STRUCTURES = ("NaCl", "NaNbO3", "NbCl5")


def rows(path):
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        lines = [line for line in handle if line.strip()]
    start = next((i for i, line in enumerate(lines) if "," in line), 0)
    return list(csv.DictReader(lines[start:]))


def load(path):
    return json.loads(path.read_text()) if path.is_file() else None


def fmt(value, digits=2):
    if value is None or value == "":
        return "–"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def table(header, body):
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" if i == 0 else "---:" for i in range(len(header)))
           + " |"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(out)


def short(name, n=70):
    name = re.sub(r"\s+", " ", name)
    return name if len(name) <= n else name[: n - 1] + "…"


def stages(run, present):
    keys, data = [], {}
    for s in present:
        summary = load(run / s / "summary.json")
        data[s] = summary
        for k in summary["stage_ms_per_step"]:
            if k not in keys:
                keys.append(k)
    body = [["Wall time, uninstrumented (ms/step)"] + [fmt(data[s]["wall_ms_per_step"], 3) for s in present],
            ["GPU span, instrumented (ms/step)"] + [fmt(data[s]["gpu_span_ms_per_step"], 3) for s in present]]
    for k in keys:
        body.append([k] + [fmt(data[s]["stage_ms_per_step"].get(k), 3) for s in present])
    body.append(["GPU idle between stages"] + [fmt(data[s]["gpu_idle_ms_per_step"], 3) for s in present])
    body.append(["Rebuilds per 1000 steps (instrumented window)"]
                + [str(data[s]["instrumented"]["rebuilds"]) for s in present])
    body.append(["Padded edges"] + [str(data[s]["padded_edges"]) for s in present])
    body.append(["Atoms"] + [str(data[s]["natoms"]) for s in present])
    rp = [data[s].get("force_graph_replay_profile") or {} for s in present]
    body.append(["B replay: kernels"] + [fmt(r.get("kernels_per_replay"), 0) for r in rp])
    body.append(["B replay: kernel ms / span ms"] + [f"{fmt(r.get('kernel_ms_per_replay'), 3)} / "
                                                     f"{fmt(r.get('span_ms_per_replay'), 3)}" for r in rp])
    body.append(["Eager force call: kernel ms (kernels)"] + [
        f"{fmt(data[s]['eager_force_kernel_ms_per_call'], 3)} ({fmt(data[s]['eager_kernels_per_call'], 0)})"
        for s in present])
    return table(["Stage"] + list(present), body), data


def categories(run, present):
    cats, data = [], {}
    for s in present:
        data[s] = {r["category"]: r for r in rows(run / s / "categories.csv")}
        for k in data[s]:
            if k not in cats:
                cats.append(k)
    first = present[-1]
    cats.sort(key=lambda k: -float(data[first].get(k, {}).get("ms_per_force_call") or 0))
    body = []
    for k in cats:
        cells = []
        for s in present:
            r = data[s].get(k)
            cells.append("–" if r is None else f"{fmt(r['ms_per_force_call'], 3)} ({float(r['share_of_kernel_time']):.0%}, "
                         f"{fmt(r['kernels_per_call'], 0)})")
        body.append([k] + cells)
    return table(["Part (compiled eager force call): ms (share, kernels)"] + list(present), body)


REFINED = (  # (substring of "op kernel", part): step_breakdown's categories predate these kernels
    ("fused_folded_forward", "Int. 2: radial last layer + convolution forward (fused_folded_forward)"),
    ("streaming_conv_folded_backward", "Int. 2: convolution backward, chunked (oeq_projected_backward)"),
    ("oeq_projected_backward", "Int. 2: convolution backward, chunked (oeq_projected_backward)"),
    ("folded_linear_input_grad", "Int. 2: radial last layer, input gradient (Triton)"),
    ("radial_table", "Radial tables (hidden radial chain, forward + backward)"),
    ("joined_products", "Atom GEMMs: segment_gemm (joined products, Triton)"),
    ("edge_outer_sum", "Int. 1: reassociated edge sums (Triton)"),
    ("cuequivariance", "Symmetric contraction (cuEq)"),
    ("splitKreduce", "Atom GEMMs: cuBLAS (incl. split-K reduce, memset)"),
)


def refined_part(row):
    name = f"{row['op']} {row['kernel']}"
    for key, part in REFINED:
        if key in name:
            return part
    if row["op"] in ("aten::mm", "aten::addmm", "aten::bmm"):
        if row["category"].startswith("Radial"):
            return row["category"]
        return "Atom GEMMs: cuBLAS (incl. split-K reduce, memset)"
    if row["category"].startswith("Edge elementwise"):
        return "Edge elementwise (geometry, basis, gathers, scatters)"
    return row["category"]


def refined(run, present):
    data, parts = {}, []
    for s in present:
        agg = {}
        for r in rows(run / s / "ops.csv"):
            p = refined_part(r)
            us, n = agg.get(p, (0.0, 0.0))
            agg[p] = (us + float(r["us_per_call"]), n + float(r["launches_per_call"]))
        data[s] = agg
        for p in agg:
            if p not in parts:
                parts.append(p)
    last = present[-1]
    parts.sort(key=lambda p: -data[last].get(p, (0, 0))[0])
    totals = {s: sum(v[0] for v in data[s].values()) for s in present}
    body = []
    for p in parts:
        body.append([p] + [("–" if p not in data[s] else
                            f"{data[s][p][0] / 1e3:.3f} ({data[s][p][0] / totals[s]:.0%}, {data[s][p][1]:g})")
                           for s in present])
    body.append(["Total eager kernel time"] + [f"{totals[s] / 1e3:.3f}" for s in present])
    return table(["Part (compiled eager force call, refined): ms (share, kernels)"] + list(present), body)


def top_ops(run, s, n=20):
    body = []
    for r in rows(run / s / "ops.csv")[:n]:
        body.append([fmt(r["us_per_call"], 1), fmt(r["launches_per_call"], 0), short(r["category"], 40),
                     short(r["op"], 40), short(r["kernel"], 60), r["grid"], r["est_achieved_occupancy_pct"]])
    return table(["µs/call", "launches", "category", "op", "kernel", "grid", "est. occ. %"], body)


def gemms(run, s, n=12):
    body = []
    for r in rows(run / s / "gemms.csv")[:n]:
        body.append([short(r["category"], 40), short(r["op"], 20), fmt(r["us_per_call"], 1), r["invocations_per_call"],
                     fmt(r["achieved_tflops"], 2), fmt(r["share_of_fp32_peak"], 2), short(r["input_dims"], 60)])
    return table(["category", "op", "µs/call", "calls", "TFLOPS", "of peak", "shapes"], body)


def ncu(run, present):
    data, cats = {}, []
    for s in present:
        data[s] = {r["category"]: r for r in rows(run / s / "ncu_categories.csv")}
        for k in data[s]:
            if k not in cats:
                cats.append(k)
    if not any(data.values()):
        return None
    first = present[-1]
    cats.sort(key=lambda k: -float(data[first].get(k, {}).get("duration_us") or 0))
    body = []
    for k in cats:
        cells = []
        for s in present:
            r = data[s].get(k)
            cells.append("–" if r is None else f"{fmt(r['dram_mb'], 0)} MB, {fmt(r['dram_gb_per_s'], 0)} GB/s, "
                         f"occ {fmt(r['achieved_occupancy_pct'], 0)} %")
        body.append([k] + cells)
    return table(["Part (ncu, one eager call): DRAM, bandwidth, occupancy"] + list(present), body)


def nsys_steps(run, present):
    out = []
    for s in present:
        record = load(run / s / "nsys-steps_pass.json")
        proj = rows(run / s / "nsys_steps_stats_nvtx_gpu_proj_sum.csv")
        if not record or not proj:
            continue
        steps = record.get("steps") or 1
        body = []
        for r in proj:
            name = r.get("Range") or r.get("Name") or ""
            total = r.get("Total Proj Time (ns)") or r.get("Total Time (ns)")
            inst = r.get("Range Instances") or r.get("Instances")
            body.append([short(name, 50), inst, fmt(float(total) / 1e6 / steps if total else None, 3),
                         fmt(float(total) / 1e3 / float(inst) if total and inst else None, 1)])
        out.append(f"**{s}** ({steps} steps, {record.get('rebuilds')} rebuilds)\n\n"
                   + table(["NVTX range (GPU projection)", "instances", "ms per step", "µs per instance"], body))
    return "\n\n".join(out) or None


def parts(run, present, depth=2):
    data, keys = {}, []
    for s in present:
        data[s] = {(r["direction"], r["module"]): r for r in rows(run / s / "parts" / "parts_tree.csv")
                   if int(r["depth"]) <= depth}
        for k in data[s]:
            if k not in keys:
                keys.append(k)
    if not any(data.values()):
        return None
    first = present[-1]
    keys.sort(key=lambda k: (k[1].split(".")[0], k[1], k[0]))
    body = []
    for k in keys:
        cells = []
        for s in present:
            r = data[s].get(k)
            cells.append("–" if r is None else f"{fmt(float(r['inclusive_us_per_call']) / 1e3, 3)} "
                         f"({float(r['share']):.0%}, {fmt(r['launches_per_call'], 0)})")
        body.append([f"{k[0]} {k[1]}"] + cells)
    sums = []
    for s in present:
        summary = load(run / s / "parts" / "summary.json") or {}
        sums.append(f"{fmt((summary.get('eager_kernel_us_per_call_without_hooks') or 0) / 1e3, 3)} / "
                    f"{fmt((summary.get('eager_kernel_us_per_call_with_hooks') or 0) / 1e3, 3)} / "
                    f"{fmt((summary.get('unattributed_us_per_call') or 0) / 1e3, 3)}")
    body.append(["eager kernel ms without hooks / with hooks / outside every range"] + sums)
    return table([f"Module (eager tier, inclusive; depth ≤ {depth}): ms (share, kernels)"] + list(present), body)


def part_top(run, s, n=25):
    body = []
    for r in rows(run / s / "parts" / "parts.csv")[:n]:
        body.append([short(r["range"], 60), fmt(float(r["exclusive_us_per_call"]) / 1e3, 3), r["launches_per_call"],
                     f"{float(r['share']):.1%}"])
    return table(["Range (exclusive)", "ms/call", "kernels", "share"], body)


def setups(run):
    out = []
    for path in sorted(run.glob("setup-*/setup.json")):
        d = load(path)
        head = (f"**{path.parent.name}** ({d['structure']}, {d['label']}): fm.md call → first step done "
                f"{fmt(d['setup_to_first_step_s'], 1)} s; process start → first step {fmt(d['process_to_first_step_s'], 1)} s "
                f"(import fastmace {fmt(d['import_fastmace_s'], 1)} s, torch {fmt(d['import_torch_s'], 1)} s); "
                f"first step {fmt((d.get('first_step') or {}).get('step_s'), 2)} s; steady state with output "
                f"{fmt(d['steady_ms_per_step_with_output'], 3)} ms/step")
        sim = d.get("simulation_setup_s") or {}
        body = [[label, fmt(v["total_s"], 3), v["calls"], ", ".join(v["parents"])[:60]]
                for label, v in list(d["stages"].items())[:30]]
        out.append(head + "\n\nSimulation._setup_s: " + ", ".join(f"{k} {fmt(v, 2)} s" for k, v in sim.items())
                   + "\n\n" + table(["Stage (nested timers)", "s", "calls", "inside"], body)
                   + ("\n\ntorch.compile phases:\n\n```\n" + str(d.get("dynamo_compile_times"))[:1500] + "\n```"))
    return "\n\n".join(out) or None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("--label", default="")
    args = p.parse_args(argv)
    run = args.run
    present = [s for s in STRUCTURES if (run / s / "summary.json").is_file()]
    print(f"# fp32 profile tables: {args.label} ({run})\n")
    if present:
        text, data = stages(run, present)
        s0 = data[present[0]]
        print(f"GPU {s0['gpu']} (sm_{s0['compute_capability'].replace('.', '')}, {s0['sms']} SMs, max clock "
              f"{s0['max_sm_clock_mhz']} MHz, FP32 peak {fmt(s0['fp32_peak_tflops'], 1)} TFLOPS); FastMACE "
              f"{s0['fastmace_commit']}; torch {s0['packages'].get('torch')}\n")
        print("## Step stages\n\n" + text + "\n")
        print("## Force call by part (compiled kernels, refined)\n\n" + refined(run, present) + "\n")
        print("## Force call by step_breakdown category\n\n" + categories(run, present) + "\n")
        for s in present:
            print(f"## Top operators, {s}\n\n" + top_ops(run, s) + "\n")
            print(f"## GEMMs, {s}\n\n" + gemms(run, s) + "\n")
        for title, text in (("Nsight Compute", ncu(run, present)), ("Whole steps under Nsight Systems",
                                                                     nsys_steps(run, present)),
                            ("Model parts, eager tier", parts(run, present))):
            if text:
                print(f"## {title}\n\n{text}\n")
        for s in present:
            if (run / s / "parts" / "parts.csv").is_file():
                print(f"## Eager ranges, {s}\n\n" + part_top(run, s) + "\n")
        for s in present:
            g = data[s].get("nsys_force_graph_gaps")
            r = (data[s].get("force_graph_replay_profile") or {}).get("gaps_inside_replays")
            print(f"Gaps {s}: profiler {json.dumps(r)[:400]}; nsys {json.dumps(g)[:400]}\n")
    text = setups(run)
    if text:
        print("## Setup\n\n" + text + "\n")


if __name__ == "__main__":
    main()
