"""Summarize the CUTLASS bench's long.csv into shapes.csv and summary.json (stdlib only).

Usage: python summarize.py RUN_DIR   (RUN_DIR holds long.csv, capture.json, cutlass_version.txt, src/shapes_in.tsv,
src/shapes_20261023_a100.csv, cu-before.txt, cu-after.txt)
"""
import csv
import json
import sys
from pathlib import Path

MODES = ("cutlass_3xtf32", "cutlass_1xtf32", "cublas_ieee", "cublas_dgemm")


def main(run):
    run = Path(run)
    long = list(csv.DictReader(open(run / "long.csv", newline="")))
    shapes = [line.split("\t") for line in (run / "src/shapes_in.tsv").read_text().splitlines() if line]
    orig = list(csv.DictReader(open(run / "src/shapes_20261023_a100.csv", newline="")))
    best, statuses = {}, {}
    for r in long:
        key = (int(r["idx"]), r["mode"])
        if r["status"] != "ok":
            statuses.setdefault(key, []).append(f'{r["tile"]}: {r["status"]}')
            continue
        r = dict(r, us=float(r["us"]), tflops=float(r["tflops"]), max_rel=float(r["max_rel"]),
                 frob_rel=float(r["frob_rel"]))
        if key not in best or r["us"] < best[key]["us"]:
            best[key] = r
    rows = []
    for s in shapes:
        i = int(s[0])
        o = orig[i]
        row = dict(idx=i, structure=s[1], m=int(s[2]), k=int(s[3]), n=int(s[4]), a_layout=s[5], lda=int(s[6]),
                   a_align=int(s[7]), b_layout=s[8], ldb=int(s[9]), b_align=int(s[10]), c_align=int(s[11]),
                   a_stride=o["a_stride"], b_stride=o["b_stride"], op=o["op"], category=o["category"],
                   invocations_per_call=float(s[13]),
                   torch_cublas_ieee_us_20261023=float(o["cublas_ieee_us"]),
                   torch_cublas_tf32_us_20261023=float(o["cublas_tf32_us"]))
        for mode in MODES:
            b = best.get((i, mode))
            if b:
                row[f"{mode}_us"] = b["us"]
                row[f"{mode}_tflops"] = b["tflops"]
                if mode != "cublas_dgemm":
                    row[f"{mode}_max_rel"] = b["max_rel"]
                    row[f"{mode}_frob_rel"] = b["frob_rel"]
                if mode.startswith("cutlass"):
                    row[f"{mode}_tile"] = b["tile"]
            if (i, mode) in statuses:
                row[f"{mode}_not_run"] = "; ".join(statuses[(i, mode)])
        rows.append(row)
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with open(run / "shapes.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    peak = {}
    for mode in MODES:
        b = best.get((-1, mode))
        if b:
            peak[mode] = dict(us=b["us"], tflops=b["tflops"], tile=b["tile"], max_rel=b["max_rel"],
                              frob_rel=b["frob_rel"])
    per = {}
    for row in rows:
        t = per.setdefault(row["structure"], {m: 0.0 for m in MODES} | {"torch_cublas_ieee_20261023": 0.0,
                                                                          "torch_cublas_tf32_20261023": 0.0})
        for mode in MODES:
            if f"{mode}_us" in row:
                t[mode] += row[f"{mode}_us"] * row["invocations_per_call"]
        t["torch_cublas_ieee_20261023"] += row["torch_cublas_ieee_us_20261023"] * row["invocations_per_call"]
        t["torch_cublas_tf32_20261023"] += row["torch_cublas_tf32_us_20261023"] * row["invocations_per_call"]
    for t in per.values():
        t["ratio_3xtf32_over_cublas_ieee"] = t["cutlass_3xtf32"] / t["cublas_ieee"]
        t["ratio_1xtf32_over_cublas_ieee"] = t["cutlass_1xtf32"] / t["cublas_ieee"]
    won = {s: sum(1 for r in rows if r["structure"] == s and r.get("cutlass_3xtf32_us", 1e30) < r["cublas_ieee_us"])
           for s in per}
    # Best-of per shape: what a dispatcher that picks min(3xTF32, cuBLAS IEEE) would get.
    for s, t in per.items():
        t["best_of_3xtf32_and_cublas_ieee"] = sum(
            min(r.get("cutlass_3xtf32_us", 1e30), r["cublas_ieee_us"]) * r["invocations_per_call"]
            for r in rows if r["structure"] == s)
    summary = dict(
        tool="opt-T01 CUTLASS 3xTF32 bench (bench.cu)",
        cutlass=(run / "cutlass_version.txt").read_text().strip(),
        timing="5 warm-up calls, then 50 calls each between CUDA events, median (peak: 10 calls)",
        errors="against cublasDgemm of the same float32 inputs: max|C-R|/max|R| and ||C-R||_F/||R||_F",
        note="cutlass modes: fastest of 3 fixed tile configs per shape (a screen's upper bound); "
             "times are from a C++ binary (no PyTorch dispatch), so they are lower than the 23 October torch.mm times",
        peak_4096=peak,
        capture=json.loads((run / "capture.json").read_text()),
        gemm_us_per_force_call=per,
        shapes_3xtf32_faster_than_cublas_ieee=won,
        shapes_3xtf32_faster_than_cublas_ieee_total=sum(won.values()),
        worst_error={m: max(r.get(f"{m}_max_rel", 0.0) for r in rows) for m in MODES if m != "cublas_dgemm"},
        worst_frob_error={m: max(r.get(f"{m}_frob_rel", 0.0) for r in rows) for m in MODES if m != "cublas_dgemm"},
        not_run={f"{i}:{m}": v for (i, m), v in statuses.items()},
        shapes=len(rows),
        cu_before=(run / "cu-before.txt").read_text().strip() if (run / "cu-before.txt").exists() else None,
        cu_after=(run / "cu-after.txt").read_text().strip() if (run / "cu-after.txt").exists() else None,
    )
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("cutlass", "peak_4096", "capture", "gemm_us_per_force_call",
                                              "shapes_3xtf32_faster_than_cublas_ieee", "worst_error")}, indent=1))
    print("not run:", len(summary["not_run"]))


if __name__ == "__main__":
    main(sys.argv[1])
