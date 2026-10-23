"""Summarize the CUTLASS bench's long.csv (operand sets gaussian / cancellation / exponent) into shapes.csv and
summary.json (stdlib only).

Timing per shape: fastest tile per mode on the Gaussian set. Errors per shape, set and mode: the worst over the
tiles that ran (conservative; for cuBLAS there is one kernel). Ratios: per shape, 3xTF32's error over cuBLAS IEEE's.
Usage: python summarize.py RUN_DIR
"""
import csv
import json
import statistics
import sys
from pathlib import Path

MODES = ("cutlass_3xtf32", "cutlass_1xtf32", "cublas_ieee", "cublas_dgemm")
SETS = ("gaussian", "cancellation", "exponent")
METRICS = ("max_rel", "frob_rel", "cancel")


def opt(path):
    return path.read_text().strip() if path.exists() else None


def main(run):
    run = Path(run)
    long = list(csv.DictReader(open(run / "long.csv", newline="")))
    shapes = [line.split("\t") for line in (run / "src/shapes_in.tsv").read_text().splitlines() if line]
    orig = list(csv.DictReader(open(run / "src/shapes_20261023_a100.csv", newline="")))
    fastest, err, bad = {}, {}, {}
    for r in long:
        i, s, mode = int(r["idx"]), r["set"], r["mode"]
        if r["status"] != "ok":
            bad.setdefault(f"{i}:{s}:{mode}", []).append(f'{r["tile"]}: {r["status"]}')
            continue
        if s == "gaussian":
            us = float(r["us"])
            if (i, mode) not in fastest or us < fastest[(i, mode)]["us"]:
                fastest[(i, mode)] = dict(us=us, tflops=float(r["tflops"]), tile=r["tile"])
        if mode != "cublas_dgemm":
            e = err.setdefault((i, s, mode), {m: 0.0 for m in METRICS})
            for m in METRICS:
                e[m] = max(e[m], float(r[m]))
    rows = []
    for sh in shapes:
        i = int(sh[0])
        o = orig[i]
        row = dict(idx=i, structure=sh[1], m=int(sh[2]), k=int(sh[3]), n=int(sh[4]), a_layout=sh[5], lda=int(sh[6]),
                   b_layout=sh[8], ldb=int(sh[9]), a_stride=o["a_stride"], b_stride=o["b_stride"], op=o["op"],
                   category=o["category"], invocations_per_call=float(sh[13]))
        for mode in MODES:
            f = fastest.get((i, mode))
            if f:
                row[f"{mode}_us"], row[f"{mode}_tflops"] = f["us"], f["tflops"]
                if mode.startswith("cutlass"):
                    row[f"{mode}_tile"] = f["tile"]
        for s in SETS:
            for mode in MODES[:3]:
                for m in METRICS:
                    row[f"{s}_{mode}_{m}"] = err[(i, s, mode)][m]
            for m in METRICS:
                ieee = err[(i, s, "cublas_ieee")][m]
                row[f"{s}_ratio_3xtf32_over_ieee_{m}"] = err[(i, s, "cutlass_3xtf32")][m] / ieee if ieee else None
        rows.append(row)
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with open(run / "shapes.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    per = {}
    for row in rows:
        t = per.setdefault(row["structure"], {m: 0.0 for m in MODES})
        for mode in MODES:
            t[mode] += row[f"{mode}_us"] * row["invocations_per_call"]
    for t in per.values():
        t["ratio_3xtf32_over_cublas_ieee"] = t["cutlass_3xtf32"] / t["cublas_ieee"]
    worst = {s: {mode: {m: max(r[f"{s}_{mode}_{m}"] for r in rows) for m in METRICS} for mode in MODES[:3]}
             for s in SETS}
    ratios = {}
    for s in SETS:
        ratios[s] = {}
        for m in METRICS:
            v = [r[f"{s}_ratio_3xtf32_over_ieee_{m}"] for r in rows if r[f"{s}_ratio_3xtf32_over_ieee_{m}"] is not None]
            top = max(rows, key=lambda r: r[f"{s}_ratio_3xtf32_over_ieee_{m}"] or 0)
            ratios[s][m] = dict(worst=max(v), median=statistics.median(v), min=min(v),
                                shapes_3xtf32_worse=sum(1 for x in v if x > 1), n=len(v),
                                worst_shape=f'{top["structure"]} {top["m"]}x{top["k"]}x{top["n"]}')
    peak = {mode: fastest[(-1, mode)] for mode in MODES if (-1, mode) in fastest}
    summary = dict(
        tool="opt-T01 CUTLASS 3xTF32 bench with cancellation / exponent operand sets (bench.cu)",
        cutlass=opt(run / "cutlass_version.txt"),
        operand_sets={
            "gaussian": "randn (timed: 5 warm-up, 50 event-timed calls, median; peak 10 calls)",
            "cancellation": "BF16x6 probe construction: per column pair q one u ~ U[0,1), A[:,2q+1] = -(A[:,2q](1+1e-3u)),"
                            " B[2q+1,:] = B[2q,:]; odd last column stays randn",
            "exponent": "randn, row i of A x 2^e_i, column j of B x 2^f_j, integer e, f ~ U{-20..20}"},
        errors="vs cublasDgemm of the same float32 inputs: max_rel = max|C-R|/max|R|, frob_rel = ||C-R||_F/||R||_F, "
               "cancel = max |C-R|/(|A||B| + 1e-30) with |A||B| by DGEMM; cutlass errors are the worst of the 3 tiles",
        peak_4096=peak,
        capture=json.loads((run / "capture.json").read_text()),
        gemm_us_per_force_call_gaussian=per,
        shapes_3xtf32_faster_than_cublas_ieee=sum(1 for r in rows if r["cutlass_3xtf32_us"] < r["cublas_ieee_us"]),
        worst_error=worst,
        ratio_3xtf32_over_ieee=ratios,
        not_run=bad,
        shapes=len(rows),
        cu_before=opt(run / "cu-before.txt"),
        cu_after=opt(run / "cu-after.txt"),
    )
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("cutlass", "peak_4096", "capture", "gemm_us_per_force_call_gaussian",
                                              "shapes_3xtf32_faster_than_cublas_ieee", "worst_error",
                                              "ratio_3xtf32_over_ieee")}, indent=1))
    print("not run:", len(bad))


if __name__ == "__main__":
    main(sys.argv[1])
