"""Per-GEMM breakdown of the TF32 screen (29 October 2026): tools/step_breakdown.py's gemms.csv and categories.csv
with and without --tf32, per category and the largest per-shape savings.
   python gemm_breakdown.py RUN_DIR   (reads RUN_DIR/sb-{fp32,tf32}/<structure>/)"""
import collections
import csv
import sys

root = sys.argv[1]
for s in ("NaCl", "NaNbO3", "NbCl5"):
    rows = {m: list(csv.DictReader(open(f"{root}/sb-{m}/{s}/gemms.csv"))) for m in ("fp32", "tf32")}
    total = {m: sum(float(r["us_per_call"]) for r in rows[m]) for m in rows}
    print(f"== {s}: GEMM ops {len(rows['fp32'])}; GEMM us/call fp32 {total['fp32']:.0f} tf32 {total['tf32']:.0f}")
    by_category = collections.defaultdict(lambda: [0.0, 0.0])
    by_shape = {}
    for i, m in enumerate(("fp32", "tf32")):
        for r in rows[m]:
            by_category[r["category"]][i] += float(r["us_per_call"])
            entry = by_shape.setdefault((r["op"], r["input_dims"]), [0.0, 0.0, "", ""])
            entry[i] += float(r["us_per_call"])
            entry[2 + i] = r["kernels"][:48]
    for c, (a, b) in sorted(by_category.items(), key=lambda x: -x[1][0]):
        print(f"  {c:45s} fp32 {a:8.0f} tf32 {b:8.0f} us  saved {a - b:7.0f} ({(b / a if a else 0):.2f}x)")
    for (op, dims), (a, b, ka, kb) in sorted(by_shape.items(), key=lambda x: x[1][1] - x[1][0])[:6]:
        print(f"   top: {op} {dims[:60]} fp32 {a:.1f} tf32 {b:.1f} us | {ka} -> {kb}")
    for m in ("fp32", "tf32"):
        categories = {r["category"]: float(r["ms_per_force_call"])
                      for r in csv.DictReader(open(f"{root}/sb-{m}/{s}/categories.csv"))}
        print(f"  {m} kernel ms per force call {sum(categories.values()):.3f}:",
              {c[:30]: round(v, 3) for c, v in categories.items()})
