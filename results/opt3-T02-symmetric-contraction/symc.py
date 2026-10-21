import csv
import sys

J = sys.argv[1]
for s in ("NaCl", "NbCl5", "NaNbO3"):
    rows = list(csv.DictReader(open(f"{J}/{s}/ncu_kernels.csv")))
    tot = sum(float(r["duration_us"] or 0) for r in rows)
    sym = [r for r in rows if "segmented_polynomial" in r["ncu_kernel"]]
    matched = sum(r["name_matches"] == "True" for r in rows)
    print("==", s, "total ncu us", round(tot), "matched", matched, "/", len(rows))
    sym_us = 0.0
    for r in sym:
        us = float(r["duration_us"])
        sym_us += us
        sec = us * 1e-6
        bw = (float(r["dram_read_bytes"] or 0) + float(r["dram_write_bytes"] or 0)) / sec / 1e9
        print("  ", r["index"], r["ncu_kernel"][:40], "|", r["category"][:28], "| us", round(us, 1),
              "| %.1f%%" % (100 * us / tot), "| DRAM GB/s", round(bw), "| TFLOPS", r["achieved_tflops"][:5],
              "| SOL sm/mem", r["sm_throughput_pct"], r["memory_throughput_pct"],
              "| occ", r["achieved_occupancy_pct"], "/", r["theoretical_occupancy_pct"],
              "| grid", r["grid_size"], "block", r["block_size"], "regs", r["registers"],
              "| L2 MB", round(float(r["l2_bytes"] or 0) / 1e6, 1),
              "| DRAM MB", round((float(r["dram_read_bytes"] or 0) + float(r["dram_write_bytes"] or 0)) / 1e6, 1))
    print("   sum", round(sym_us, 1), "us = %.1f%% of ncu kernel time" % (100 * sym_us / tot))
