import os
import glob

for path in sorted(glob.glob("/home/fzwin/tmp/lt_*.txt")):
    name = os.path.basename(path).replace(".txt", "")
    rows = []
    base = None
    with open(path) as f:
        for line in f:
            if line.startswith("cublas_sgemm"):
                base = float(line.strip().split(",")[1])
            elif line.startswith("lt_algo"):
                parts = line.strip().split(",")
                d = {}
                for p in parts[1:]:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        d[k] = v
                if "us" in d:
                    rows.append((float(d["us"]), d))
    rows.sort(key=lambda x: x[0])
    print(name, "baseline_sgemm_us:", base, "total_candidates:", len(rows))
    for us, d in rows[:5]:
        pct = (us / base - 1.0) * 100.0 if base else 0.0
        ws_val = d.get("ws")
        idx_val = d.get("idx")
        tile_val = d.get("tile")
        splitk_val = d.get("splitk")
        stages_val = d.get("stages")
        diff_val = d.get("maxdiff")
        print(f"  best: {us:.2f} us ({pct:+.1f}%) | ws={ws_val} idx={idx_val} tile={tile_val} splitk={splitk_val} stages={stages_val} diff={diff_val}")
