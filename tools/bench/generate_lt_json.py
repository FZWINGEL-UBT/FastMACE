import glob
import json
import os

data = {}
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
                d = {p.split("=", 1)[0]: p.split("=", 1)[1] for p in parts[1:] if "=" in p}
                if "us" in d:
                    rows.append(d)
    rows.sort(key=lambda x: float(x["us"]))
    data[name] = {"baseline_sgemm_us": base, "candidates": rows}

out_path = "/mnt/d/Margraf/FastMACE/results/opt5-T09-screens-20261026/cublaslt_summary.json"
with open(out_path, "w") as f:
    json.dump(data, f, indent=2)
print("Saved cublaslt_summary.json with", len(data), "shapes")
