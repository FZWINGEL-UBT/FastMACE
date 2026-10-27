import csv
import json
from pathlib import Path

for d in sorted(Path(".").glob("share-*")):
    if not d.is_dir():
        continue
    s = d / "summary.json"
    if not s.exists():
        print(d.name, "no summary")
        continue
    summary = json.loads(s.read_text())
    step = summary.get("gpu_span_ms_per_step")
    stages = summary.get("stage_ms_per_step", {})
    rebuild = next((v for k, v in stages.items() if "rebuild" in k.lower()), None)
    cats = {r["category"]: float(r["ms_per_force_call"]) for r in csv.DictReader(open(d / "categories.csv"))}
    total = sum(cats.values())
    conv = {k: v for k, v in cats.items() if "convolution" in k.lower() or "edge sums" in k.lower()}
    bwd = sum(v for k, v in conv.items() if "backward" in k.lower())
    fwd = sum(v for k, v in conv.items() if "forward" in k.lower())
    print(f"{d.name:24s} step {step:.2f} ms  force kernels {total:.2f}  conv fwd {fwd:.2f} ({fwd/step:.1%})  "
          f"conv bwd {bwd:.2f} ({bwd/step:.1%})  rebuild {rebuild:.3f}")
    for k, v in sorted(cats.items(), key=lambda kv: -kv[1])[:5]:
        print(f"      {v:7.3f}  {v/step:5.1%}  {k}")
