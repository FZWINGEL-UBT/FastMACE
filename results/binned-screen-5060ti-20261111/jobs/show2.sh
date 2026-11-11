#!/bin/bash
python3 - <<'PY'
import json, glob, os
for out in ("out", "out2"):
    for f in sorted(glob.glob(os.path.expanduser(f"~/fm-runs/binned-5060-20261111/{out}/ab-*.json"))):
        d = json.load(open(f))
        s = d["structures"]["SrTiO3"] if isinstance(d["structures"], dict) else d["structures"][0]
        ab = s["ab"]["sides"]
        c = s["engine_counters_timed"]
        print(out, os.path.basename(f), "atoms", s["natoms"], "b/a %.4f" % s["ratio_b_over_a"], "valid", s["valid"])
        for side in ("a", "b"):
            print("   ", side, "median %.3f ms" % ab[side]["median_ms"], "spread %.3f" % ab[side]["spread"],
                  "blocks", [round(x, 2) for x in ab[side]["block_ms"]], "rebuilds", c[side]["rebuilds"])
        print("    pair ratios", [round(x, 4) for x in s["pair_ratios"]])
PY
mkdir -p /mnt/d/Margraf/FastMACE/results/binned-screen-5060ti-20261111/records
R=$HOME/fm-runs/binned-5060-20261111
for o in out out2; do mkdir -p /mnt/d/Margraf/FastMACE/results/binned-screen-5060ti-20261111/records/$o; cp $R/$o/*.json $R/$o/*.log $R/$o/peak-*.txt /mnt/d/Margraf/FastMACE/results/binned-screen-5060ti-20261111/records/$o/ 2>/dev/null; done
ls /mnt/d/Margraf/FastMACE/results/binned-screen-5060ti-20261111/records/out /mnt/d/Margraf/FastMACE/results/binned-screen-5060ti-20261111/records/out2 | head -40
python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py status
