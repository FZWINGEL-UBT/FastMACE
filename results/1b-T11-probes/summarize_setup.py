"""Table of the 1b-T11 setup probes (setup-*/setup.json): setup and its stages per run and structure."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
GROUPS = {
    "self-checks": lambda s: s.startswith("selfcheck:"),
    "rewrites (apply)": lambda s: s.startswith("apply:") or s.startswith("rewrites:"),
    "model copy": lambda s: "deepcopy" in s or "private_copy" in s,
    "first compiled call": lambda s: s.startswith("build: initial forces"),
}
for path in sorted(root.glob("setup-*/setup.json")):
    run = json.loads(path.read_text())
    for structure, entry in run["structures"].items():
        stages = entry["setup_stages"]
        parts = {g: sum(v["total_s"] for k, v in stages.items() if f(k)) for g, f in GROUPS.items()}
        captures = sum(v["total_s"] for k, v in entry["warmup_stages"].items() if "capture" in k)
        load = run["process"]["stages"].get("load (checkpoint, conversion)", 0)
        print(f"{path.parent.name:34s} {structure:6s} load {load:5.1f} "
              f"setup {entry['setup_s']:6.2f} | " + " ".join(f"{g} {v:6.2f}" for g, v in parts.items())
              + f" | warm-up {entry['warmup_s']:6.2f} (captures {captures:6.2f}) step {entry['step_ms']:.3f}")
