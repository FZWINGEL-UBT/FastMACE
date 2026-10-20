"""2-T19: the per-stage setup tables from setup_stages.py's setup.json files (one run root).

    python summarize.py RUN_ROOT [--json OUT]

Groups processes by name prefix (warm-, cold-, ext4-, detail-, sim-gate-, sim-skip-, mace-first-gate-; the
-fill processes only fill the cache and are left out) and side, and prints the median (min-max) per stage.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Stage groups: (row label, predicate on the timer name). Totals are separate rows; parts never overlap.
PARTS = (
    ("execution model (reload, hash, conversion)", lambda s: s == "build: execution model (reload, hash, conversion)"),
    ("rewrites: apply", lambda s: s.startswith("apply: ")),
    ("rewrites: self-checks", lambda s: s.startswith("self-check: ")),
    ("private copy", lambda s: s == "compile: private_copy"),
    ("first compiled call", lambda s: s == "build: initial forces (first compiled call)"),
    ("gate: references", lambda s: s == "gate: references"),
    ("gate: production", lambda s: s == "gate: production"),
    ("plan", lambda s: s in ("plan: FixedPlanner.plan", "plan: MinimalPlanner.plan (total)")),
)
LOAD_PARTS = (
    ("load: sha256", lambda s: s == "load: sha256_of"),
    ("load: MACE calculator (torch.load, conversion)", lambda s: s.startswith("load: MACECalculator")),
    ("load: OEq/cuEq checks", lambda s: s in ("load: check_oeq", "load: check_cueq", "load: require_fused_cueq")),
)


def stage_sum(stages, predicate):
    return sum(v["total_s"] for k, v in stages.items() if predicate(k))


def rows_of(data, structure):
    entry = data["structures"].get(structure)
    if entry is None:
        return None
    stages = entry["setup_stages"]
    row = {"load (process)": data["process"]["stages"]["load"],
           "prepare_hybrid (process)": data["process"]["stages"].get("prepare_hybrid", 0.0)}
    for label, predicate in LOAD_PARTS:
        row[f"{label} (process)"] = stage_sum(data["process"].get("load_stages", {}), predicate)
    row["setup"] = entry["setup_s"]
    for label, predicate in PARTS:
        row[label] = stage_sum(stages, predicate)
    row["setup, rest"] = row["setup"] - sum(row[label] for label, _ in PARTS)
    row["load + setup"] = row["load (process)"] + row["setup"] if structure_is_first(data, structure) else row["setup"]
    row["warm-up"] = entry["warmup_s"]
    row["warm-up: captures"] = stage_sum(entry["warmup_stages"], lambda s: s.startswith("warm-up: capture"))
    if "since_process_start_s" in entry:
        row["process start to built"] = entry["since_process_start_s"]
    for key, value in entry.get("simulation_setup", {}).items():
        row[f"Simulation._setup_s[{key}]"] = value
    return row


def structure_is_first(data, structure):
    return next(iter(data["structures"])) == structure


def group_of(name):
    match = re.match(r"(warm|cold|ext4|detail|sim-gate|sim-skip|mace-first-gate)-(\d+)(?:-(1a|1b))?$", name)
    if not match:
        return None
    return match.group(1), match.group(3) or "1b"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--json", type=Path)
    args = p.parse_args(argv)
    groups = defaultdict(lambda: defaultdict(list))  # (group, side, structure) -> row label -> values
    for path in sorted(args.root.glob("*/setup.json")):
        name = path.parent.name
        key = group_of(name)
        if key is None:
            continue
        data = json.loads(path.read_text())
        for structure in data["structures"]:
            row = rows_of(data, structure)
            for label, value in row.items():
                groups[key + (structure,)][label].append(value)
    out = {}
    for key in sorted(groups):
        label = " / ".join(key)
        values = groups[key]
        print(f"\n## {label} (n={len(values['setup'])})")
        out[label] = {}
        for row, series in values.items():
            median = statistics.median(series)
            out[label][row] = dict(median=median, min=min(series), max=max(series), n=len(series))
            print(f"  {row:52s} {median:8.2f}  ({min(series):.2f}-{max(series):.2f})")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
