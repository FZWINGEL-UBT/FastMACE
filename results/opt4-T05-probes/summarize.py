"""OPT4-T05: stage tables from setup_md.py's setup.json files (medians, min-max), and the final-state comparison.

    python summarize.py RUN_ROOT [RUN_ROOT ...] > summary.md

A run is named <set>-<model>-<structure>[-<variant>]; sets cold-N, warm-N (fill excluded), ab-N, cold-dyn-N.
Stages (seconds, from the timers' totals, so nothing is counted twice):

- OEq runtime: ``prepare_hybrid`` (the extension build when cold);
- load: ``load_pristine`` without the OEq runtime (checkpoint hash, torch.load, e3nn model);
- conversion: the plan's execution model (template copy or reload and conversion to the backend);
- self-checks: every ``Registry.selfcheck``; rewrites, rest: ``Registry.apply`` without them;
- first compiled call; captures: every graph capture, setup and run; steps: ``Simulation.run`` without its
  captures; rest: the process wall time minus all of these (imports, plan, private copies, output).
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

STAGES = ("OEq runtime", "load", "conversion", "self-checks", "rewrites, rest", "first compiled call", "captures",
          "of which B's first capture", "steps", "rest", "process")


def total(stages, name):
    return stages.get(name, {}).get("total_s", 0.0)


def split(record):
    s = record["stages"]
    oeq = total(s, "load: prepare_hybrid (OEq runtime)")
    checks = sum(v["total_s"] for k, v in s.items() if k.startswith("self-check: "))
    captures = sum(v["total_s"] for k, v in s.items() if k.startswith("capture: ") and k != "capture: prepare (setup)"
                   and not k.startswith("capture: eager"))
    in_setup = total(s, "capture: prepare (setup)")
    first_b = s.get("capture: B", {}).get("each_s", [0.0])[0]
    out = {"OEq runtime": oeq, "load": total(s, "load: load_pristine") - oeq,
           "conversion": total(s, "build: execution model (template / conversion)"), "self-checks": checks,
           "rewrites, rest": total(s, "rewrites: apply") - checks,
           "first compiled call": total(s, "compile: first compiled call"), "captures": captures,
           "of which B's first capture": first_b,
           "steps": total(s, "run: Simulation.run") - (captures - in_setup),
           "process": record["since_process_start_s"]}
    parts = [k for k in STAGES if k not in ("rest", "process", "of which B's first capture")]
    out["rest"] = out["process"] - sum(out[k] for k in parts)
    return out


def cell_of(name):
    parts = name.split("-")
    # cold-2-mh1-NaCl, ab-1-mh1-NaCl-on1, cold-dyn-1-omat0-NaCl
    if parts[0] == "cold" and parts[1] == "dyn":
        return "cold-dyn", f"{parts[3]} {parts[4]}"
    if parts[0] == "ab":
        return f"ab-{parts[4].rstrip('0123456789')}", f"{parts[2]} {parts[3]}"
    return parts[0], f"{parts[2]} {parts[3]}"


def fmt(values):
    if not values:
        return "–"
    med = statistics.median(values)
    return f"{med:.1f}" if len(values) == 1 else f"{med:.1f} ({min(values):.1f}–{max(values):.1f})"


def main(roots):
    groups = defaultdict(lambda: defaultdict(list))  # (set kind, cell) -> stage -> values
    states = {}
    for root in roots:
        for path in sorted(Path(root).glob("*/setup.json")):
            name = path.parent.name
            if name.startswith(("fill", "dyn-fill", "diag")):
                continue
            record = json.loads(path.read_text())
            kind, cell = cell_of(name)
            for stage, value in split(record).items():
                groups[(kind, cell)][stage].append(value)
            groups[(kind, cell)]["n"].append(1)
            state = path.parent / "final_state.npz"
            if state.exists():
                states[name] = dict(np.load(state))
    kinds = sorted({kind for kind, _ in groups})
    cells = sorted({cell for _, cell in groups})
    for kind in kinds:
        present = [cell for cell in cells if (kind, cell) in groups]
        print(f"\n### {kind}\n")
        print("| Stage | " + " | ".join(f"{c} (n = {len(groups[(kind, c)]['n'])})" for c in present) + " |")
        print("| --- | " + " | ".join("---:" for _ in present) + " |")
        for stage in STAGES:
            print(f"| {stage} | " + " | ".join(fmt(groups[(kind, c)][stage]) for c in present) + " |")
    if states:
        print("\n### Final states (ab sets): max |difference| against the first off run of the cell\n")
        print("| Run | step | positions Å | velocities | forces eV/Å | energy eV |")
        print("| --- | ---: | ---: | ---: | ---: | ---: |")
        for name in sorted(states):
            if not name.startswith("ab-"):
                continue
            reference = states[name.rsplit("-", 1)[0] + "-off1"]
            row = states[name]
            diffs = [float(np.max(np.abs(row[k] - reference[k]))) for k in ("positions", "velocities", "forces",
                                                                            "energy")]
            print(f"| {name} | {int(row['step'])} | " + " | ".join(f"{d:.3g}" for d in diffs) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
