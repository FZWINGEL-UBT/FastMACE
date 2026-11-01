"""The probe ladder's points: per cell the rungs rule 12 planned, the probe (stages, replica, measured and corrected,
cost), the plan's estimate against the measured peak (largest reserved over build, warm-up and steady), ms/step.
    python read_ladder.py POINTS_DIR
"""

import json
import sys
from pathlib import Path

G = 2**30
RUNGS = ("recompute_conv_weights", "recompute_first_table", "recompute_node_segments", "gate_epilogue",
         "recompute_node_region")
rows = []
for f in sorted(Path(sys.argv[1]).glob("point-*.json"), key=lambda f: (f.name.split("-")[1], json.loads(
        f.read_text())["natoms"])):
    d = json.loads(f.read_text())
    plan = d.get("plan") or {}
    peak = max(v.get("max_reserved", 0) for v in d["phases"].values() if isinstance(v, dict))
    rungs = {name: value for name, value, *_ in plan.get("rule_12") or [] if name.startswith("RewriteConfig.")}
    probe = plan.get("memory_probe")
    ladder = plan.get("memory_ladder")
    estimate = plan.get("predicted_memory_bytes") or 0
    print(f"{d['structure']} {d['natoms']}: {d['status']}, {d.get('ms_per_step') or 0:.0f} ms/step, estimate "
          f"{estimate / G:.2f} GiB, peak {peak / G:.2f} GiB, ratio {estimate / peak if peak else float('nan'):.3f}, "
          f"build {d['times'].get('build_s', 0):.0f} s")
    print("   rungs:", {k.split(".")[1]: v for k, v in rungs.items()})
    if ladder:
        print(f"   ladder: free {ladder['free_bytes'] / G:.2f} GiB, chosen {ladder['chosen']['stage'] or 'default'} "
              f"{ladder['chosen']['estimate'] / G:.2f}, cheaper "
              + (f"{ladder['cheaper']['stage'] or 'default'} {ladder['cheaper']['estimate'] / G:.2f}"
                 if ladder["cheaper"] else "none"))
    if probe:
        print(f"   probe: {probe['before'] or 'default'} -> {probe['chosen'] or 'default'}, "
              f"changed {probe['changed_plan']}, {probe['setup_s']:.0f} s")
        for s in probe["stages"]:
            print(f"     {s['stage'] or 'default'}: replica {s['replica_atoms']} estimate {s['replica_estimate'] / G:.2f}"
                  f" reserved {s['replica_reserved'] / G:.2f} allocated {s['replica_allocated'] / G:.2f} scale "
                  f"{s['scale']:.3f}; cell {s['estimate'] / G:.2f} -> {s['corrected'] / G:.2f} GiB, {s['seconds']:.0f} s")
    else:
        print("   probe: none")
    rows.append(dict(structure=d["structure"], atoms=d["natoms"], status=d["status"], ms=d.get("ms_per_step"),
                     estimate=estimate, peak=peak, probe=probe, rungs=rungs, build_s=d["times"].get("build_s")))
Path("ladder.json").write_text(json.dumps(rows, indent=1))
