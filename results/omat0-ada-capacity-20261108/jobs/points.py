"""Table of capacity_scan.py points: status, atoms, ms/step, us/atom/step, rule 12's estimate, peak reserved, phase
and error of failures, the plan's memory rungs."""
import glob
import json
import sys

GiB = 2**30
rows = []
for path in glob.glob(f"{sys.argv[1]}/point-*.json"):
    r = json.load(open(path))
    rows.append(r)
rows.sort(key=lambda r: (r.get("structure", ""), r.get("natoms", 0)))
for r in rows:
    ph = r.get("phases") or {}
    peak = max((p.get("max_reserved", 0) or 0) for p in ph.values() if isinstance(p, dict)) if ph else 0
    peak = max(peak, ((r.get("failure") or {}).get("memory") or {}).get("max_reserved", 0) or 0)
    plan = r.get("plan") or {}
    est = plan.get("predicted_memory_bytes")
    rungs = [row[0].split(".")[-1] for row in plan.get("rule_12", []) if len(row) > 1 and row[1] == "on"] or "-"
    memory = (plan.get("memory") or ["", "", ""])[-1]
    ms = r.get("ms_per_step")
    us = 1000 * ms / r["natoms"] if ms else None
    fail = r.get("failure") or r.get("error") or {}
    if isinstance(fail, dict):
        fail = f"{fail.get('phase', '')} {str(fail.get('error', fail.get('message', '')))[:110]}"
    print(f"{r.get('structure')} {r.get('repeat')} {r.get('natoms')} {r.get('status')} "
          f"ms={ms and round(ms, 1)} us/atom={us and round(us, 2)} peak={peak / GiB:.2f}GiB "
          f"est={est and round(est / GiB, 2)} est/peak={est and peak and round(est / peak, 3)} rungs={rungs} "
          f"{fail if r.get('status') != 'ok' else ''}")
    if len(sys.argv) > 2:
        print("   ", memory[:400])
