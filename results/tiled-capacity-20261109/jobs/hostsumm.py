"""hostsumm.py JOBDIR NAME...: the host profile and the call records of tiled_host_profile runs, per atom."""
import glob
import json
import sys

job = sys.argv[1]
for name in sys.argv[2:]:
    d = json.load(open(f"{job}/hostprof-{name}/host-profile.json"))
    n = d["atoms"]
    print(f"===== {name} {n} atoms, process peak {d['process_peak'] / n:.0f} B/atom")
    for k, v in list(d["functions"].items())[:18]:
        print(f"  {k:40s} calls {v['calls']:5d} rise {v['peak_rise'] / n:7.0f} net {v['net'] / n:7.0f} "
              f"peak {v['peak'] / n:7.0f}")
    r = json.load(open(glob.glob(f"{job}/hostprof-{name}/tiled-*.json")[0]))
    print("  phases", {k: round(v["host_peak"] / n) for k, v in r["phases"].items()})
    for c in r.get("calls", []):
        wave = c.get("wavefront") or {}
        print(f"  call {c['seconds']:.0f} s {c['mode']} rss {c['host_rss'] / n:.0f} peak {c['host_peak'] / n:.0f} "
              f"state {c.get('host_bytes', 0) / n:.0f} kept {(c.get('list_kept_bytes') or 0) / n:.0f} "
              f"dtype {c.get('state_dtype')} pool {wave.get('pool_fraction')} reuse {c.get('list_reuse')}")
    for lname in ("build", "after call 1"):
        rows = (r.get("ledgers") or {}).get(lname) or []
        print(f"  -- ledger {lname} (host)")
        for row in rows:
            if row["device"] == "cpu" and row["bytes"] / n >= 2:
                print(f"     {row['bytes'] / n:7.1f} B/atom {row['dtype']:14s} {str(row['shape']):22s} "
                      f"x{row['count']} {', '.join(row.get('names') or [])[:120]}")
