"""censsumm.py JOBDIR NAME...: censuses of tiled_host_profile runs (HOSTPROF_CENSUS=1), per atom, and the call times."""
import glob
import json
import sys

job = sys.argv[1]
for name in sys.argv[2:]:
    d = json.load(open(f"{job}/hostprof-{name}/host-profile.json"))
    n = d["atoms"]
    r = json.load(open(glob.glob(f"{job}/hostprof-{name}/tiled-*.json")[0]))
    calls = " ".join(f"{c['mode']} {c['seconds']:.1f}s peak {c['host_peak'] / n:.0f}" for c in r.get("calls", []))
    print(f"===== {name}: process peak {d['process_peak'] / n:.0f} B/atom; {calls}")
    for tag, lst in (d.get("censuses") or {}).items():
        for c in lst:
            st = {k: round(v / n) for k, v in c["status"].items()}
            pin = (c.get("pinned") or {}).get("allocated_bytes.current", 0) / n
            print(f"  {tag:5s} rss {c['rss'] / n:5.0f} trimmed {c['rss_trimmed'] / n:5.0f} arrays {c['arrays'] / n:4.0f} "
                  f"pinned {pin:4.0f} {st} unaccounted anon {st.get('RssAnon', 0) - c['arrays'] / n - pin:4.0f}")
            for k, cnt, b in c["groups"][:8]:
                if b / n >= 2:
                    print(f"      {b / n:6.1f} B/atom x{cnt} {k}")
