#!/bin/bash
# Tiled capacity scan, MACE-OMAT-0 medium on cubic SrTiO3 (perfect crystal: the exact edge check applies), on the
# machine it runs on: tiled_capacity.py --mode auto (the first call eager, measuring its memory; the second compiled
# when it fits), tile lists, the host state aliased, the species index, the default budget.
# Usage: SRC=... HOST=leto bash scan.sh REPEAT [REPEAT ...]; stops at the first failed or invalid size.
# Host guard before each size: the last size's host peak per atom (start: BPA, default 4000 bytes) times 1.15, and
# 20 GB kept free (FACTOR overrides the 1.15, e.g. 1.0 at the limit).
set -u
R=/mnt/homes/zwingel/fm-tiled/run.sh
HOST=${HOST:-$(hostname)}
bpa=${BPA:-4000}
for rep in "$@"; do
  atoms=$(echo $rep | awk -F x '{printf "%.0f", $1 * $2 * $3 * 5}')
  avail=$(free -b | awk 'NR==2 {print $7}')
  need=$(awk -v n=$atoms -v b=$bpa -v f=${FACTOR:-1.15} 'BEGIN {printf "%.0f", n * b * f}')
  echo "$rep: $atoms atoms; host available $((avail / 1000000000)) GB, needs about $((need / 1000000000)) GB"
  if [ $need -gt $((avail - 20000000000)) ]; then echo "$rep skipped: host memory"; break; fi
  n=$HOST-omat-scan-SrTiO3-$rep
  bash $R $n tools/bench/tiled_capacity.py --structure SrTiO3 --model omat0-medium --repeat $rep --mode auto \
    --receivers 4096 --calls 2 --tile-list --host-state --alias-state --species-index > /dev/null 2>&1
  code=$?
  out=$(python3 - /mnt/homes/zwingel/fm-tiled/runs/$n <<'EOF'
import glob, json, sys
files = glob.glob(sys.argv[1] + "/tiled-*.json")
if not files:
    print("NONE 0")
    sys.exit()
r = json.load(open(files[0]))
n = r["atoms"]
peaks = [v.get("host_peak") or 0 for v in list((r.get("phases") or {}).values()) + list(r.get("calls") or [])]
host = max(peaks) if peaks else 0
print(r.get("phase"), round(host / n) if host else 0)
for name, v in (r.get("phases") or {}).items():
    print(f"  phase {name}: gpu peak above {v.get('peak_above_before', 0) / 2**30:.2f} GiB, allocated "
          f"{v.get('allocated', 0) / 2**30:.2f} GiB, host rss {(v.get('host_rss') or 0) / 2**30:.1f} GiB, host peak "
          f"{(v.get('host_peak') or 0) / 2**30:.1f} GiB, {v.get('seconds', 0):.0f} s")
for c in r.get("calls") or []:
    print(f"  call {c['mode']} R {c['receivers']} tiles {c['tiles']}: {c['seconds']:.0f} s, gpu total peak "
          f"{c['total_peak'] / 2**30:.2f} GiB, host peak {(c.get('host_peak') or 0) / 2**30:.1f} GiB, "
          f"reuse {c.get('list_reuse')}, kept {c.get('list_kept_bytes')}")
print("  valid", r.get("valid"), r.get("check"), "error", str(r.get("error"))[:300])
EOF
)
  echo "$rep exit $code"
  echo "$out" | tail -n +2
  set -- $(echo "$out" | head -1)
  [ "$code" = 0 ] && [ "$1" = done ] || { echo "stop at $rep"; break; }
  [ "${2:-0}" -gt 0 ] && bpa=$2
done
