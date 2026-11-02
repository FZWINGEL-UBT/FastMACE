#!/bin/bash
# Speed of generic_radial_tables on a workstation (SRC set by the caller), each job through run.sh's idle check:
# 1. whole-system MD step, same process, both build orders (tools/ab_rewrites.py, side b radial_tables=on), NaCl
#    10x10x10 and SrTiO3 11x11x11 (MACE-OMAT-0 medium, hybrid float32, NVT);
# 2. the tiled call, same process: compiled tiles with the budget, without and with the tables (ab_arms.py), SrTiO3
#    58x58x58 (975,560 atoms).
#   HOST=murphy bash ab_ws.sh
set -u
R=/mnt/homes/zwingel/fm-tiled/run.sh
HOST=${HOST:-$(hostname)}
CK=/mnt/homes/zwingel/ResearchModule/Models/mace-omat-0-medium.model
for spec in NaCl@10x10x10 SrTiO3@11x11x11; do
  for order in ab ba; do
    if [ $order = ab ]; then pattern=abab; else pattern=abba; fi
    n=$HOST-grt-ab-${spec%%@*}-${spec##*@}-$order
    bash $R $n tools/ab_rewrites.py --checkpoint $CK --model omat0-medium --structures $spec \
      --b radial_tables=on --order $order --pattern $pattern > /dev/null 2>&1
    echo "$n exit $? $(grep -E 'b/a' /mnt/homes/zwingel/fm-tiled/runs/$n/run.log | tail -1 | cut -c1-200)"
  done
done
n=$HOST-grt-tiled-SrTiO3-58x58x58
bash $R $n results/tiled-speed-20261101/ab_arms.py --structure SrTiO3 --model omat0-medium --repeat 58x58x58 \
  --arms compiled+budget,compiled+budget+tables --calls 1 > /dev/null 2>&1
echo "$n exit $?"
grep '^{"structure' /mnt/homes/zwingel/fm-tiled/runs/$n/run.log | python3 -c "
import json, sys
r = json.loads(sys.stdin.read())
print({k: r.get(k) for k in ('medians', 'ratios', 'gate', 'receivers_chosen', 'builds', 'rewrites')})"
