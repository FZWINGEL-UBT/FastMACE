#!/bin/bash
# Short NVE drift of MACE-OMAT-0 medium (hybrid float32 default plan), without and with generic_radial_tables,
# each through run.sh's idle check (tools/bench/nve_drift.py: NaCl 300 K, 10 ps at 1 fs).
#   HOST=leto bash nve_ws.sh
set -u
R=/mnt/homes/zwingel/fm-tiled/run.sh
HOST=${HOST:-$(hostname)}
for tables in off on; do
  n=$HOST-grt-nve-NaCl-tables-$tables
  bash $R $n tools/bench/nve_drift.py --model omat0-medium --structure NaCl --ps 10 \
    --rewrites radial_tables=$tables > /dev/null 2>&1
  echo "$n exit $? $(tail -3 /mnt/homes/zwingel/fm-tiled/runs/$n/run.log | cut -c1-400)"
done
