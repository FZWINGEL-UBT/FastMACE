#!/bin/bash
# The tiled call's whole-system stages after slimming (SRC set by the caller), on leto: the stage peaks at 22.4M
# NbCl5 atoms, then the two-call (MD) limit, growing until a size fails; host memory checked before each size
# (about 3.6 kB per atom with the tile-list cache; 20 GB kept free).
set -u
R=/mnt/homes/zwingel/fm-tiled/run.sh
bash $R leto-stagepeaks2-NbCl5-47x47x141 results/tiled-call-memory-20261101/stage_peaks.py --repeat 47x47x141 \
  > /dev/null 2>&1
echo "stage peaks exit $?"
for rep in ${NBCL5:-68x68x204 69x69x207}; do
  atoms=$(echo $rep | awk -F x '{printf "%.0f", $1 * $2 * $3 * 72}')
  avail=$(free -b | awk 'NR==2 {print $7}')
  need=$(awk -v n=$atoms 'BEGIN {printf "%.0f", n * 3450}')
  echo "host available $((avail / 1000000000)) GB, $rep needs about $((need / 1000000000)) GB"
  if [ $need -gt $((avail - 20000000000)) ]; then echo "$rep skipped: host memory"; break; fi
  bash $R leto-limit2-NbCl5-$rep tools/bench/tiled_capacity.py --structure NbCl5 --repeat $rep --mode eager \
    --receivers 4096 --calls 2 --tile-list --host-state --alias-state --species-index > /dev/null 2>&1
  echo "two calls $rep exit $?"
  grep -q '"phase": "done"' /mnt/homes/zwingel/fm-tiled/runs/leto-limit2-NbCl5-$rep/tiled-*.json 2>/dev/null || break
done
