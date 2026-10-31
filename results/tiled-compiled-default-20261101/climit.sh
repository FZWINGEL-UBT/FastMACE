#!/bin/bash
# Compiled residency and limit on leto (SRC set by the caller).
set -u
R=/mnt/homes/zwingel/fm-tiled/run.sh
T=tools/bench/tiled_capacity.py
A="--receivers 4096 --calls 2 --tile-list --host-state --alias-state --species-index"
for mode in eager compiled; do
  bash $R leto-resid-$mode-NbCl5-35x35x105 $T --structure NbCl5 --repeat 35x35x105 --mode $mode $A > /dev/null 2>&1
  echo "9.26M $mode exit $?"
done
for rep in 68x68x204 67x67x201; do
  atoms=$(echo $rep | awk -F x '{printf "%.0f", $1 * $2 * $3 * 72}')
  avail=$(free -b | awk 'NR==2 {print $7}')
  need=$(awk -v n=$atoms 'BEGIN {printf "%.0f", n * 3600}')
  echo "host available $((avail / 1000000000)) GB, $rep needs about $((need / 1000000000)) GB"
  if [ $need -gt $((avail - 20000000000)) ]; then echo "$rep skipped: host memory"; break; fi
  bash $R leto-climit-NbCl5-$rep $T --structure NbCl5 --repeat $rep --mode compiled $A > /dev/null 2>&1
  echo "compiled $rep exit $?"
  grep -q '"phase": "done"' /mnt/homes/zwingel/fm-tiled/runs/leto-climit-NbCl5-$rep/tiled-*.json 2>/dev/null && break
done
