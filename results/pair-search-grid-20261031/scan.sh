#!/bin/bash
# NbCl5 past 2.1M atoms with the tiled path (T2) and the pair-search grid fix: growing until one fails.
# Source: fastmace/cap-tiled-t1t2 with this branch's pair_search.py (see .source_commit). run.sh's idle check.
set -u
R=/mnt/homes/zwingel/fm-tiled/run.sh
for rep in 22x22x66 23x23x69 24x24x72 25x25x75 26x26x78; do
  bash $R ps-cap-NbCl5-$rep tools/bench/tiled_capacity.py --structure NbCl5 --repeat $rep --receivers 4096 --calls 1 \
    > /dev/null 2>&1
  echo "capacity NbCl5 $rep exit $?"
  grep -q '"phase": "done"' /mnt/homes/zwingel/fm-tiled/runs/ps-cap-NbCl5-$rep/tiled-*.json 2>/dev/null || break
done
