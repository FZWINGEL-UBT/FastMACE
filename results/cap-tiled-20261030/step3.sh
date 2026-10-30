#!/bin/bash
# Tiled capacity path, step 3 on murphy (RTX 4000 Ada 20 GB). Each job goes through run.sh (idle check first).
#   capacity  the largest NbCl5 and NaCl cells of the tiled force call (R=4096), growing until one fails;
#   main      current main's default MD plan at sizes the tiled path also runs (capacity_scan.py point: step time
#             and peak), and main's own limit on this GPU (capacity_scan.py scan);
#   md        short NVE checks with neighbour rebuilds, tiled against the whole-system call, and tiled only at
#             a large cell.
set -u
R=/mnt/homes/zwingel/fm-tiled/run.sh
phase=${1:-all}
if [ "$phase" = capacity ] || [ "$phase" = all ]; then
  for rep in 15x15x45 16x16x48 17x17x51 18x18x54; do
    bash $R cap-NbCl5-$rep tools/bench/tiled_capacity.py --structure NbCl5 --repeat $rep --receivers 4096 \
      --calls 1 > /dev/null 2>&1
    echo "capacity NbCl5 $rep exit $?"
    grep -q '"phase": "done"' /mnt/homes/zwingel/fm-tiled/runs/cap-NbCl5-$rep/tiled-*.json || break
  done
  for rep in 42x42x42 46x46x46 50x50x50 54x54x54; do
    bash $R cap-NaCl-$rep tools/bench/tiled_capacity.py --structure NaCl --repeat $rep --receivers 4096 \
      --calls 1 > /dev/null 2>&1
    echo "capacity NaCl $rep exit $?"
    grep -q '"phase": "done"' /mnt/homes/zwingel/fm-tiled/runs/cap-NaCl-$rep/tiled-*.json || break
  done
fi
if [ "$phase" = main ] || [ "$phase" = all ]; then
  for spec in "NbCl5 5x5x12" "NbCl5 6x6x20" "NaCl 14x14x14" "NaCl 18x18x18"; do
    set -- $spec
    bash $R main-$1-$2 tools/bench/capacity_scan.py point --structure $1 --repeat $2 --steps 50 > /dev/null 2>&1
    echo "main point $1 $2 exit $?"
  done
  for s in NbCl5 NaCl; do
    bash $R main-scan-$s tools/bench/capacity_scan.py scan --structure $s --start-atoms 50000 --growth 1.15 \
      --budget-minutes 40 > /dev/null 2>&1
    echo "main scan $s exit $?"
  done
fi
if [ "$phase" = md ] || [ "$phase" = all ]; then
  bash $R md-NbCl5-4x4x8 tools/bench/tiled_md.py --structure NbCl5 --repeat 4x4x8 --path both --steps 300 > /dev/null 2>&1
  echo "md NbCl5 exit $?"
  bash $R md-NaCl-12x12x12 tools/bench/tiled_md.py --structure NaCl --repeat 12x12x12 --path both --steps 300 > /dev/null 2>&1
  echo "md NaCl exit $?"
  bash $R md-NbCl5-12x12x36 tools/bench/tiled_md.py --structure NbCl5 --repeat 12x12x36 --path tiled --steps 40 \
    > /dev/null 2>&1
  echo "md NbCl5 large exit $?"
fi
