#!/bin/bash
# TileConfig.budget_receivers on murphy (SRC set by the caller): the compiled call at NbCl5 51,840 atoms with
# budget_receivers (8192, 16384) against the fixed R = 4,096 in one process; then eager tile-list capacity
# points with the budget (two calls: the second at the chosen cap), below the limit and at it. run.sh's idle
# check before each job.
set -u
R=/mnt/homes/zwingel/fm-tiled/run.sh
bash $R budget-screen-NbCl5-6x6x20 tools/bench/tiled_graphs.py screen --structure NbCl5 --repeat 6x6x20 \
  --grid 4096:0.15,4096:0.15:8192+16384 > /dev/null 2>&1
echo "screen exit $?"
for rep in 22x22x66 25x25x75; do
  bash $R budget-cap-NbCl5-$rep tools/bench/tiled_capacity.py --structure NbCl5 --repeat $rep --receivers 4096 \
    --calls 2 --tile-list --budget-receivers 8192 16384 > /dev/null 2>&1
  echo "capacity NbCl5 $rep exit $?"
done
