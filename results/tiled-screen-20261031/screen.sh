#!/bin/bash
# Ninth idea list, item 3 on murphy (SRC set by the caller): the compiled tiled call over R x bucket at NbCl5
# 21,600 and 51,840 atoms, each against R = 4,096 / bucket 0.15 in one process. run.sh's idle check before each job.
set -u
R=/mnt/homes/zwingel/fm-tiled/run.sh
for rep in 5x5x12 6x6x20; do
  bash $R screen-NbCl5-$rep tools/bench/tiled_graphs.py screen --structure NbCl5 --repeat $rep > /dev/null 2>&1
  echo "screen NbCl5 $rep exit $?"
done
