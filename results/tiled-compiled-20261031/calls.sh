#!/bin/bash
# Compiled tile bodies on murphy: the falsifier (largest tile, compiled against uncompiled) and whole calls in
# modes padded and compiled on the compiled force field, at NbCl5 21,600 and 51,840 atoms. run.sh's idle check.
set -u
R=/mnt/homes/zwingel/fm-tiled/run.sh
T=tools/bench/tiled_graphs.py
bash $R tc2-fals-5x5x12 $T compiled --repeat 5x5x12 --receivers 4096 --iters 16 > /dev/null 2>&1
echo "falsifier exit $?"
for rep in 5x5x12 6x6x20; do
  bash $R tc2-calls-$rep $T call --repeat $rep --receivers 4096 --modes padded compiled --calls 4 > /dev/null 2>&1
  echo "calls $rep exit $?"
done
