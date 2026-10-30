#!/bin/bash
# Tiled limits with the chunked compact (main + this branch): NbCl5 from 2.3M and NaCl from 1.48M atoms, growing
# until one fails (tiled_capacity.py records the failing phase and its traceback). run.sh's idle check.
set -u
R=/mnt/homes/zwingel/fm-tiled/run.sh  # with SRC= set by the caller for a source copy of its own
for rep in 22x22x66 23x23x69 24x24x72 25x25x75 26x26x78 27x27x81; do
  bash $R ${PREFIX:-cc}-cap-NbCl5-$rep tools/bench/tiled_capacity.py --structure NbCl5 --repeat $rep --receivers 4096 --calls 1 \
    > /dev/null 2>&1
  echo "capacity NbCl5 $rep exit $?"
  grep -q '"phase": "done"' /mnt/homes/zwingel/fm-tiled/runs/${PREFIX:-cc}-cap-NbCl5-$rep/tiled-*.json 2>/dev/null || break
done
for rep in 57x57x57 60x60x60 63x63x63 66x66x66 69x69x69; do
  bash $R ${PREFIX:-cc}-cap-NaCl-$rep tools/bench/tiled_capacity.py --structure NaCl --repeat $rep --receivers 4096 --calls 1 \
    > /dev/null 2>&1
  echo "capacity NaCl $rep exit $?"
  grep -q '"phase": "done"' /mnt/homes/zwingel/fm-tiled/runs/${PREFIX:-cc}-cap-NaCl-$rep/tiled-*.json 2>/dev/null || break
done
