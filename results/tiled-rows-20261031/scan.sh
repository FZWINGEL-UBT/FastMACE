#!/bin/bash
# Species tables at the call's row count (MACEForceField call_rows; tiled_capacity.py builds with
# call_rows=--receivers) on murphy (SRC set by the caller): NbCl5 4x4x8 with the global list and with tile lists
# (energy and largest force against T3's records), then the tile-list limits past T3's build limit, growing until
# a size fails or is invalid. run.sh's idle check before each job.
set -u
R=/mnt/homes/zwingel/fm-tiled/run.sh
T=tools/bench/tiled_capacity.py
P=${PREFIX:-rows}
bash $R $P-small-global $T --structure NbCl5 --repeat 4x4x8 --receivers 4096 --rattle 0.05 > /dev/null 2>&1
echo "small global exit $?"
bash $R $P-small-tiles $T --structure NbCl5 --repeat 4x4x8 --receivers 4096 --rattle 0.05 --tile-list > /dev/null 2>&1
echo "small tiles exit $?"
for rep in ${NBCL5:-25x25x72 25x25x75 26x26x78}; do
  bash $R $P-cap-NbCl5-$rep $T --structure NbCl5 --repeat $rep --receivers 4096 --calls 1 --tile-list > /dev/null 2>&1
  echo "capacity NbCl5 $rep exit $?"
  grep -q '"phase": "done"' /mnt/homes/zwingel/fm-tiled/runs/$P-cap-NbCl5-$rep/tiled-*.json 2>/dev/null || break
done
for rep in ${NACL:-74x74x74 75x75x75 76x76x76}; do
  bash $R $P-cap-NaCl-$rep $T --structure NaCl --repeat $rep --receivers 4096 --calls 1 --tile-list > /dev/null 2>&1
  echo "capacity NaCl $rep exit $?"
  grep -q '"phase": "done"' /mnt/homes/zwingel/fm-tiled/runs/$P-cap-NaCl-$rep/tiled-*.json 2>/dev/null || break
done
