#!/bin/bash
# T3 per-tile neighbour lists on murphy (SRC set by the caller): NbCl5 4x4x8 with and without --tile-list (energy
# and largest force must agree), then the tile-list limits of NbCl5 and NaCl, growing until a size fails or is
# invalid (tiled_capacity.py's edge and force check). run.sh's idle check before each job.
set -u
R=/mnt/homes/zwingel/fm-tiled/run.sh
T=tools/bench/tiled_capacity.py
P=${PREFIX:-t3}
bash $R $P-small-global $T --structure NbCl5 --repeat 4x4x8 --receivers 4096 --rattle 0.05 > /dev/null 2>&1
echo "small global exit $?"
bash $R $P-small-tiles $T --structure NbCl5 --repeat 4x4x8 --receivers 4096 --rattle 0.05 --tile-list > /dev/null 2>&1
echo "small tiles exit $?"
for rep in ${NBCL5:-22x22x66 24x24x72 26x26x78 27x27x81 28x28x84}; do
  bash $R $P-cap-NbCl5-$rep $T --structure NbCl5 --repeat $rep --receivers 4096 --calls 1 --tile-list > /dev/null 2>&1
  echo "capacity NbCl5 $rep exit $?"
  grep -q '"phase": "done"' /mnt/homes/zwingel/fm-tiled/runs/$P-cap-NbCl5-$rep/tiled-*.json 2>/dev/null || break
done
for rep in ${NACL:-60x60x60 66x66x66 72x72x72 76x76x76}; do
  bash $R $P-cap-NaCl-$rep $T --structure NaCl --repeat $rep --receivers 4096 --calls 1 --tile-list > /dev/null 2>&1
  echo "capacity NaCl $rep exit $?"
  grep -q '"phase": "done"' /mnt/homes/zwingel/fm-tiled/runs/$P-cap-NaCl-$rep/tiled-*.json 2>/dev/null || break
done
