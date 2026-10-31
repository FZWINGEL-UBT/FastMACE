#!/bin/bash
# The build without whole-system intermediates on murphy (SRC set by the caller): E0 by species index, the
# batch's edges never moved to the device, the allocator's cache released before the first tile. Host-state
# tile-list limits from 9.26M atoms up, growing until a size fails or is invalid (tiled_capacity.py's edge and
# force check). run.sh's idle check before each job; host memory checked before each size.
set -u
R=/mnt/homes/zwingel/fm-tiled/run.sh
T=tools/bench/tiled_capacity.py
P=${PREFIX:-build}
for rep in ${NBCL5:-35x35x105 37x37x111 39x39x117 41x41x123 43x43x129}; do
  free -g | awk 'NR==2 {print "host memory free " $7 " GB"}'
  bash $R $P-cap-NbCl5-$rep $T --structure NbCl5 --repeat $rep --receivers 4096 --calls 1 --tile-list --host-state \
    > /dev/null 2>&1
  echo "capacity NbCl5 $rep exit $?"
  grep -q '"phase": "done"' /mnt/homes/zwingel/fm-tiled/runs/$P-cap-NbCl5-$rep/tiled-*.json 2>/dev/null || break
done
for rep in ${NACL:-105x105x105 111x111x111 117x117x117 123x123x123 129x129x129}; do
  free -g | awk 'NR==2 {print "host memory free " $7 " GB"}'
  bash $R $P-cap-NaCl-$rep $T --structure NaCl --repeat $rep --receivers 4096 --calls 1 --tile-list --host-state \
    > /dev/null 2>&1
  echo "capacity NaCl $rep exit $?"
  grep -q '"phase": "done"' /mnt/homes/zwingel/fm-tiled/runs/$P-cap-NaCl-$rep/tiled-*.json 2>/dev/null || break
done
