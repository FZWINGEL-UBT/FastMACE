#!/bin/bash
# TileConfig.state='host' on murphy (SRC set by the caller): NbCl5 4x4x8 tile lists with the state on the device
# and on the host (energy, forces), then the host-state tile-list limits, growing until a size fails or is
# invalid. run.sh's idle check before each job.
set -u
R=/mnt/homes/zwingel/fm-tiled/run.sh
T=tools/bench/tiled_capacity.py
P=${PREFIX:-host}
bash $R $P-small-device $T --structure NbCl5 --repeat 4x4x8 --receivers 4096 --rattle 0.05 --tile-list > /dev/null 2>&1
echo "small device exit $?"
bash $R $P-small-host $T --structure NbCl5 --repeat 4x4x8 --receivers 4096 --rattle 0.05 --tile-list --host-state \
  > /dev/null 2>&1
echo "small host exit $?"
for rep in ${NBCL5:-26x26x78 30x30x90 33x33x99 35x35x105}; do
  bash $R $P-cap-NbCl5-$rep $T --structure NbCl5 --repeat $rep --receivers 4096 --calls 1 --tile-list --host-state \
    > /dev/null 2>&1
  echo "capacity NbCl5 $rep exit $?"
  grep -q '"phase": "done"' /mnt/homes/zwingel/fm-tiled/runs/$P-cap-NbCl5-$rep/tiled-*.json 2>/dev/null || break
done
for rep in ${NACL:-80x80x80 90x90x90 99x99x99 105x105x105}; do
  bash $R $P-cap-NaCl-$rep $T --structure NaCl --repeat $rep --receivers 4096 --calls 1 --tile-list --host-state \
    > /dev/null 2>&1
  echo "capacity NaCl $rep exit $?"
  grep -q '"phase": "done"' /mnt/homes/zwingel/fm-tiled/runs/$P-cap-NaCl-$rep/tiled-*.json 2>/dev/null || break
done
