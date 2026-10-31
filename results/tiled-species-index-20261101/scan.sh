#!/bin/bash
# The tiled path with the species index instead of the one-hot (TileConfig.species="index"; SRC set by the
# caller): NbCl5 4x4x8 with and without it (energy, forces), then host-state tile-list capacity points, growing
# until a size fails or is invalid (tiled_capacity.py's edge and force check). run.sh's idle check before each
# job. Host memory: a point needs about 6.7 kB per atom on the host (the tiled state 4.86 kB, MACE's host batch
# with its own neighbour list about 1.6 kB); a size whose need exceeds the available memory less 20 GB is not
# started (a shared workstation must never run out of memory).
set -u
R=/mnt/homes/zwingel/fm-tiled/run.sh
T=tools/bench/tiled_capacity.py
P=${HOST:-$(hostname)}-species
if [ -n "${SMALL:-}" ]; then
  bash $R $P-small-onehot $T --structure NbCl5 --repeat 4x4x8 --receivers 4096 --rattle 0.05 --tile-list --host-state \
    > /dev/null 2>&1
  echo "small onehot exit $?"
  bash $R $P-small-index $T --structure NbCl5 --repeat 4x4x8 --receivers 4096 --rattle 0.05 --tile-list --host-state \
    --species-index > /dev/null 2>&1
  echo "small index exit $?"
fi
for spec in ${NBCL5:+$(for r in $NBCL5; do echo NbCl5:$r; done)} ${NACL:+$(for r in $NACL; do echo NaCl:$r; done)}; do
  s=${spec%%:*}; rep=${spec#*:}
  atoms=$(echo $rep | awk -F x -v s=$s '{print $1 * $2 * $3 * (s == "NaCl" ? 8 : 72)}')
  avail=$(free -b | awk 'NR==2 {print $7}')
  need=$(awk -v n=$atoms 'BEGIN {printf "%.0f", n * 6700}')  # %d overflows at 2^31 in mawk
  echo "host memory available $((avail / 1000000000)) GB, $s $rep needs about $((need / 1000000000)) GB"
  if [ $need -gt $((avail - 20000000000)) ]; then echo "capacity $s $rep skipped: host memory"; break; fi
  bash $R $P-cap-$s-$rep $T --structure $s --repeat $rep --receivers 4096 --calls 1 --tile-list --host-state \
    --species-index > /dev/null 2>&1
  echo "capacity $s $rep exit $?"
  grep -q '"phase": "done"' /mnt/homes/zwingel/fm-tiled/runs/$P-cap-$s-$rep/tiled-*.json 2>/dev/null || break
done
