#!/bin/bash
# One device copy of the one-hot species during the construction (SRC set by the caller): host-state tile-list
# limits, NbCl5 from 25.4M atoms up (NBCL5) or one NaCl confirmation point (NACL), growing until a size fails or
# is invalid (tiled_capacity.py's edge and force check). run.sh's idle check before each job; host memory logged
# before each size. HOST labels the runs.
set -u
R=/mnt/homes/zwingel/fm-tiled/run.sh
T=tools/bench/tiled_capacity.py
P=${HOST:-$(hostname)}-onehot
for spec in ${NBCL5:+$(for r in $NBCL5; do echo NbCl5:$r; done)} ${NACL:+$(for r in $NACL; do echo NaCl:$r; done)}; do
  s=${spec%%:*}; rep=${spec#*:}
  free -g | awk 'NR==2 {print "host memory free " $7 " GB"}'
  bash $R $P-cap-$s-$rep $T --structure $s --repeat $rep --receivers 4096 --calls 1 --tile-list --host-state \
    > /dev/null 2>&1
  echo "capacity $s $rep exit $?"
  grep -q '"phase": "done"' /mnt/homes/zwingel/fm-tiled/runs/$P-cap-$s-$rep/tiled-*.json 2>/dev/null || break
done
