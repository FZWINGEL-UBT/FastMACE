#!/bin/bash
# The aliased host state (SRC set by the caller): host-state (aliased), species-index tile-list capacity
# points, growing until a size fails or is invalid (tiled_capacity.py's edge and force check). run.sh's idle check
# before each job. Host memory: a point peaks at about 3.45 kB per atom on the host (measured at 9.26M NbCl5
# atoms: the call, whose aliased state is 2.5 kB; the build peaks at 3.3 kB), 3.3 kB with a margin here (3.6 kB in the first run, which skipped 71.0M; the call measured 3.14 kB); a size
# whose need exceeds the available memory less 20 GB is not started (a shared workstation must never run out
# of memory). HOST labels the runs.
set -u
R=/mnt/homes/zwingel/fm-tiled/run.sh
T=tools/bench/tiled_capacity.py
P=${HOST:-$(hostname)}-alias
for spec in ${NBCL5:+$(for r in $NBCL5; do echo NbCl5:$r; done)} ${NACL:+$(for r in $NACL; do echo NaCl:$r; done)}; do
  s=${spec%%:*}; rep=${spec#*:}
  atoms=$(echo $rep | awk -F x -v s=$s '{printf "%.0f", $1 * $2 * $3 * (s == "NaCl" ? 8 : 72)}')
  avail=$(free -b | awk 'NR==2 {print $7}')
  need=$(awk -v n=$atoms 'BEGIN {printf "%.0f", n * 3300}')  # %.0f: mawk's %d stops at 2^31
  echo "host memory available $((avail / 1000000000)) GB, $s $rep ($atoms atoms) needs about $((need / 1000000000)) GB"
  if [ $need -gt $((avail - 20000000000)) ]; then echo "capacity $s $rep skipped: host memory"; break; fi
  bash $R $P-cap-$s-$rep $T --structure $s --repeat $rep --receivers 4096 --calls 1 --tile-list --host-state \
    --alias-state --species-index > /dev/null 2>&1
  echo "capacity $s $rep exit $?"
  grep -q '"phase": "done"' /mnt/homes/zwingel/fm-tiled/runs/$P-cap-$s-$rep/tiled-*.json 2>/dev/null || break
done
