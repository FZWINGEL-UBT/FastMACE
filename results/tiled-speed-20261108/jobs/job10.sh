#!/bin/bash
# tiled gap job 10: device time by call site (kprof2.py) of a steady tiled call at 975,560 atoms
set -u
RUN=/home/fzwin/fm-runs/tiled-speed-20261108
SRC=$RUN/src-${COMMIT}
OUT=$RUN/${JOB}
mkdir -p $OUT
export PATH=/home/fzwin/environments/mace_cueq_oeq/bin:/usr/lib/wsl/lib:$PATH PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1
export FASTMACE_MODELS=/home/fzwin/models FASTMACE_SOURCE_COMMIT=$COMMIT
cd $SRC
cp $RUN/kprof2.py $OUT/
echo "== kprof2 $(date +%T)"
python $OUT/kprof2.py 58x58x58 $OUT/kprof2-58.json > $OUT/kprof2.log 2>&1
echo "exit=$? $(date +%T)"; tail -1 $OUT/kprof2.log | cut -c1-300
echo done
