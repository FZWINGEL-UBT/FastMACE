#!/bin/bash
set -u
RUN=/home/fzwin/fm-runs/tiled-speed-20261108
SRC=$RUN/src-${COMMIT}
OUT=$RUN/${JOB}
mkdir -p $OUT
export PATH=/home/fzwin/environments/mace_cueq_oeq/bin:$PATH PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1
export FASTMACE_MODELS=/home/fzwin/models FASTMACE_SOURCE_COMMIT=$COMMIT
cd $SRC
cp $RUN/kprof.py $OUT/
for spec in "ws 18x18x18" "tiled 18x18x18" "tiled 58x58x58"; do
  set -- $spec
  echo "== $1 $2 $(date +%T)"; python $OUT/kprof.py $1 $2 $OUT/kprof-$1-$2.json > $OUT/kprof-$1-$2.log 2>&1; echo "exit=$? $(date +%T)"; tail -1 $OUT/kprof-$1-$2.log
done
echo done
