#!/bin/bash
# tiled job 42 (11 November 2026): kprof7.py, every kernel charged to its launch site, at 975,560 SrTiO3 atoms.
set -u
RUN=/home/fzwin/fm-runs/tiled-speed-20261108
SRC=$RUN/src-${COMMIT}
OUT=$RUN/${JOB}
mkdir -p $OUT
export PATH=/home/fzwin/environments/mace_cueq_oeq/bin:/usr/lib/wsl/lib:$PATH PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1
export FASTMACE_MODELS=/home/fzwin/models FASTMACE_SOURCE_COMMIT=$COMMIT
unset PYTORCH_CUDA_ALLOC_CONF
cd $SRC
echo "== kprof7 $(date +%T) $(cut -d' ' -f1-3 /proc/loadavg)"
python $RUN/kprof7.py 58x58x58 $OUT/kprof7-58.json > $OUT/kprof7.log 2>&1
echo "exit=$? $(date +%T)"; tail -1 $OUT/kprof7.log | cut -c1-1500
echo done
