#!/bin/bash
# tiled job 39 (11 November 2026): gapprof.py (where the steady device-state call leaves the GPU idle) at 975,560
# SrTiO3 atoms, R = 16,384; then job 38's forced-drop census again on the TILE_HEAP_TRIM commit.
set -u
RUN=/home/fzwin/fm-runs/tiled-speed-20261108
SRC=$RUN/src-${COMMIT}
OUT=$RUN/${JOB}
mkdir -p $OUT
export PATH=/home/fzwin/environments/mace_cueq_oeq/bin:/usr/lib/wsl/lib:$PATH PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1
export FASTMACE_MODELS=/home/fzwin/models FASTMACE_SOURCE_COMMIT=$COMMIT
unset PYTORCH_CUDA_ALLOC_CONF MALLOC_MMAP_THRESHOLD_ MALLOC_ARENA_MAX
cd $SRC
echo "== gapprof $(date +%T) $(cut -d' ' -f1-3 /proc/loadavg)"
python $RUN/gapprof.py 58x58x58 $OUT/gapprof-58.json > $OUT/gapprof.log 2>&1
echo "exit=$? $(date +%T)"; tail -1 $OUT/gapprof.log | cut -c1-1500
echo "== census $(date +%T) $(free -g | sed -n 2p)"
python $RUN/dropcensus.py --structure SrTiO3 --model omat0-medium --repeat 159x159x159 --mode auto --tile-list \
  --host-state --alias-state --species-index --wavefront on --calls 2 --state-dtype auto --output $OUT/census \
  > $OUT/census.log 2>&1
echo "exit=$? $(date +%T)"
grep -E "forced|before drop|census" $OUT/census.log | cut -c1-400
echo done
