#!/bin/bash
# tiled job 38: host census around a forced drop of the kept lists (idea 1, 11 November 2026): SrTiO3 159^3
# (20,099,295 atoms), the A100 record's settings (wavefront, host state, alias, species index, state "auto"), two
# calls; default allocator, glibc defaults (no MALLOC_ environment).
set -u
RUN=/home/fzwin/fm-runs/tiled-speed-20261108
SRC=$RUN/src-${COMMIT}
OUT=$RUN/${JOB}
mkdir -p $OUT
export PATH=/home/fzwin/environments/mace_cueq_oeq/bin:/usr/lib/wsl/lib:$PATH PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1
export FASTMACE_MODELS=/home/fzwin/models FASTMACE_SOURCE_COMMIT=$COMMIT
unset PYTORCH_CUDA_ALLOC_CONF MALLOC_MMAP_THRESHOLD_ MALLOC_ARENA_MAX
cd $SRC
echo "== start $(date +%T) $(free -g | sed -n 2p)"
python $RUN/dropcensus.py --structure SrTiO3 --model omat0-medium --repeat 159x159x159 --mode auto --tile-list \
  --host-state --alias-state --species-index --wavefront on --calls 2 --state-dtype auto --output $OUT \
  > $OUT/run.log 2>&1
echo "exit=$? $(date +%T)"
grep -E "forced|before drop|census|list_reuse|state_dtype" $OUT/run.log | cut -c1-400
echo done
