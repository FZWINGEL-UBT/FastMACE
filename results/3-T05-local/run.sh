#!/bin/bash
# 3-T05 local GPU A/Bs: fixed skin (0.3 A, 1.15) against rule 11's, MH-1 hybrid float32 NVT, tools/ab_skin.py.
# One structure and one build order per local_gpu.py reservation (at most 15 min each). Run it as a background
# task of the session (no setsid, no --quiet), with the pinned copy's local_gpu.py.
# Usage (WSL): bash results/3-T05-local/run.sh <pinned source dir> <work dir> [target_steps]
set -u
SRC=$1; W=$2; TARGET=${3:-}
mkdir -p $W/tmp $W/ab
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$SRC FASTMACE_MODELS=$HOME/models FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18
export TMPDIR=$W/tmp TORCHINDUCTOR_COMPILE_THREADS=4 MACE_COMPILE_CACHE=$HOME/compile-cache/3-t05 TORCHINDUCTOR_CACHE_DIR=$HOME/compile-cache/3-t05-inductor
P=$HOME/environments/mace_cueq_oeq/bin/python
GPU="python3 $SRC/tools/ci/local_gpu.py run --wait-minutes 240 --min-available-gib 6"
log() { echo "$(date -u +%FT%TZ) $*" >> $W/ab.log; }
extra=(); [ -n "$TARGET" ] && extra=(--target-steps $TARGET)
job() {  # name temperature timestep structure order pattern
  log "start $1 gpu=$(nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader)"
  $GPU --purpose "3-T05 $1 (<15 min)" -- nice -n 0 $P -W ignore $SRC/tools/ab_skin.py --output $W/ab/$1 \
    --checkpoint $HOME/models/mace-mh-1.model --structures $4 --temperature-K $2 --timestep-fs $3 --order $5 \
    --pattern $6 --warmup 1500 "${extra[@]}" > $W/ab/$1.out 2> $W/ab/$1.err
  log "end $1 exit $? $(tail -1 $W/ab/$1.out)"
}
log "source $(cat $SRC/.source_commit) host $(hostname) target=${TARGET:-default}"
for case in "1500 1.0" "1000 1.0"; do
  set -- $case
  for structure in NaCl NbCl5; do
    job skin-${1}K-${2}fs-$structure-ab $1 $2 $structure ab abab
    job skin-${1}K-${2}fs-$structure-ba $1 $2 $structure ba abba
  done
done
log "all done"
