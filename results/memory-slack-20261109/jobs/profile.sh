#!/usr/bin/env bash
# tools/bench/memory_profile.py on one or more cases: profile.sh ROOT CASE:REPEAT...  (CASE md or single)
# Env: EXTRA (more memory_profile.py arguments), TAG (log suffix); FM (source copy with .source_commit), PY, CACHE; optional SITE for the cu13 / cuEq libraries.
set -u
ROOT=$1; shift
FM=${FM:?set FM to a pinned source copy}
PY=${PY:-$HOME/environments/mace_cueq_oeq/bin/python}
CACHE=${CACHE:-$HOME/fastmace-cache/memory-profile}
SITE=${SITE:-$(dirname "$PY")/../lib/python3.10/site-packages}
mkdir -p "$ROOT/tmp" "$CACHE"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$CACHE FASTMACE_MODELS=${FASTMACE_MODELS:-$HOME/models} \
  FASTMACE_STRUCTURES=${FASTMACE_STRUCTURES:-$HOME/fastmace-baseline/1a-T18} TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor \
  TRITON_CACHE_DIR=$CACHE/triton MACE_COMPILE_CACHE=$CACHE/compile XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$ROOT/tmp \
  TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=${THREADS:-4} FASTMACE_HYBRID_PYTHON=$PY \
  LD_LIBRARY_PATH=$SITE/nvidia/cu13/lib:$SITE/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
log=$ROOT/run.log
echo "$(date -u +%FT%TZ) $(hostname) fastmace=$(cat $FM/.source_commit) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)" >> $log
for job in "$@"; do
  case=${job%%:*} repeat=${job#*:}
  [ -n "${WAIT_IDLE:-}" ] && $WAIT_IDLE
  # the GPU's processes every 30 s: a case that ran beside someone else's job is void
  (while sleep 30; do echo "$(date -u +%T) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | paste -sd' ' -)"; done)     > "$ROOT/$case-$repeat${TAG:-}.gpu" 2>&1 &
  monitor=$!
  "$PY" "$FM/tools/bench/memory_profile.py" --case "$case" --structure ${STRUCTURE:-SrTiO3} --repeat "$repeat" \
    --output "$ROOT/profiles" ${EXTRA:-} > "$ROOT/$case-$repeat${TAG:-}.log" 2>&1 < /dev/null
  status=$?
  kill $monitor 2>/dev/null
  echo "$(date -u +%FT%TZ) end $case $repeat${TAG:-} exit=$status $(grep -m1 "atoms:" "$ROOT/$case-$repeat${TAG:-}.log" | cut -c1-300)" >> $log
done
echo "$(date -u +%FT%TZ) DONE" >> $log
