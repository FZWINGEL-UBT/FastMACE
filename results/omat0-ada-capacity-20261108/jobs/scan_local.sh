#!/usr/bin/env bash
# The default plan's capacity limit for OMAT-0 medium (float32 NVT) on the local RTX 5060 Ti (WSL2):
# capacity_scan.py scan per structure (20 warm-up and 200 timed steps per size). Run inside a reservation:
#   python3 tools/ci/local_gpu.py run --quiet --purpose "..." -- env FM=<pinned source> bash scan_local.sh ROOT STRUCTURE[:START]...
set -u
ROOT=$1; shift
FM=${FM:?set FM to a pinned source copy}
PY=$HOME/environments/mace_cueq_oeq/bin/python
CACHE=$HOME/fastmace-cache/ada-capacity-local
SITE=$HOME/environments/mace_cueq_oeq/lib/python3.10/site-packages
mkdir -p "$ROOT/tmp" "$CACHE"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$CACHE FASTMACE_MODELS=$HOME/models \
  FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18 TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor \
  TRITON_CACHE_DIR=$CACHE/triton MACE_COMPILE_CACHE=$CACHE/compile XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$ROOT/tmp \
  TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_HYBRID_PYTHON=$PY \
  LD_LIBRARY_PATH=$SITE/nvidia/cu13/lib:$SITE/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
log=$ROOT/run.log
echo "$(date -u +%FT%TZ) local fastmace=$(cat $FM/.source_commit) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)" >> $log
for job in "$@"; do
  structure=${job%%:*} start=${job#*:}
  [ "$start" = "$job" ] && start=4000
  "$PY" "$FM/tools/bench/capacity_scan.py" scan --structure "$structure" --model omat0-medium \
    --precision float32 --output "$ROOT/points" --start-atoms "$start" --growth ${GROWTH:-1.25} --budget-minutes ${BUDGET:-120} \
    > "$ROOT/scan-$structure.log" 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) end $structure exit=$? $(tail -n 2 "$ROOT/scan-$structure.log" | tr '\n' ' ' | cut -c1-300)" >> $log
done
echo "$(date -u +%FT%TZ) DONE" >> $log
