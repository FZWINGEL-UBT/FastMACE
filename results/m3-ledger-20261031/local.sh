#!/usr/bin/env bash
# M3: where the memory sits near the 16 GB limit with the memory-rescue records on (plans/capacity-ideas-20261030.md).
# tools/bench/memory_ledger.py per cell, each its own process after an idle check, local RTX 5060 Ti. Holding the
# local GPU reservation:
#   python3 tools/ci/local_gpu.py run --purpose "fm M3 ledger" -- env FM=<source> bash local.sh ROOT CELL...
# CELL: Structure:NXxNYxNZ:records, records "region" (conv_weights, node_segments, node_region on) or "segments"
# (node_region off).
set -u -o pipefail
ROOT=$1; shift
FM=${FM:?set FM to a source copy}
CACHE=$HOME/fastmace-cache/chunked-region-local
WSMI=/mnt/c/Windows/System32/nvidia-smi.exe
mkdir -p "$ROOT/tmp" "$CACHE"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_MODELS=$HOME/models FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18 \
  TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_CACHE=$CACHE \
  TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton MACE_COMPILE_CACHE=$CACHE/compile \
  XDG_CACHE_HOME=$CACHE/xdg FASTMACE_HYBRID_PYTHON=$HOME/environments/mace_cueq_oeq/bin/python
PY=$HOME/environments/mace_cueq_oeq/bin/python
log=$ROOT/run.log
idle() {
  local ok=0 tries=0 pids util wutil
  while [ $ok -lt 6 ]; do
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    wutil=$($WSMI --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r')
    if [ -z "$pids" ] && [ "$util" -le 10 ] && [ "$wutil" -le 10 ]; then ok=$((ok + 1)); else ok=0; fi
    tries=$((tries + 1)); [ $tries -ge 60 ] && { echo "$(date -u +%FT%TZ) NOT IDLE" >> "$log"; return 1; }
    sleep 5
  done
}
echo "$(date -u +%FT%TZ) fastmace=$(cat "$FM/.source_commit") gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)" >> "$log"
for cell in "$@"; do
  IFS=: read -r structure repeat records <<< "$cell"
  region=on; [ "$records" = segments ] && region=off
  idle || exit 3
  echo "$(date -u +%FT%TZ) start $cell" >> "$log"
  "$PY" "$FM/tools/bench/memory_ledger.py" --structure "$structure" --repeat "$repeat" --output "$ROOT" \
    --tag "-$records" --rewrites recompute_conv_weights=on recompute_node_segments=on recompute_node_region=$region \
    > "$ROOT/ledger-$structure-$repeat-$records.log" 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) end $cell exit=$?" >> "$log"
done
