#!/usr/bin/env bash
# Atom-chunked combined region (ninth list item 2) on the local RTX 5060 Ti: region_probe.py per layout, each its own
# process after an idle check, NbCl5 5x5x12 (21,600 atoms). Holding the local GPU reservation:
#   python3 tools/ci/local_gpu.py run --purpose "fm chunked region" -- env FM=<source> bash local.sh ROOT LAYOUT...
# LAYOUT: separate, or combined-<chunks>.
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
for layout in "$@"; do
  region=${layout%%-*}; chunks=1
  [ "$region" = combined ] && chunks=${layout#combined-}
  idle || exit 3
  echo "$(date -u +%FT%TZ) start $layout" >> "$log"
  "$PY" "$FM/results/chunked-region-20261031/region_probe.py" --checkpoint "$HOME/models/mace-mh-1.model" \
    --structure "${STRUCTURE:-NbCl5@5x5x12}" --region "$region" --chunks "$chunks" --output "$ROOT" \
    > "$ROOT/$layout.log" 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) end $layout exit=$? $(grep 'peak above' "$ROOT/$layout.log")" >> "$log"
done
