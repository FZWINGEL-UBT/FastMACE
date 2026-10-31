#!/usr/bin/env bash
# recompute_node_region's capacity limit points on the local RTX 5060 Ti (16 GB), as CAP-T08's: capacity_scan.py point,
# default plan with every memory-rescue record forced on (recompute_conv_weights, recompute_node_segments,
# recompute_node_region), 200 steps, the point's edge check. Holding the local GPU reservation:
#   python3 tools/ci/local_gpu.py run --purpose "fm node_region limits" -- env FM=<source> bash points.sh ROOT CELL...
# CELL: Structure:NXxNYxNZ, e.g. NbCl5:6x6x28.
set -u -o pipefail
ROOT=$1; shift
FM=${FM:?set FM to a source copy}
CACHE=$HOME/fastmace-cache/chunked-region-local
WSMI=/mnt/c/Windows/System32/nvidia-smi.exe
mkdir -p "$ROOT/tmp" "$ROOT/points" "$CACHE"
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
  structure=${cell%%:*} repeat=${cell#*:}
  idle || exit 3
  echo "$(date -u +%FT%TZ) start $cell" >> "$log"
  "$PY" "$FM/tools/bench/capacity_scan.py" point --structure "$structure" --model mh1-r2scan --precision float32 \
    --repeat "$repeat" --output "$ROOT/points" --tag region \
    --rewrites recompute_conv_weights=on recompute_node_segments=on recompute_node_region=on \
    > "$ROOT/point-$structure-$repeat.log" 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) end $cell exit=$? $(tail -n 2 "$ROOT/point-$structure-$repeat.log" | tr '\n' ' ' | cut -c1-240)" >> "$log"
done
