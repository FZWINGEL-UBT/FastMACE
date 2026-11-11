#!/usr/bin/env bash
# Capacity points on the local RTX 5060 Ti (WSL2), per structure in the given order, stopping a structure at its first
# failure (capacity_scan.py point, 40-minute limit). Run inside a reservation, with an idle check before each point:
#   python3 tools/ci/local_gpu.py run --purpose "..." -- env FM=<pinned source> bash points_local.sh ROOT STRUCTURE:REPEAT,...
set -u
ROOT=$1; shift
FM=${FM:?set FM to a pinned source copy}
PY=$HOME/environments/mace_cueq_oeq/bin/python
CACHE=$HOME/fastmace-cache/ada-capacity-local
SITE=$HOME/environments/mace_cueq_oeq/lib/python3.10/site-packages
CK=$HOME/models/mace-omat-0-medium.model
WSMI=/mnt/c/Windows/System32/nvidia-smi.exe
mkdir -p "$ROOT/tmp" "$CACHE"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$CACHE FASTMACE_MODELS=$HOME/models \
  FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18 TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor \
  TRITON_CACHE_DIR=$CACHE/triton MACE_COMPILE_CACHE=$CACHE/compile XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$ROOT/tmp \
  TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_HYBRID_PYTHON=$PY \
  LD_LIBRARY_PATH=$SITE/nvidia/cu13/lib:$SITE/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
log=$ROOT/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
idle() {  # 6 samples, 5 s apart, all idle
  local ok=0 tries=0 pids util wutil
  while [ $ok -lt 6 ]; do
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    wutil=$($WSMI --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r')
    if [ -z "$pids" ] && [ "$util" -le 10 ] && [ "$wutil" -le 10 ]; then ok=$((ok + 1)); else ok=0; fi
    tries=$((tries + 1))
    if [ $tries -ge 60 ]; then note "NOT IDLE pids=[$pids] util=$util win=$wutil"; return 1; fi
    sleep 5
  done
}
note "points local fastmace=$(cat $FM/.source_commit) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
cd "$FM"
for spec in "$@"; do
  structure=${spec%%:*}
  for repeat in $(echo "${spec#*:}" | tr ',' ' '); do
    idle || exit 1
    timeout 2400 "$PY" tools/bench/capacity_scan.py point --structure "$structure" --model omat0-medium       --precision float32 --output "$ROOT" --repeat "$repeat" > "$ROOT/point-$structure-$repeat.log" 2>&1 < /dev/null
    note "end $structure $repeat exit=$? $(grep -m1 '^point ' "$ROOT/point-$structure-$repeat.log" | cut -c1-300)"
    grep -q "^point .*): ok," "$ROOT/point-$structure-$repeat.log" || break
  done
done
note "POINTS DONE"
