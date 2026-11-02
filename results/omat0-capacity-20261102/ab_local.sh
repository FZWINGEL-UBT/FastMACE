#!/usr/bin/env bash
# ab_ws.sh on the local RTX 5060 Ti (WSL2), inside a reservation, with local.sh's idle wait before every A/B:
#   python3 tools/ci/local_gpu.py run --purpose "..." -- env FM=<pinned source copy> \
#     bash results/omat0-capacity-20261102/ab_local.sh ROOT "FIELD=VALUE ..." STRUCTURE_SPEC...
set -u
ROOT=$1 SIDE_B=$2; shift 2
FM=${FM:?set FM to a pinned source copy}
PY=$HOME/environments/mace_cueq_oeq/bin/python
CACHE=$HOME/fastmace-cache/capacity-local
CK=$HOME/models/mace-omat-0-medium.model
WSMI=/mnt/c/Windows/System32/nvidia-smi.exe
mkdir -p "$ROOT/tmp" "$CACHE"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$CACHE FASTMACE_MODELS=$HOME/models \
  FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18 TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor \
  TRITON_CACHE_DIR=$CACHE/triton MACE_COMPILE_CACHE=$CACHE/compile XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$ROOT/tmp \
  TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4
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
  note "idle: mem=$($WSMI --query-gpu=memory.used,memory.total --format=csv,noheader | tr -d '\r')"
}
note "fastmace=$(cat "$FM/.source_commit") gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) b=[$SIDE_B]"
for spec in "$@"; do
  for order in ab ba; do
    if [ "$order" = ab ]; then pattern=abab; else pattern=abba; fi
    idle || exit 3
    name=$(basename "${spec%%@*}" .cif)-${spec##*@}-$order
    "$PY" "$FM/tools/ab_rewrites.py" --output "$ROOT/$name" --checkpoint "$CK" --model omat0-medium \
      --structures "$spec" --b $SIDE_B --order "$order" --pattern "$pattern" > "$ROOT/$name.log" 2>&1 < /dev/null
    note "end $name exit=$? $(grep -E 'b/a' "$ROOT/$name.log" | tail -n 1 | cut -c1-200)"
  done
done
note DONE
