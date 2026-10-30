#!/usr/bin/env bash
# CAP-T08 inventory (the conductor's check 1, 30 October 2026) on the local RTX 5060 Ti (WSL2), in a reservation:
#   python3 tools/ci/local_gpu.py run --purpose "fm CAP-T08 inventory (<15 min)" -- \
#     env FM=<pinned source copy> bash results/cap-t08-inventory-20261030/local.sh ROOT JOB...
# Jobs: saved:<structure>:<NXxNYxNZ>:<mode>[:<rewrites>] (tools/bench/saved_tensors.py, float32; mode default or
# eager = compile_mode=none; rewrites FIELD=VALUE[,FIELD=VALUE]); kernels:<structure>:<NXxNYxNZ> (the conductor's
# check 2: tools/bench/kernel_resources.py --smem-spilling, the plan's NVRTC kernels' local memory);
# wrebuild:<structure>:<NXxNYxNZ> (tools/bench/wrebuild_bound.py, the w-rebuild falsifier's cost bound).
# Before each job: no WSL compute process and utilisation <= 10 % in both nvidia-smi views for 30 s.
# Template: results/capacity-16gb-20261029/local.sh.
set -u -o pipefail
ROOT=$1; shift
FM=${FM:?set FM to a pinned source copy}
B=$HOME/fastmace-baseline/1a-T18
PY=$HOME/environments/mace_cueq_oeq/bin/python
CACHE=$HOME/fastmace-cache/capacity-local
WSMI=/mnt/c/Windows/System32/nvidia-smi.exe
mkdir -p "$ROOT/tmp" "$CACHE"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$CACHE FASTMACE_MODELS=$HOME/models \
  FASTMACE_STRUCTURES=$B TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton \
  MACE_COMPILE_CACHE=$CACHE/compile XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp \
  TORCHINDUCTOR_COMPILE_THREADS=4
log=$ROOT/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
idle() {  # 6 samples, 5 s apart, all idle
  local name=$1 ok=0 tries=0 pids util wutil
  while [ $ok -lt 6 ]; do
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    wutil=$($WSMI --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r')
    if [ -z "$pids" ] && [ "$util" -le 10 ] && [ "$wutil" -le 10 ]; then ok=$((ok + 1)); else ok=0; fi
    tries=$((tries + 1))
    if [ $tries -ge 60 ]; then note "$name NOT IDLE pids=[$pids] util=$util win=$wutil"; return 1; fi
    sleep 5
  done
  note "$name idle: util=$util win=$wutil mem=$($WSMI --query-gpu=memory.used,memory.total --format=csv,noheader | tr -d '\r') load=$(cut -d' ' -f1-3 /proc/loadavg) avail_mb=$(free -m | awk '/^Mem:/{print $7}')"
}
note "fastmace=$(cat "$FM/.source_commit") source=$FM host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
for job in "$@"; do
  IFS=: read -r kind structure repeat mode rewrites <<< "$job"
  idle "$job" || exit 3
  note "start $job"
  opts=(); [ "$mode" = eager ] && opts+=(--engine compile_mode=none)
  [ -n "${rewrites:-}" ] && opts+=(--rewrites ${rewrites//,/ })
  case $kind in
    saved) "$PY" "$FM/tools/bench/saved_tensors.py" --structure "$structure" --repeat "$repeat" \
             --output "$ROOT/saved" "${opts[@]}" > "$ROOT/saved-$structure-$repeat-$mode${rewrites:+-$rewrites}.log" 2>&1 < /dev/null ;;
    wrebuild) "$PY" "$FM/tools/bench/wrebuild_bound.py" --structure "$structure" --repeat "$repeat"                 --output "$ROOT/wrebuild" > "$ROOT/wrebuild-$structure-$repeat.log" 2>&1 < /dev/null ;;
    kernels) "$PY" "$FM/tools/bench/kernel_resources.py" --structure "$structure" --repeat "$repeat" \
               --output "$ROOT/kernels" --smem-spilling > "$ROOT/kernels-$structure-$repeat.log" 2>&1 < /dev/null ;;
    *) note "unknown job $job"; exit 2 ;;
  esac
  note "end $job exit=$?"
done
note done
