#!/usr/bin/env bash
# Capacity scan on the local RTX 5060 Ti 16 GB (WSL2), inside a reservation:
#   python3 tools/ci/local_gpu.py run --purpose "capacity scan ... (<15 min)" -- \
#     env FM=<pinned source copy> bash results/capacity-16gb-20261029/local.sh ROOT JOB...
# Jobs: point:<structure>:<precision>:<NXxNYxNZ> (tools/bench/capacity_scan.py point);
#       scan:<structure>:<precision>[:<start atoms>] (tools/bench/capacity_scan.py scan, 13-minute budget);
#       diag:<structure>:<precision>:<NXxNYxNZ> (point --diagnose with CUDA_LAUNCH_BLOCKING=1, tag -diag).
# Optional environment: TAG (point-file suffix), REWRITES (FIELD=VALUE ..., RewriteConfig overrides) and
# MODEL (a label; default mh1-r2scan), ENGINE (FIELD=VALUE ..., EngineConfig overrides) and TRACE (non-empty:
# point jobs sample device memory over time, --trace).
# Before each job: no WSL compute process and utilisation <= 10 % in both nvidia-smi views for 30 s; the
# Windows view's memory use (the desktop) is logged.
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
  IFS=: read -r kind structure precision extra <<< "$job"
  idle "$job" || exit 3
  note "start $job"
  opts=("--tag=${TAG:-}" --model "${MODEL:-mh1-r2scan}"); [ -n "${REWRITES:-}" ] && opts+=(--rewrites $REWRITES)
  [ -n "${ENGINE:-}" ] && opts+=(--engine $ENGINE)
  [ -n "${TRACE:-}" ] && [ "$kind" = point ] && opts+=(--trace)
  case $kind in
    point) "$PY" "$FM/tools/bench/capacity_scan.py" point --structure "$structure" --precision "$precision" \
             --output "$ROOT/points" --repeat "$extra" "${opts[@]}" >> "$ROOT/jobs.log" 2>&1 < /dev/null ;;
    scan) "$PY" "$FM/tools/bench/capacity_scan.py" scan --structure "$structure" --precision "$precision" \
            --output "$ROOT/points" ${extra:+--start-atoms "$extra"} "${opts[@]}" >> "$ROOT/jobs.log" 2>&1 < /dev/null ;;
    diag) CUDA_LAUNCH_BLOCKING=1 "$PY" "$FM/tools/bench/capacity_scan.py" point --structure "$structure" \
            --precision "$precision" --output "$ROOT/points" --repeat "$extra" --diagnose "--tag=${TAG:--diag}" \
            --model "${MODEL:-mh1-r2scan}" ${REWRITES:+--rewrites $REWRITES} ${ENGINE:+--engine $ENGINE} \
            >> "$ROOT/jobs.log" 2>&1 < /dev/null ;;
    *) note "unknown job $job"; exit 2 ;;
  esac
  note "end $job exit=$?"
done
