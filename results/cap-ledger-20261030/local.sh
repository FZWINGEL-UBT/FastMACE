#!/usr/bin/env bash
# Ideas 4 and 5 of 30 October 2026 (memory ledger at the limit; the first padded edge size) on the local RTX 5060 Ti:
#   python3 tools/ci/local_gpu.py run --purpose "fm cap-ledger JOB" -- \
#     env FM=<pinned branch copy> [TAG=...] [POINT_ARGS=...] bash results/cap-ledger-20261030/local.sh ROOT JOB...
# Jobs:
#   ledger-<structure>-<repeat>: tools/bench/memory_ledger.py (20 warm-up and 20 steady steps);
#   point-<structure>-<repeat>[-trace]: tools/bench/capacity_scan.py point (--trace: memory samples and FastMACE's
#     log lines with times, to see which stage fails); extra capacity_scan arguments in $POINT_ARGS.
# Before each job: no WSL compute process and utilisation <= 10 % in both nvidia-smi views for 30 s.
# Template: results/cap-nlist-20261030/local.sh.
set -u -o pipefail
ROOT=$1; shift
FM=${FM:?set FM to a pinned source copy}
B=$HOME/fastmace-baseline/1a-T18
PY=$HOME/environments/mace_cueq_oeq/bin/python
CACHE=$HOME/fastmace-cache/cap-ledger-local
WSMI=/mnt/c/Windows/System32/nvidia-smi.exe
mkdir -p "$ROOT/tmp" "$CACHE"
export PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$CACHE FASTMACE_MODELS=$HOME/models \
  FASTMACE_STRUCTURES=$B TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton \
  MACE_COMPILE_CACHE=$CACHE/compile XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp \
  TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_HYBRID_PYTHON=$PY
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
  note "$name idle: util=$util win=$wutil mem=$($WSMI --query-gpu=memory.used --format=csv,noheader | tr -d '\r') load=$(cut -d' ' -f1-3 /proc/loadavg) avail_mb=$(free -m | awk '/^Mem:/{print $7}')"
}
run() {  # name command...
  local name=$1; shift
  idle "$name" || exit 3
  note "start $name"
  "$@" > "$ROOT/$name.log" 2>&1 < /dev/null
  note "end $name exit=$?"
}
note "fastmace=$(cat "$FM/.source_commit") host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
for job in "$@"; do
  case $job in
    ledger-*) IFS=- read -r _ structure repeat <<< "$job"
      PYTHONPATH=$FM run "$job" "$PY" "$FM/tools/bench/memory_ledger.py" --structure "$structure" --repeat "$repeat" \
          --output "$ROOT/ledger" ;;
    point-*) IFS=- read -r _ structure repeat trace <<< "$job"
      # shellcheck disable=SC2086
      PYTHONPATH=$FM run "$job" "$PY" "$FM/tools/bench/capacity_scan.py" point --structure "$structure" \
          --model mh1-r2scan --precision float32 --repeat "$repeat" --output "$ROOT/points" \
          ${trace:+--trace} --tag="${TAG:-}${trace:+-trace}" ${POINT_ARGS:-} ;;
    *) note "unknown job $job"; exit 2 ;;
  esac
done
note done
