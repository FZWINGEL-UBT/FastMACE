#!/usr/bin/env bash
# OPT-T10: step breakdown of the plain default plan (no switches), local RTX 5060 Ti, from a pinned copy. Per structure: the main pass (wall time, stages, eager kernels, force-graph replays), then an ncu process
# (one eager force call) and an nsys process (force-graph replays, every node traced), as 2-T20a's run.sh.
# A monitor samples Windows nvidia-smi (WSL cannot see Windows apps), loadavg and free memory every 5 s.
# Usage: FM=<pinned copy> bash t09.sh ROOT STRUCTURE...
set -u -o pipefail
ROOT=$1; shift
FM=${FM:?pinned copy}
B=$HOME/fastmace-baseline/1a-T18
CACHE=$HOME/fastmace-cache/opt-T10
WSMI=/mnt/c/Windows/System32/nvidia-smi.exe
mkdir -p "$ROOT/tmp" "$CACHE"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$CACHE FASTMACE_MODELS=$HOME/models \
  FASTMACE_STRUCTURES=$B TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton \
  MACE_COMPILE_CACHE=$CACHE/compile XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp \
  TORCHINDUCTOR_COMPILE_THREADS=4
NCU=/usr/local/cuda/bin/ncu
NSYS=/usr/local/cuda/bin/nsys
PY=$HOME/environments/mace_cueq_oeq/bin/python
TOOL=$FM/tools/step_breakdown.py
log=$ROOT/run.log
monitor() {
  while true; do
    printf '%s win=%s wsl=%s load=%s avail_mb=%s\n' "$(date -u +%FT%TZ)" \
      "$($WSMI --query-gpu=utilization.gpu,memory.used,clocks.sm,temperature.gpu --format=csv,noheader | tr -d ' \r')" \
      "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)" \
      "$(cut -d' ' -f1-3 /proc/loadavg | tr ' ' ,)" "$(free -m | awk '/^Mem:/{print $7}')" >> "$ROOT/monitor.log"
    sleep 5
  done
}
idle() {  # 6 samples 5 s apart, all idle (both views), else note and continue after 2 min
  local name=$1 ok=0 tries=0 pids util wutil
  while [ $ok -lt 6 ] && [ $tries -lt 24 ]; do
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    wutil=$($WSMI --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r')
    if [ -z "$pids" ] && [ "$util" -le 5 ] && [ "$wutil" -le 5 ]; then ok=$((ok + 1)); else ok=0; fi
    tries=$((tries + 1)); sleep 5
  done
  echo "$(date -u +%FT%TZ) $name idle_ok=$ok/6 pids=[$pids] util=$util win=$wutil load=$(cut -d' ' -f1-3 /proc/loadavg)" >> "$log"
}
step() {
  local label=$1; shift
  echo "$(date -u +%FT%TZ) start $label" >> "$log"
  "$@" > "$ROOT/$label.log" 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) end $label exit=$?" >> "$log"
}
monitor & MON=$!
trap 'kill $MON 2>/dev/null' EXIT
echo "$(date -u +%FT%TZ) fastmace=$(cat $FM/.source_commit) source=$FM gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)" >> "$log"
for s in "$@"; do
  dir=$ROOT/$s
  ARGS=(--checkpoint "$HOME/models/mace-mh-1.model" --model mh1-r2scan --backend hybrid --temperature-K 1000
        --structure "$s")
  idle "$s-main"; step "$s-main" "$PY" "$TOOL" --output "$dir" "${ARGS[@]}"
  idle "$s-ncu"
  step "$s-ncu" "$NCU" --profile-from-start off --target-processes all --cache-control none --clock-control none \
    --metrics "$("$PY" "$TOOL" --ncu-metrics)" --export "$dir/ncu" --force-overwrite \
    "$PY" "$TOOL" --pass ncu --output "$dir" "${ARGS[@]}"
  step "$s-ncu-csv" bash -c "'$NCU' --import '$dir/ncu.ncu-rep' --csv --page raw --print-units base > '$dir/ncu_raw.csv'"
  step "$s-ncu-join" "$PY" "$TOOL" --output "$dir" --summarize-ncu "$dir/ncu_raw.csv"
  idle "$s-nsys"
  step "$s-nsys" "$NSYS" profile -t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi \
    --capture-range-end=stop -o "$dir/nsys" --force-overwrite true "$PY" "$TOOL" --pass nsys --output "$dir" "${ARGS[@]}"
  step "$s-nsys-csv" "$NSYS" stats --report cuda_gpu_trace --format csv --output "$dir/nsys_stats" \
    --force-export true "$dir/nsys.nsys-rep"
  step "$s-nsys-gaps" "$PY" "$TOOL" --output "$dir" --summarize-nsys "$dir/nsys_stats_cuda_gpu_trace.csv" "${ARGS[@]}"
done
echo "$(date -u +%FT%TZ) done" >> "$log"
