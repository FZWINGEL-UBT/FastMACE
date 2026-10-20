#!/usr/bin/env bash
# 2-T20a: kernel timing breakdown of one MD step of FastMACE, MH-1 hybrid FP32 production defaults,
# NaCl and NbCl5 3x3x3, local RTX 5060 Ti (WSL2). Launch detached, holding the local GPU reservation:
#   setsid -f python3 tools/ci/local_gpu.py run --purpose "2-T20a step breakdown" -- \
#     bash results/2-T20a-step-breakdown-20261020/run.sh smoke|breakdown ROOT >/dev/null 2>&1
#   (nsys ROOT STRUCTURE repeats one structure's nsys pass of an earlier breakdown, e.g. after a busy GPU)
# smoke: NaCl, 20 + 50 + 50 steps, 2 calls (checks the tool and the profilers). breakdown: NaCl, then NbCl5,
# each one fresh process with 300 equilibration steps, 1,000 steps per window and 10 profiled force calls,
# as the research engine's 15 October probe; then per crystal an ncu process (one eager force call: FLOPs,
# DRAM/L2 bytes, grid, occupancy per kernel) and an nsys process (10 force-graph replays, gaps per node).
# Before the process an idle check (WSL and Windows nvidia-smi); during it a monitor samples Windows
# nvidia-smi, free -m and loadavg every 10 s.
set -u -o pipefail
MODE=$1
ROOT=$2
FM=/mnt/d/Margraf/FastMACE/.claude/worktrees/2-t20-5ac2bd
B=$HOME/fastmace-baseline/1a-T18
ENVS=$HOME/environments
CACHE=$HOME/fastmace-cache/2-T20a
WSMI=/mnt/c/Windows/System32/nvidia-smi.exe
mkdir -p "$ROOT" "$CACHE" "$ROOT/tmp"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$CACHE FASTMACE_MODELS=$HOME/models \
  FASTMACE_STRUCTURES=$B TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton \
  MACE_COMPILE_CACHE=$CACHE/compile XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp \
  TORCHINDUCTOR_COMPILE_THREADS=4
COMMON=(--checkpoint "$HOME/models/mace-mh-1.model" --model mh1-r2scan --backend hybrid --temperature-K 1000)

monitor() {
  while true; do
    printf '%s wsl_smi=%s win_smi=%s loadavg=%s mem_avail_mb=%s\n' "$(date -u +%FT%TZ)" \
      "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d ' ')" \
      "$($WSMI --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d ' \r')" \
      "$(cut -d' ' -f1-3 /proc/loadavg | tr ' ' ,)" "$(free -m | awk '/^Mem:/{print $7}')" >> "$ROOT/monitor.log"
    sleep 10
  done
}

idle_check() {  # name: settle up to 60 s for utilization <= 5 % in both views and no WSL compute process
  local name=$1 waited=0 pids util wutil
  while true; do
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    wutil=$($WSMI --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r')
    if [ -z "$pids" ] && [ "$util" -le 5 ] && [ "$wutil" -le 5 ]; then break; fi
    if [ $waited -ge 60 ]; then
      echo "$(date -u +%FT%TZ) $name NOT IDLE pids=$pids util=$util win_util=$wutil" >> "$ROOT/idle-checks.log"
      return 1
    fi
    sleep 5; waited=$((waited + 5))
  done
  {
    echo "$(date -u +%FT%TZ) $name idle (waited ${waited}s) wsl_compute_pids=[$pids] util=$util win_util=$wutil"
    echo "  wsl: $(nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader)"
    echo "  win: $($WSMI --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\r')"
    echo "  loadavg: $(cat /proc/loadavg)"
    echo "  free -m: $(free -m | awk '/^Mem:/{print "total="$2" used="$3" avail="$7}')"
    echo "  top cpu: $(ps -eo pcpu,comm --sort=-pcpu | sed -n 2,4p | tr -s ' ' | paste -sd';' -)"
  } >> "$ROOT/idle-checks.log"
}

NCU=/usr/local/cuda/bin/ncu
NSYS=/usr/local/cuda/bin/nsys
PY=$ENVS/mace_cueq_oeq/bin/python
TOOL=$FM/tools/step_breakdown.py

step() {  # label command...: one logged command
  local label=$1; shift
  echo "$(date -u +%FT%TZ) start $label" >> "$ROOT/run.log"
  "$@" > "$ROOT/$label.log" 2>&1
  local rc=$?
  echo "$(date -u +%FT%TZ) end $label exit=$rc" >> "$ROOT/run.log"
  return $rc
}

breakdown() {  # name structure; extra args: the main pass, then the ncu and nsys passes and their tables
  local name=$1 structure=$2; shift 2
  local dir=$ROOT/$name
  idle_check "$name" || return 1
  step "$name" "$PY" "$TOOL" --output "$dir" "${COMMON[@]}" --structure "$structure" "$@" || return 1
  # ncu: one eager force call, every kernel, real clocks and caches (not ncu's base clocks and flushes).
  idle_check "$name-ncu" &&
  step "$name-ncu" "$NCU" --profile-from-start off --target-processes all --cache-control none \
    --clock-control none --metrics "$("$PY" "$TOOL" --ncu-metrics)" --export "$dir/ncu" --force-overwrite \
    "$PY" "$TOOL" --pass ncu --output "$dir" "${COMMON[@]}" --structure "$structure" "$@" &&
  step "$name-ncu-csv" bash -c "'$NCU' --import '$dir/ncu.ncu-rep' --csv --page raw --print-units base \
    > '$dir/ncu_raw.csv'" &&
  step "$name-ncu-join" "$PY" "$TOOL" --output "$dir" --summarize-ncu "$dir/ncu_raw.csv"
  nsys_pass "$name" "$structure" "$@"
  return 0
}

nsys_pass() {  # name structure; extra args: --calls replays of the force graph with every graph node traced
  local name=$1 structure=$2; shift 2
  local dir=$ROOT/$name
  idle_check "$name-nsys" &&
  step "$name-nsys" "$NSYS" profile -t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi \
    --capture-range-end=stop -o "$dir/nsys" --force-overwrite true \
    "$PY" "$TOOL" --pass nsys --output "$dir" "${COMMON[@]}" --structure "$structure" "$@" &&
  step "$name-nsys-csv" "$NSYS" stats --report cuda_gpu_trace --format csv --output "$dir/nsys_stats" \
    --force-export true "$dir/nsys.nsys-rep" &&
  step "$name-nsys-gaps" "$PY" "$TOOL" --output "$dir" --summarize-nsys "$dir/nsys_stats_cuda_gpu_trace.csv" "$@"
}

monitor & MON=$!
trap 'kill $MON 2>/dev/null' EXIT
echo "$(date -u +%FT%TZ) mode=$MODE fastmace=$(git -C $FM rev-parse HEAD)" >> "$ROOT/run.log"
case $MODE in
  smoke) breakdown smoke NaCl --equilibrate-steps 20 --steps 50 --calls 2 ;;
  breakdown) breakdown NaCl NaCl; breakdown NbCl5 NbCl5 ;;
  nsys) nsys_pass "$3" "$3" ;;  # a pass again, into ROOT/<structure> of an earlier breakdown
esac
echo "$(date -u +%FT%TZ) done" >> "$ROOT/run.log"
