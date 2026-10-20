#!/usr/bin/env bash
# 2-T20a: kernel timing breakdown of one MD step of FastMACE, MH-1 hybrid FP32 production defaults,
# NaCl 3x3x3, local RTX 5060 Ti (WSL2). Launch detached, holding the local GPU reservation:
#   setsid -f python3 tools/ci/local_gpu.py run --purpose "2-T20a step breakdown" -- \
#     bash results/2-T20a-step-breakdown-20261020/run.sh smoke|breakdown ROOT >/dev/null 2>&1
# smoke: 20 + 50 + 50 steps, 2 calls (checks the tool). breakdown: one fresh process, 300 equilibration
# steps, 1,000 steps per window, 10 profiled force calls, as the research engine's 15 October probe.
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
COMMON=(--checkpoint "$HOME/models/mace-mh-1.model" --model mh1-r2scan --backend hybrid --structure NaCl
  --temperature-K 1000)

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

breakdown() {  # name; extra args
  local name=$1; shift
  idle_check "$name" || return 1
  echo "$(date -u +%FT%TZ) start $name" >> "$ROOT/run.log"
  "$ENVS/mace_cueq_oeq/bin/python" "$FM/tools/step_breakdown.py" --output "$ROOT/$name" "${COMMON[@]}" "$@" \
    > "$ROOT/$name.log" 2>&1
  local rc=$?
  echo "$(date -u +%FT%TZ) end $name exit=$rc" >> "$ROOT/run.log"
  return $rc
}

monitor & MON=$!
trap 'kill $MON 2>/dev/null' EXIT
echo "$(date -u +%FT%TZ) mode=$MODE fastmace=$(git -C $FM rev-parse HEAD)" >> "$ROOT/run.log"
case $MODE in
  smoke) breakdown smoke --equilibrate-steps 20 --steps 50 --calls 2 ;;
  breakdown) breakdown NaCl ;;
esac
echo "$(date -u +%FT%TZ) done" >> "$ROOT/run.log"
