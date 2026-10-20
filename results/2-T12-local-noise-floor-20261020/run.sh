#!/usr/bin/env bash
# 2-T12: noise floor of the local RTX 5060 Ti (WSL2). Launch detached:
#   setsid -f bash run.sh smoke|floor ROOT >/dev/null 2>&1
# smoke: one 20-step FastMACE A/B on NaCl. floor: three FastMACE-against-FastMACE processes (ABAB, ABBA, ABAB)
# and one research-engine-against-itself process (ABAB), MH-1 hybrid FP32 NVT, NaCl and NbCl5 3x3x3,
# 300 warm-up steps and 12 alternating blocks of 300 steps per side. Before each process an idle check
# (WSL and Windows nvidia-smi); during the run a monitor samples Windows nvidia-smi, free -m and loadavg
# every 10 s.
set -u -o pipefail
MODE=$1
ROOT=$2
FM=/mnt/d/Margraf/FastMACE/.claude/worktrees/2-t12-69e3f9
B=$HOME/fastmace-baseline/1a-T18
ENVS=$HOME/environments
CACHE=$HOME/fastmace-cache/2-T12
WSMI=/mnt/c/Windows/System32/nvidia-smi.exe
mkdir -p "$ROOT" "$CACHE" "$ROOT/tmp"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$CACHE FASTMACE_MODELS=$HOME/models \
  FASTMACE_STRUCTURES=$B TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton \
  MACE_COMPILE_CACHE=$CACHE/compile XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp \
  TORCHINDUCTOR_COMPILE_THREADS=4
COMMON=(--checkpoint "$HOME/models/mace-mh-1.model" --model mh1-r2scan --backend hybrid)

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
    echo "  loadavg: $(cat /proc/loadavg); users: $(who | awk '{print $1"@"$2}' | paste -sd' ' -)"
    echo "  free -m: $(free -m | awk '/^Mem:/{print "total="$2" used="$3" avail="$7}')"
    echo "  top cpu: $(ps -eo pcpu,comm --sort=-pcpu | sed -n 2,4p | tr -s ' ' | paste -sd';' -)"
  } >> "$ROOT/idle-checks.log"
}

fastmace_ab() {  # name pattern structures... ; extra args
  local name=$1 pattern=$2; shift 2
  idle_check "$name" || return 1
  echo "$(date -u +%FT%TZ) start $name" >> "$ROOT/run.log"
  "$ENVS/mace_cueq_oeq/bin/python" "$FM/tools/ab_step_time.py" --output "$ROOT/$name" --sides fastmace fastmace \
    --pattern "$pattern" "${COMMON[@]}" "$@" > "$ROOT/$name.log" 2>&1
  local rc=$?
  echo "$(date -u +%FT%TZ) end $name exit=$rc" >> "$ROOT/run.log"
  return $rc
}

baseline_ab() {
  local name=$1
  idle_check "$name" || return 1
  echo "$(date -u +%FT%TZ) start $name" >> "$ROOT/run.log"
  bash "$B/Experiments/Langevin_Thermostat/studies/hybrid_oeq/scripts/oeq_no_nvcc.sh" "$ENVS/mace_cueq_oeq" \
    "$CACHE/oeq-launcher" "$ENVS/mace_cueq_oeq/bin/python" "$FM/tools/ab_step_time.py" --output "$ROOT/$name" \
    --baseline "$B" --sides baseline baseline --pattern abab "${COMMON[@]}" \
    --structures NaCl NbCl5 --blocks 12 > "$ROOT/$name.log" 2>&1
  local rc=$?
  echo "$(date -u +%FT%TZ) end $name exit=$rc" >> "$ROOT/run.log"
  return $rc
}

monitor & MON=$!
trap 'kill $MON 2>/dev/null' EXIT
echo "$(date -u +%FT%TZ) mode=$MODE fastmace=$(git -C $FM rev-parse HEAD) baseline=$(cat $B/COMMIT)" >> "$ROOT/run.log"
case $MODE in
  smoke)
    fastmace_ab smoke abab --structures NaCl --warmup 20 --block 20 --blocks 2 ;;
  floor)
    fastmace_ab nf-fm-p1-abab abab --structures NaCl NbCl5 --blocks 12 &&
    fastmace_ab nf-fm-p2-abba abba --structures NaCl NbCl5 --blocks 12 &&
    fastmace_ab nf-fm-p3-abab abab --structures NaCl NbCl5 --blocks 12 &&
    baseline_ab nf-base-p4-abab ;;
esac
echo "$(date -u +%FT%TZ) done" >> "$ROOT/run.log"
