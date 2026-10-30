#!/usr/bin/env bash
# CAP-T05 (capture and warm-up snapshots keep buffers of 1 MiB or more in pinned host memory) on the local
# RTX 5060 Ti (sm_120, WSL2):
#   python3 tools/ci/local_gpu.py run --purpose "fm CAP-T05 JOB" -- #     env FM=<pinned source copy> bash results/cap-t05-20261030/local.sh ROOT JOB...
# Jobs: <cell>-<order>: tools/ab_rewrites.py --prepare, cell nbcl5 (3x3x3, 12 blocks of 300 steps) or big (NbCl5
# 4x4x8, 8 blocks of 100 steps after 100 warm-up steps); a = no offload (fastmace.engine.graph:
# CAPTURE_SNAPSHOT_OFFLOAD_BYTES = 2**62 while side a is built and its first-step stages are captured), b = the
# default (1 MiB); order ab (pattern abab) or ba (abba).
# Before each job: no WSL compute process and utilisation <= 10 % in both nvidia-smi views for 30 s.
# Template: results/opt10-gate-epilogue-20261029/local.sh.
set -u -o pipefail
ROOT=$1; shift
FM=${FM:?set FM to a pinned source copy}
B=$HOME/fastmace-baseline/1a-T18
PY=$HOME/environments/mace_cueq_oeq/bin/python
REF=$HOME/environments/mace_cueq_latest/bin/python
CACHE=$HOME/fastmace-cache/cap-t05-local
WSMI=/mnt/c/Windows/System32/nvidia-smi.exe
CK=$HOME/models/mace-mh-1.model
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
  note "$name idle: util=$util win=$wutil mem=$($WSMI --query-gpu=memory.used --format=csv,noheader | tr -d '\r') load=$(cut -d' ' -f1-3 /proc/loadavg) avail_mb=$(free -m | awk '/^Mem:/{print $7}')"
}
run() {  # name command...
  local name=$1; shift
  idle "$name" || exit 3
  note "start $name"
  "$@" > "$ROOT/$name.log" 2>&1 < /dev/null
  note "end $name exit=$?"
}
ab() {  # job structure order pattern [extra args]
  local job=$1 structure=$2 order=$3 pattern=$4; shift 4
  run "$job" "$PY" "$FM/tools/ab_rewrites.py" --output "$ROOT/$job" --checkpoint "$CK" --model mh1-r2scan       --structures "$structure" --prepare --a-constant "fastmace.engine.graph:CAPTURE_SNAPSHOT_OFFLOAD_BYTES=4611686018427387904"       --order "$order" --pattern "$pattern" "$@"
}
note "fastmace=$(cat "$FM/.source_commit") source=$FM host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
for job in "$@"; do
  case $job in
    nbcl5-ab|nbcl5-ba) ab "$job" NbCl5 "${job#nbcl5-}" "$([ "$job" = nbcl5-ab ] && echo abab || echo abba)" ;;
    big-ab|big-ba) ab "$job" NbCl5@4x4x8 "${job#big-}" "$([ "$job" = big-ab ] && echo abab || echo abba)"          --warmup 100 --block 100 --blocks 8 ;;
    *) note "unknown job $job"; exit 2 ;;
  esac
done
note done
