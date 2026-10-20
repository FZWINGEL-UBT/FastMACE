#!/usr/bin/env bash
# Optimisation round 1 on the local RTX 5060 Ti (sm_120, WSL2), one job at a time. Launch detached, holding the
# local GPU reservation:
#   setsid -f python3 tools/ci/local_gpu.py run --purpose "opt-1 local runs" -- \
#     bash results/opt-T02b-segment-gemm/local.sh ROOT JOB... >/dev/null 2>&1
# Jobs: stageA-ab, stageA-ba (gated_rows off against on, tools/ab_rewrites.py, 12 blocks, both orders);
# stageB-ab, stageB-ba (gated_rows against gated_rows + segment_gemm); t03 (block_linear_concurrency_bench);
# t01 (gemm_precision_bench with the 4096^3 peak test, on 2-T20a's gemms.csv). Before each job: no WSL compute
# process and utilisation <= 5 % in both nvidia-smi views for 30 s (the desktop idle), else it stops.
set -u -o pipefail
ROOT=$1; shift
FM=/mnt/d/Margraf/FastMACE/.claude/worktrees/2-t20-5ac2bd
B=$HOME/fastmace-baseline/1a-T18
PY=$HOME/environments/mace_cueq_oeq/bin/python
CACHE=$HOME/fastmace-cache/opt-1-local
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
    if [ -z "$pids" ] && [ "$util" -le 5 ] && [ "$wutil" -le 5 ]; then ok=$((ok + 1)); else ok=0; fi
    tries=$((tries + 1))
    if [ $tries -ge 120 ]; then note "$name NOT IDLE pids=[$pids] util=$util win=$wutil"; return 1; fi
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
note "fastmace=$(cat $FM/.source_commit) host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
for job in "$@"; do
  case $job in
    stageA-ab) run $job "$PY" "$FM/tools/ab_rewrites.py" --output "$ROOT/$job" --checkpoint "$CK" --b gated_rows=on \
                 --order ab --pattern abab ;;
    stageA-ba) run $job "$PY" "$FM/tools/ab_rewrites.py" --output "$ROOT/$job" --checkpoint "$CK" --b gated_rows=on \
                 --order ba --pattern abba ;;
    stageB-ab) run $job "$PY" "$FM/tools/ab_rewrites.py" --output "$ROOT/$job" --checkpoint "$CK" --a gated_rows=on \
                 --b gated_rows=on segment_gemm=on --order ab --pattern abab ;;
    stageB-ba) run $job "$PY" "$FM/tools/ab_rewrites.py" --output "$ROOT/$job" --checkpoint "$CK" --a gated_rows=on \
                 --b gated_rows=on segment_gemm=on --order ba --pattern abba ;;
    t03) run $job "$PY" "$FM/tools/block_linear_concurrency_bench.py" --output "$ROOT/$job" ;;
    t01) run $job "$PY" "$FM/tools/gemm_precision_bench.py" --output "$ROOT/$job" \
           --gemms "$FM/results/2-T20a-step-breakdown-20261020/NaCl/gemms.csv" \
           "$FM/results/2-T20a-step-breakdown-20261020/NbCl5/gemms.csv" ;;
    *) note "unknown job $job" ;;
  esac
done
note done
