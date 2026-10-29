#!/usr/bin/env bash
# CAP-T07 (recompute_conv_weights) on the local RTX 5060 Ti (sm_120, WSL2), inside a reservation:
#   python3 tools/ci/local_gpu.py run --purpose "fm CAP-T07 JOB" -- \
#     env FM=<pinned source copy> bash results/cap-t07-20261029/local.sh ROOT JOB...
# Jobs: gate (tools/force_stress_gate.py, recompute_conv_weights=on; NaCl NbCl5 NaNbO3); gate-default (the default plan);
# nbcl5-ab / nbcl5-ba: tools/ab_rewrites.py on NbCl5 3x3x3, 12 blocks of 300 steps;
# big-ab / big-ba: NbCl5 4x4x8 (9,216 atoms), 8 blocks of 100 steps after 100 warm-up steps;
# a = recompute_conv_weights=off, b = on, build order ab (pattern abab) / ba (abba);
# plans: tools/bench/rescue_plans.py, the default plan's recompute_conv_weights per cell on the idle GPU's free memory.
# Before each job: no WSL compute process and utilisation <= 10 % in both nvidia-smi views for 30 s.
# Template: results/opt10-gate-epilogue-20261029/local.sh.
set -u -o pipefail
ROOT=$1; shift
FM=${FM:?set FM to a pinned source copy}
B=$HOME/fastmace-baseline/1a-T18
PY=$HOME/environments/mace_cueq_oeq/bin/python
REF=$HOME/environments/mace_cueq_latest/bin/python
CACHE=$HOME/fastmace-cache/cap-t07-local
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
  run "$job" "$PY" "$FM/tools/ab_rewrites.py" --output "$ROOT/$job" --checkpoint "$CK" --model mh1-r2scan \
      --structures "$structure" --a recompute_conv_weights=off --b recompute_conv_weights=on --order "$order" \
      --pattern "$pattern" "$@"
}
note "fastmace=$(cat "$FM/.source_commit") source=$FM host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
for job in "$@"; do
  case $job in
    gate) run $job "$PY" "$FM/tools/force_stress_gate.py" --output "$ROOT/gate" --engine fastmace \
              --checkpoint "$CK" --model mh1-r2scan --structures NaCl NbCl5 NaNbO3 --structures-root "$B" \
              --rewrites recompute_conv_weights=on --reference-command "$REF" --production-command "$PY" ;;
    gate-default) run $job "$PY" "$FM/tools/force_stress_gate.py" --output "$ROOT/gate-default" --engine fastmace \
              --checkpoint "$CK" --model mh1-r2scan --structures NaCl NbCl5 NaNbO3 --structures-root "$B" \
              --reference-command "$REF" --production-command "$PY" ;;
    plans) run $job "$PY" "${PLAN_TOOL:-$FM/tools/bench/rescue_plans.py}" --output "$ROOT/plans.json" \
               NaCl@3x3x3 NbCl5@3x3x3 NaNbO3@3x3x3 NbCl5@4x4x8 NbCl5@5x5x12 NbCl5@5x5x13 NbCl5@5x5x14 NbCl5@5x5x15 NbCl5@5x5x16 NbCl5@5x5x18 NbCl5@5x6x17 ;;
    nbcl5-ab) ab $job NbCl5 ab abab ;;  nbcl5-ba) ab $job NbCl5 ba abba ;;
    big-ab) ab $job NbCl5@4x4x8 ab abab --warmup 100 --block 100 --blocks 8 ;;
    big-ba) ab $job NbCl5@4x4x8 ba abba --warmup 100 --block 100 --blocks 8 ;;
    *) note "unknown job $job"; exit 2 ;;
  esac
done
note done
