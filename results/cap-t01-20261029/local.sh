#!/usr/bin/env bash
# CAP-T01 (force_graphs) on the local RTX 5060 Ti (sm_120, WSL2), inside a reservation:
#   python3 tools/ci/local_gpu.py run --purpose "fm CAP-T01 JOB" -- \
#     env FM=<pinned source copy> bash results/cap-t01-20261029/local.sh ROOT JOB...
# Jobs: gate (tools/force_stress_gate.py, default plan: force_graphs resolves to fit; NaCl NbCl5 NaNbO3);
# <s>-ab / <s>-ba: tools/ab_rewrites.py, 12 blocks of 300 steps, a = force_graphs=keep, b = fit, build order ab/ba.
# Before each job: no WSL compute process and utilisation <= 10 % in both nvidia-smi views for 30 s.
# Template: results/opt10-gate-epilogue-20261029/local.sh.
set -u -o pipefail
ROOT=$1; shift
FM=${FM:?set FM to a pinned source copy}
B=$HOME/fastmace-baseline/1a-T18
PY=$HOME/environments/mace_cueq_oeq/bin/python
REF=$HOME/environments/mace_cueq_latest/bin/python
CACHE=$HOME/fastmace-cache/cap-t01-local
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
ab() {  # job structure order
  run "$1" "$PY" "$FM/tools/ab_rewrites.py" --output "$ROOT/$1" --checkpoint "$CK" --model mh1-r2scan \
      --structures "$2" --a-engine force_graphs=keep --b-engine force_graphs=fit --order "$3" --pattern abab
}
note "fastmace=$(cat "$FM/.source_commit") source=$FM host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
for job in "$@"; do
  case $job in
    gate) run $job "$PY" "$FM/tools/force_stress_gate.py" --output "$ROOT/gate" --engine fastmace \
              --checkpoint "$CK" --model mh1-r2scan --structures NaCl NbCl5 NaNbO3 --structures-root "$B" \
              --reference-command "$REF" --production-command "$PY" ;;
    nacl-ab) ab $job NaCl ab ;;  nacl-ba) ab $job NaCl ba ;;
    nbcl5-ab) ab $job NbCl5 ab ;;  nbcl5-ba) ab $job NbCl5 ba ;;
    nanbo3-ab) ab $job NaNbO3 ab ;;  nanbo3-ba) ab $job NaNbO3 ba ;;
    *) note "unknown job $job"; exit 2 ;;
  esac
done
