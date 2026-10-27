#!/usr/bin/env bash
# OPT6-T04 radial-table stub ceiling on the local RTX 5060 Ti (sm_120, WSL2), one reservation:
#   python3 tools/ci/local_gpu.py run --quiet --purpose "opt6-T04 stub" -- \
#     env FM=<pinned source copy> bash results/opt6-T04-radial-stub-20261027/local.sh ROOT JOB...
# Jobs: smoke (NaCl, 3 blocks of 100 steps); <structure>-ab / -ba (nacl, nbcl5, nanbo3):
# tools/bench/opt6_radial_stub.py, 12 blocks of 300 steps; a = the default plan, b = radial chain stubbed.
# Before each job: no WSL compute process and utilisation <= 10 % in both nvidia-smi views for 30 s.
set -u -o pipefail
ROOT=$1; shift
FM=${FM:?set FM to a pinned source copy}
B=$HOME/fastmace-baseline/1a-T18
PY=$HOME/environments/mace_cueq_oeq/bin/python
CACHE=$HOME/fastmace-cache/opt-6-local
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
stub() {  # job structure order pattern [extra args]
  local job=$1 structure=$2 order=$3 pattern=$4; shift 4
  run "$job" "$PY" "$FM/tools/bench/opt6_radial_stub.py" --output "$ROOT/$job" --checkpoint "$CK" \
      --structures "$structure" --order "$order" --pattern "$pattern" "$@"
}
note "fastmace=$(cat "$FM/.source_commit") source=$FM host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
for job in "$@"; do
  case $job in
    smoke) stub $job NaCl ab abab --warmup 100 --block 100 --blocks 3 ;;
    nacl-ab) stub $job NaCl ab abab ;;
    nacl-ba) stub $job NaCl ba abba ;;
    nbcl5-ab) stub $job NbCl5 ab abab ;;
    nbcl5-ba) stub $job NbCl5 ba abba ;;
    nanbo3-ab) stub $job NaNbO3 ab abab ;;
    nanbo3-ba) stub $job NaNbO3 ba abba ;;
    *) note "unknown job $job" ;;
  esac
done
note done
