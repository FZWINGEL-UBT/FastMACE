#!/usr/bin/env bash
# OPT2-T03 (folded_backward_tiles) on the local RTX 5060 Ti (sm_120, WSL2), one job per reservation (at most
# 15 min each, the round-2 fair-share rule):
#   python3 tools/ci/local_gpu.py run --purpose "opt2-T03 JOB" -- \
#     env FM=<pinned source copy> bash results/opt2-T03-folded-tiles/local.sh ROOT JOB
# FM is a pinned copy of the source (git archive, with .source_commit), never the live worktree. Jobs:
# - check: the CUDA bitwise test (tests/gpu/test_folded_tiles_cuda.py);
# - ncu: the kernel at the three padded edge counts, fixed and waves tiles (ncu_kernel.py);
# - <structure>-ab, <structure>-ba (nacl, nbcl5, nanbo3): tools/ab_rewrites.py, 12 blocks, one order each;
#   a = the round-2 baseline (OPT-T08 records on), b = a + folded_backward_tiles.
# Before each job: no WSL compute process and utilisation <= 5 % in both nvidia-smi views for 30 s.
set -u -o pipefail
ROOT=$1; shift
FM=${FM:?set FM to a pinned source copy}
B=$HOME/fastmace-baseline/1a-T18
PY=$HOME/environments/mace_cueq_oeq/bin/python
CACHE=$HOME/fastmace-cache/opt-2-local
WSMI=/mnt/c/Windows/System32/nvidia-smi.exe
CK=$HOME/models/mace-mh-1.model
NCU=/usr/local/cuda/bin/ncu
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
SIDE_A="species_tables=on fold_readout_tail=on fold_product_linear=on folded_backward_tiles=off"
SIDE_B="species_tables=on fold_readout_tail=on fold_product_linear=on folded_backward_tiles=on"
ab() {  # job structure order pattern
  run "$1" "$PY" "$FM/tools/ab_rewrites.py" --output "$ROOT/$1" --checkpoint "$CK" --structures "$2" \
      --a $SIDE_A --b $SIDE_B --order "$3" --pattern "$4"
}
note "fastmace=$(cat $FM/.source_commit) source=$FM host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
for job in "$@"; do
  case $job in
    check) run $job "$PY" -m pytest -p no:cacheprovider -q -rf "$FM/tests/gpu/test_folded_tiles_cuda.py" ;;
    ncu) run $job "$NCU" --kernel-name kernel --section SpeedOfLight --section Occupancy \
             --section LaunchStats --csv --page details "$PY" "$FM/results/opt2-T03-folded-tiles/ncu_kernel.py" ;;
    nacl-ab) ab $job NaCl ab abab ;;
    nacl-ba) ab $job NaCl ba abba ;;
    nbcl5-ab) ab $job NbCl5 ab abab ;;
    nbcl5-ba) ab $job NbCl5 ba abba ;;
    nanbo3-ab) ab $job NaNbO3 ab abab ;;
    nanbo3-ba) ab $job NaNbO3 ba abba ;;
    *) note "unknown job $job" ;;
  esac
done
note done
