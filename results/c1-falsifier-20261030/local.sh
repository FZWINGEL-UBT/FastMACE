#!/usr/bin/env bash
# C1 falsifier (plans/capacity-ideas-20261030.md): the symmetric contraction's [N, 16, 512] input and its fusion with
# a neighbouring GEMM. MH-1 mh1-r2scan float32, hybrid, default plan, NbCl5, local RTX 5060 Ti (sm_120, WSL2):
#   python3 tools/ci/local_gpu.py run --purpose "fm c1 JOB" -- env FM=<pinned main copy> bash \
#     results/c1-falsifier-20261030/local.sh ROOT JOB...
# Jobs:
#   sb-<repeat>: tools/step_breakdown.py main pass (1 fs, 1000 K): wall time, stages, eager force calls of the compiled
#     model by kernel and operator (torch.profiler);
#   ncu-<repeat>: the ncu pass on sb-<repeat>'s directory (one eager force call; DRAM and L2 bytes per kernel), joined;
#   saved-<repeat>-<off|on>: tools/bench/saved_tensors.py, compiled default plan, recompute_conv_weights on and
#     recompute_node_segments off or on: every saved storage, its size and whether it is alive at the peak.
# Before each job: no WSL compute process and utilisation <= 10 % in both nvidia-smi views for 30 s.
# Template: results/cap-t08-chunks-20261030/local.sh and results/fp32-profile-20261028/run.sh.
set -u -o pipefail
ROOT=$1; shift
FM=${FM:?set FM to a pinned source copy}
B=$HOME/fastmace-baseline/1a-T18
PY=$HOME/environments/mace_cueq_oeq/bin/python
CACHE=$HOME/fastmace-cache/c1-falsifier-local
WSMI=/mnt/c/Windows/System32/nvidia-smi.exe
CK=$HOME/models/mace-mh-1.model
NCU=/usr/local/cuda/bin/ncu
TOOL=$FM/tools/step_breakdown.py
mkdir -p "$ROOT/tmp" "$CACHE"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$CACHE FASTMACE_MODELS=$HOME/models \
  FASTMACE_STRUCTURES=$B TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton \
  MACE_COMPILE_CACHE=$CACHE/compile XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp \
  TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_HYBRID_PYTHON=$PY
COMMON=(--checkpoint "$CK" --model mh1-r2scan --backend hybrid --temperature-K 1000 --timestep-fs 1)
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
sizes() {  # repeat -> step_breakdown sizes (the large cells take about 0.6 s per step)
  case $1 in
    6x6x20) SIZES=(--equilibrate-steps 30 --steps 60 --calls 4) ;;
    *) SIZES=(--equilibrate-steps 100 --steps 300 --calls 10) ;;
  esac
}
note "fastmace=$(cat "$FM/.source_commit") host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
for job in "$@"; do
  case $job in
    sb-*) rep=${job#sb-}; sizes "$rep"
      run "$job" "$PY" "$TOOL" --output "$ROOT/$job" "${COMMON[@]}" --structure "NbCl5@$rep" "${SIZES[@]}" ;;
    ncu-*) rep=${job#ncu-}; sizes "$rep"; dir=$ROOT/sb-$rep
      run "$job" "$NCU" --profile-from-start off --target-processes all --cache-control none --clock-control none \
        --metrics "$("$PY" "$TOOL" --ncu-metrics)" --export "$dir/ncu" --force-overwrite \
        "$PY" "$TOOL" --pass ncu --output "$dir" "${COMMON[@]}" --structure "NbCl5@$rep" "${SIZES[@]}"
      "$NCU" --import "$dir/ncu.ncu-rep" --csv --page raw --print-units base > "$dir/ncu_raw.csv" 2>> "$ROOT/$job.log"
      "$PY" "$TOOL" --output "$dir" --summarize-ncu "$dir/ncu_raw.csv" >> "$ROOT/$job.log" 2>&1
      note "joined $job exit=$?" ;;
    saved-*) IFS=- read -r _ rep on <<< "$job"
      run "$job" "$PY" "$FM/tools/bench/saved_tensors.py" --structure NbCl5 --repeat "$rep" \
        --output "$ROOT/$job" --rewrites recompute_conv_weights=on recompute_node_segments=$on ;;
    *) note "unknown job $job"; exit 2 ;;
  esac
done
note done
