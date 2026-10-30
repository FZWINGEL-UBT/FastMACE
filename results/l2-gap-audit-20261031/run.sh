#!/usr/bin/env bash
# M1 (L2 audit) and M2 (gap trace), plans/capacity-ideas-20261030.md, eighth list: FastMACE's default float32 MD
# path (MH-1 matpes_r2scan, hybrid, CUDA graphs, every default rewrite), 1 fs, 1000 K, as
# results/fp32-profile-20261028/run.sh, on the local RTX 5060 Ti. Holding the local GPU reservation:
#   python3 tools/ci/local_gpu.py run --purpose "fm M1/M2" -- env FM=<source copy> bash run.sh ROOT
# Per cell, each its own process after an idle check:
#   main        wall time, stages (CUDA events), the eager call's kernel sequence, force-graph replays
#   ncu         (large cells) one eager force call of the compiled model under Nsight Compute: DRAM and L2 bytes and
#               the L2 sector hit rate per kernel (step_breakdown's metrics plus lts__t_sector_hit_rate.pct)
#   nsys        force-graph replays, every graph node traced: the gaps between the graph's kernels (M2 b)
#   nsys-steps  whole MD steps, NVTX ranges per stage: GPU idle time between steps and stages (M2 a), and the
#               kernel time per name over whole steps (M1's share of the step)
set -u -o pipefail
ROOT=$1
FM=${FM:?set FM to a source copy}
B=$HOME/fastmace-baseline/1a-T18
CACHE=$HOME/fastmace-cache/l2-gap-audit
WSMI=/mnt/c/Windows/System32/nvidia-smi.exe
mkdir -p "$ROOT/tmp" "$CACHE"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_MODELS=$HOME/models FASTMACE_STRUCTURES=$B \
  TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 \
  FASTMACE_CACHE=$CACHE TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton \
  MACE_COMPILE_CACHE=$CACHE/compile XDG_CACHE_HOME=$CACHE/xdg
PY=$HOME/environments/mace_cueq_oeq/bin/python
TOOL=$FM/tools/step_breakdown.py
NCU=/usr/local/cuda/bin/ncu
NSYS=/usr/local/cuda/bin/nsys
COMMON=(--checkpoint "$HOME/models/mace-mh-1.model" --model mh1-r2scan --backend hybrid --temperature-K 1000
        --timestep-fs 1)
idle() {  # name: up to 30 min for no WSL compute process and <= 5 % utilisation in both views
  local name=$1 waited=0 pids util wutil
  while true; do
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    wutil=$($WSMI --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r')
    if [ -z "$pids" ] && [ "$util" -le 5 ] && [ "$wutil" -le 5 ]; then break; fi
    if [ $waited -ge 1800 ]; then echo "$(date -u +%FT%TZ) $name NOT IDLE" >> "$ROOT/run.log"; return 1; fi
    sleep 5; waited=$((waited + 5))
  done
  echo "$(date -u +%FT%TZ) $name idle (waited ${waited}s) util=$util win=$wutil load=$(cut -d' ' -f1-3 /proc/loadavg)" \
    >> "$ROOT/run.log"
}
step() {  # label command...
  local label=$1; shift
  echo "$(date -u +%FT%TZ) start $label" >> "$ROOT/run.log"
  "$@" > "$ROOT/$label.log" 2>&1
  local rc=$?
  echo "$(date -u +%FT%TZ) end $label exit=$rc" >> "$ROOT/run.log"
  return $rc
}
cell() {  # name structure steps ncu(0/1)
  local name=$1 structure=$2 steps=$3 ncu=$4 dir=$ROOT/$1
  mkdir -p "$dir"
  idle "$name" &&  # main pass: wall time, stages, the eager call's kernel sequence (the ncu join reads it)
  step "$name" "$PY" "$TOOL" --output "$dir" "${COMMON[@]}" --structure "$structure" --equilibrate-steps 100 \
    --steps 300 || return 1
  if [ "$ncu" = 1 ]; then
    idle "$name-ncu" &&
    step "$name-ncu" "$NCU" --profile-from-start off --target-processes all --cache-control none \
      --clock-control none --metrics "$("$PY" "$TOOL" --ncu-metrics),lts__t_sector_hit_rate.pct,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum" \
      --export "$dir/ncu" --force-overwrite \
      "$PY" "$TOOL" --pass ncu --output "$dir" "${COMMON[@]}" --structure "$structure" --equilibrate-steps 100 &&
    step "$name-ncu-csv" bash -c "'$NCU' --import '$dir/ncu.ncu-rep' --csv --page raw --print-units base > '$dir/ncu_raw.csv'" &&
    step "$name-ncu-join" "$PY" "$TOOL" --output "$dir" --summarize-ncu "$dir/ncu_raw.csv"
  fi
  idle "$name-nsys" &&
  step "$name-nsys" "$NSYS" profile -t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi \
    --capture-range-end=stop -o "$dir/nsys" --force-overwrite true \
    "$PY" "$TOOL" --pass nsys --output "$dir" "${COMMON[@]}" --structure "$structure" --equilibrate-steps 100 &&
  step "$name-nsys-csv" "$NSYS" stats --report cuda_gpu_trace --format csv --output "$dir/nsys_stats" \
    --force-export true "$dir/nsys.nsys-rep" &&
  step "$name-nsys-gaps" "$PY" "$TOOL" --output "$dir" --summarize-nsys "$dir/nsys_stats_cuda_gpu_trace.csv"
  idle "$name-nsys-steps" &&
  step "$name-nsys-steps" "$NSYS" profile -t cuda,nvtx,osrt --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop -o "$dir/nsys_steps" --force-overwrite true \
    "$PY" "$TOOL" --pass nsys-steps --output "$dir" "${COMMON[@]}" --structure "$structure" --equilibrate-steps 100 \
    --steps "$steps" &&
  step "$name-nsys-steps-csv" "$NSYS" stats --report nvtx_gpu_proj_sum,nvtx_sum,cuda_gpu_kern_sum,cuda_gpu_trace \
    --format csv --output "$dir/nsys_steps_stats" --force-export true "$dir/nsys_steps.nsys-rep"
  return 0
}
echo "$(date -u +%FT%TZ) fastmace=$(cat "$FM/.source_commit") gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)" \
  >> "$ROOT/run.log"
cell NbCl5-9216 NbCl5@4x4x8 100 1
cell NaNbO3-8640 NaNbO3@6x6x6 100 1
cell NbCl5-1944 NbCl5 300 0
cell NaNbO3-1080 NaNbO3 300 0
echo "$(date -u +%FT%TZ) done" >> "$ROOT/run.log"
