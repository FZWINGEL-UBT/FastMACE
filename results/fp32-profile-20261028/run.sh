#!/usr/bin/env bash
# fp32 profile (owner request, 28 October 2026): FastMACE's default float32 MD path, MH-1 matpes_r2scan, hybrid
# cuEq/OEq, every default rewrite, CUDA graphs; NaCl, NaNbO3 and NbCl5 3x3x3 at 1 fs, 1000 K.
# Local RTX 5060 Ti (WSL2), holding the local GPU reservation (quiet):
#   python3 tools/ci/local_gpu.py run --quiet --purpose "fp32 profile" -- \
#     bash results/fp32-profile-20261028/run.sh smoke|full ROOT
# Per structure, each its own process after an idle check:
#   main        tools/step_breakdown.py: wall time, stages (CUDA events), eager force call by kernel and category
#               (torch.profiler), force-graph replays (kernels, gaps)
#   ncu         one eager force call under Nsight Compute (DRAM/L2 bytes, FLOPs, occupancy per kernel)
#   nsys        force-graph replays under Nsight Systems with every graph node traced (gaps between nodes)
#   nsys-steps  whole MD steps under Nsight Systems, NVTX ranges per stage (stage GPU time, rebuilds, gaps)
#   parts       tools/bench/fp32_parts.py: the same plan on the eager tier, kernel time per model module
#   setup       tools/bench/fp32_setup.py: fm.md setup stages, warm caches (twice); NaCl also cold (fresh caches)
set -u -o pipefail
MODE=$1
ROOT=$2
FM=/mnt/d/Margraf/fm-fp32-profile
B=$HOME/fastmace-baseline/1a-T18
ENVS=$HOME/environments
CACHE=$HOME/fastmace-cache/fp32-profile
WSMI=/mnt/c/Windows/System32/nvidia-smi.exe
mkdir -p "$ROOT" "$CACHE" "$ROOT/tmp"
cache_env() {  # dir: every compile and build cache under dir
  export FASTMACE_CACHE=$1 TORCHINDUCTOR_CACHE_DIR=$1/inductor TRITON_CACHE_DIR=$1/triton \
    MACE_COMPILE_CACHE=$1/compile XDG_CACHE_HOME=$1/xdg
}
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_MODELS=$HOME/models FASTMACE_STRUCTURES=$B \
  TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4
cache_env "$CACHE"
CKPT=$HOME/models/mace-mh-1.model
COMMON=(--checkpoint "$CKPT" --model mh1-r2scan --backend hybrid --temperature-K 1000 --timestep-fs 1)
NCU=/usr/local/cuda/bin/ncu
NSYS=/usr/local/cuda/bin/nsys
PY=$ENVS/mace_cueq_oeq/bin/python
TOOL=$FM/tools/step_breakdown.py

monitor() {
  while true; do
    printf '%s wsl_smi=%s win_smi=%s loadavg=%s mem_avail_mb=%s\n' "$(date -u +%FT%TZ)" \
      "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d ' ')" \
      "$($WSMI --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d ' \r')" \
      "$(cut -d' ' -f1-3 /proc/loadavg | tr ' ' ,)" "$(free -m | awk '/^Mem:/{print $7}')" >> "$ROOT/monitor.log"
    sleep 10
  done
}

idle_check() {  # name: settle up to 120 s for utilization <= 5 % in both views and no WSL compute process
  local name=$1 waited=0 pids util wutil
  while true; do
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    wutil=$($WSMI --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r')
    if [ -z "$pids" ] && [ "$util" -le 5 ] && [ "$wutil" -le 5 ]; then break; fi
    if [ $waited -ge 120 ]; then
      echo "$(date -u +%FT%TZ) $name NOT IDLE pids=$pids util=$util win_util=$wutil" >> "$ROOT/idle-checks.log"
      return 1
    fi
    sleep 5; waited=$((waited + 5))
  done
  {
    echo "$(date -u +%FT%TZ) $name idle (waited ${waited}s) wsl_compute_pids=[$pids] util=$util win_util=$wutil"
    echo "  wsl: $(nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader)"
    echo "  loadavg: $(cat /proc/loadavg)  free -m: $(free -m | awk '/^Mem:/{print "total="$2" avail="$7}')"
    echo "  top cpu: $(ps -eo pcpu,comm --sort=-pcpu | sed -n 2,4p | tr -s ' ' | paste -sd';' -)"
  } >> "$ROOT/idle-checks.log"
}

step() {  # label command...: one logged command
  local label=$1; shift
  echo "$(date -u +%FT%TZ) start $label" >> "$ROOT/run.log"
  "$@" > "$ROOT/$label.log" 2>&1
  local rc=$?
  echo "$(date -u +%FT%TZ) end $label exit=$rc" >> "$ROOT/run.log"
  return $rc
}

profile_structure() {  # name structure; extra args to every step_breakdown pass
  local name=$1 structure=$2; shift 2
  local dir=$ROOT/$name
  idle_check "$name" || return 1
  step "$name" "$PY" "$TOOL" --output "$dir" "${COMMON[@]}" --structure "$structure" "$@" || return 1
  idle_check "$name-ncu" &&
  step "$name-ncu" "$NCU" --profile-from-start off --target-processes all --cache-control none \
    --clock-control none --metrics "$("$PY" "$TOOL" --ncu-metrics)" --export "$dir/ncu" --force-overwrite \
    "$PY" "$TOOL" --pass ncu --output "$dir" "${COMMON[@]}" --structure "$structure" "$@" &&
  step "$name-ncu-csv" bash -c "'$NCU' --import '$dir/ncu.ncu-rep' --csv --page raw --print-units base \
    > '$dir/ncu_raw.csv'" &&
  step "$name-ncu-join" "$PY" "$TOOL" --output "$dir" --summarize-ncu "$dir/ncu_raw.csv"
  idle_check "$name-nsys" &&
  step "$name-nsys" "$NSYS" profile -t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi \
    --capture-range-end=stop -o "$dir/nsys" --force-overwrite true \
    "$PY" "$TOOL" --pass nsys --output "$dir" "${COMMON[@]}" --structure "$structure" "$@" &&
  step "$name-nsys-csv" "$NSYS" stats --report cuda_gpu_trace --format csv --output "$dir/nsys_stats" \
    --force-export true "$dir/nsys.nsys-rep" &&
  step "$name-nsys-gaps" "$PY" "$TOOL" --output "$dir" --summarize-nsys "$dir/nsys_stats_cuda_gpu_trace.csv" "$@"
  idle_check "$name-nsys-steps" &&
  step "$name-nsys-steps" "$NSYS" profile -t cuda,nvtx,osrt --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop -o "$dir/nsys_steps" --force-overwrite true \
    "$PY" "$TOOL" --pass nsys-steps --output "$dir" "${COMMON[@]}" --structure "$structure" "$@" \
    --steps "$STEPS_NSYS" &&
  step "$name-nsys-steps-csv" "$NSYS" stats --report nvtx_gpu_proj_sum,nvtx_sum,cuda_gpu_kern_sum,cuda_gpu_trace,cuda_api_sum \
    --format csv --output "$dir/nsys_steps_stats" --force-export true "$dir/nsys_steps.nsys-rep"
  idle_check "$name-parts" &&
  step "$name-parts" "$PY" "$FM/tools/bench/fp32_parts.py" --output "$dir/parts" --checkpoint "$CKPT" \
    --structure "$structure" --temperature-K 1000 --timestep-fs 1 --calls "$CALLS" --equilibrate-steps "$EQ_PARTS"
  return 0
}

setup_run() {  # name structure label cache; extra args to fp32_setup.py
  local name=$1 structure=$2 label=$3 cache=$4; shift 4
  ( cache_env "$cache"
    idle_check "$name" &&
    step "$name" "$PY" "$FM/tools/bench/fp32_setup.py" --output "$ROOT/$name" --structure "$structure" \
      --label "$label" "$@" )
}

monitor & MON=$!
trap 'kill $MON 2>/dev/null' EXIT
echo "$(date -u +%FT%TZ) mode=$MODE fastmace=$(cat $FM/.source_commit 2>/dev/null) root=$ROOT" >> "$ROOT/run.log"
case $MODE in
  smoke)
    STEPS_NSYS=30 CALLS=2 EQ_PARTS=10
    profile_structure smoke-NaCl NaCl --equilibrate-steps 20 --steps 50 --calls 2
    setup_run smoke-setup NaCl warm "$CACHE" --steps 50 --timed-steps 50 ;;
  full)
    STEPS_NSYS=300 CALLS=10 EQ_PARTS=100
    setup_run setup-NaCl-cold NaCl cold "$ROOT/cold-cache-NaCl"
    for s in NaCl NaNbO3 NbCl5; do profile_structure "$s" "$s"; done
    for s in NaCl NaNbO3 NbCl5; do
      setup_run "setup-$s-warm1" "$s" warm "$CACHE"
      setup_run "setup-$s-warm2" "$s" warm "$CACHE"
    done
    setup_run setup-NaCl-cprofile NaCl warm-cprofile "$CACHE" --cprofile ;;
  structure) STEPS_NSYS=300 CALLS=10 EQ_PARTS=100; profile_structure "$3" "$3" ;;
esac
echo "$(date -u +%FT%TZ) done" >> "$ROOT/run.log"
