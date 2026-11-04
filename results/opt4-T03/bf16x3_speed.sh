#!/usr/bin/env bash
# BF16x3 (FACTO, arXiv 2611.03276) speed screen: GEMM probe (cuBLAS IEEE, Triton ieee/tf32x3/bf16x3/bf16x6) and a
# same-process whole-step A/B, default plan (a) against tf32x3_gemms=on grouped_launch=off with the split products
# in bf16x3 (b). Usage: FM=<source> PY=<python> bash bf16x3_speed.sh ROOT [probe] [ab] [ab-ba]
set -u -o pipefail
ROOT=$1; shift
FM=${FM:?}
PY=${PY:?}
CK=${CK:?}
CACHE=${CACHE:?}
STRUCT=${STRUCT:?}
mkdir -p "$ROOT/tmp" "$CACHE"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$CACHE FASTMACE_STRUCTURES=$STRUCT \
  TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton MACE_COMPILE_CACHE=$CACHE/compile \
  XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_SPLIT_PRECISION=bf16x3
log=$ROOT/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "fastmace=$(cat "$FM/.source_commit") host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
for job in "$@"; do
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
  if [ -n "$pids" ]; then note "$job NOT IDLE pids=[$pids]"; exit 3; fi
  note "start $job"
  case $job in
    probe) "$PY" "$FM/results/opt-T01-gemm-precision-20261020/colab-bf16x6/a100/probe_bf16x6.py" \
             "$ROOT/probe.json" > "$ROOT/probe.log" 2>&1 ;;
    ab|ab-ba) order=ab; [ "$job" = ab-ba ] && order=ba
       "$PY" "$FM/tools/ab_rewrites.py" --output "$ROOT/$job" --checkpoint "$CK" --timestep-fs 1 --order $order \
         --b tf32x3_gemms=on grouped_launch=off > "$ROOT/$job.log" 2>&1 ;;
  esac
  note "end $job exit=$?"
done
note done
