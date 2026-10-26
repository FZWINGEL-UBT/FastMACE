#!/usr/bin/env bash
# OPT6-T01 falsifier on the local RTX 5060 Ti (WSL), hybrid float32 (mace_cueq_oeq).
# Launch: setsid nohup bash results/opt6-T01-l2-chunks-20261026/run.sh <out-subdir> [probe args] > <log> 2>&1 &
set -u
ROOT=/mnt/d/Margraf/fm-opt6-T01
OUT=$ROOT/results/opt6-T01-l2-chunks-20261026/$1
shift
cd "$ROOT"
export PYTHONPATH=$ROOT PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=4
export FASTMACE_MODELS=$HOME/models FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18
export FASTMACE_CACHE=$HOME/fastmace-cache
OEQ=$HOME/environments/mace_cueq_oeq/bin/python
python3 tools/ci/local_gpu.py run --quiet --purpose "opt6-T01 L2-chunk falsifier" -- \
    "$OEQ" tools/bench/opt6_l2_chunked_conv.py --output "$OUT" "$@"
