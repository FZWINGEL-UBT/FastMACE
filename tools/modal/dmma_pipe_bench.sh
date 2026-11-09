#!/usr/bin/env bash
# FP64 tensor cores (DMMA) as the second pipe beside FFMA (9 November 2026): tools/fma_dual_pipe_bench.py --dmma on
# any GPU (A100: FP32 64 FMA/clk/SM, DMMA peak 64; H200: FFMA 128, DMMA peak 128). The gate for a w' / forward lab:
# an FFMA:DMMA mix sustains at least 1.5x FP32 alone.
# Environment: OUT (the job sets it), SECONDS_PER_MIX (default 6).
set -uo pipefail
out=${OUT:?the job sets OUT}
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader > "$out/gpu.txt"
python tools/fma_dual_pipe_bench.py --output "$out/bench" --dmma --seconds "${SECONDS_PER_MIX:-6}" > "$out/bench.log" 2>&1
echo "exit=$?" >> "$out/bench.log"
cat "$out/gpu.txt" "$out/bench.log" > "$out/run.log"
