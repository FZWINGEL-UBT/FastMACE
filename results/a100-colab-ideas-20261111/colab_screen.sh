#!/bin/bash
# Colab A100 screen of the 11 Nov 2026 ideas (run from the source root by tools/colab/fastmace_colab.py; $OUT set).
# 1: HFMA2 beside FFMA on sm_80;  4: TF32 big*big + FP16 cross GEMM;  5: forward ghoist on the A100 default;
# 2: two half-size force graphs on two streams.
ckpt=$FASTMACE_MODELS/mace-omat-0-medium.model
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,power.limit,driver_version --format=csv > $OUT/gpu.csv
cat $OUT/gpu.csv
python tools/fma_dual_pipe_bench.py --hfma2 --seconds 5 --output $OUT/hfma2 > $OUT/hfma2.log 2>&1; echo "hfma2 $?"
grep "^mix" $OUT/hfma2.log
python tools/fp16x3_gemm_bench.py --output $OUT/fp16x3_hybrid.json > $OUT/fp16x3.log 2>&1; echo "gemm $?"
cat $OUT/fp16x3.log
python tools/spline_table_bound.py --checkpoint $ckpt --output $OUT/ghoist --table-reads ghoist \
  > $OUT/ghoist.log 2>&1; echo "ghoist $?"
grep -i "forward\|ghoist\|registers" $OUT/ghoist.log | tail -30
python tools/bench/half_system_streams.py --checkpoint $ckpt --output $OUT/halves > $OUT/halves.log 2>&1
echo "halves $?"
tail -20 $OUT/halves.log
