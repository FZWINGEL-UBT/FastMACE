#!/bin/bash
# Measurement 5: paired MD-step A/B of (a) today's defaults against (b) PACKED8 off, same process, both build orders,
# Benchmark_loup settings (1 fs, skin 0.5, warm-up 200, 6 blocks of 256 steps per side), SrTiO3 10x10x10 and NbCl5
# 3x3x3. tools/ab_rewrites.py keeps every block time (statistics[side].block_ms) beside the medians.
#   local: source local_common.sh first (inside local_gpu.py run); Alex: alex_common.sh
S=fastmace.kernels.oeq_streaming.spline_forward
for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  ${RUNNER:-step} speed-$order ${SPEED_LIMIT:-3600} ${PY:-python} tools/ab_rewrites.py --output ${LOUT:-$OUT}/speed/ab-$order \
    --checkpoint $CK --model omat0-medium --structures "results/loup-a100-20261103/SrTiO3.cif@10x10x10" NbCl5 \
    --warmup 200 --block 256 --blocks 6 --order $order --pattern $pattern --timestep-fs 1 \
    --a-neighbours skin=0.5 --b-neighbours skin=0.5 --b-constant "$S:PACKED8=False"
done
