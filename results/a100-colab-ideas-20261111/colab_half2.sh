#!/bin/bash
# Colab A100 batch 3 (11 Nov 2026), idea 1 stage 3: the two-far-edges-per-half2 w' channel walk against today's
# fp32 walk (half2_wprime_lab.py: timing per edge, numbers against fp32), at two far-edge counts.
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,power.limit,driver_version --format=csv > $OUT/gpu.csv
cat $OUT/gpu.csv
for n in 131072 65536; do
  python results/a100-colab-ideas-20261111/half2_wprime_lab.py --output $OUT/half2-$n --edges $n --reps 100 \
    > $OUT/half2-$n.log 2>&1; echo "half2-$n $?"
  tail -40 $OUT/half2-$n.log
done
