#!/bin/bash
# Colab A100 batch 2 (11 Nov 2026): GHOIST against today's A100 defaults. MD-step A/B in both build orders on
# Benchmark_loup SrTiO3 10x10x10 and NbCl5 3x3x3 (ab_pack v1), then the force/stress gate (gate_pack v1).
export EXPECT_GPU=A100
export PACK="ghoist|fastmace.kernels.oeq_streaming.spline_forward:GHOIST=True"
base=$OUT
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,power.limit,driver_version --format=csv > $base/gpu.csv
mkdir -p $base/ab-srtio3 $base/ab-nbcl5 $base/gate
OUT=$base/ab-srtio3 bash tools/jobs/ab_pack.sh; echo "ab-srtio3 $?"; cat $base/ab-srtio3/run.log
OUT=$base/ab-nbcl5 STRUCTURE=NbCl5 CELL=3x3x3 bash tools/jobs/ab_pack.sh; echo "ab-nbcl5 $?"; cat $base/ab-nbcl5/run.log
OUT=$base/gate bash tools/jobs/gate_pack.sh; echo "gate $?"; cat $base/gate/run.log
# the production transform timed in-process against today's forward, with its rounding against real
OUT=$base python tools/spline_table_bound.py --checkpoint $FASTMACE_MODELS/mace-omat-0-medium.model \
  --output $base/ghoist-prod --table-reads ghoist ghoist-prod > $base/ghoist-prod.log 2>&1; echo "ghoist-prod $?"
grep -E "^forward|registers" $base/ghoist-prod.log | tail -8
