#!/bin/bash
# Colab A100 batch (11 Nov 2026): HALF_FAR against today's A100 defaults. First the force/stress gate (gate_pack v1:
# SrTiO3 3x3x3, NaCl, NbCl5; the default once, then each switch); A/Bs only if every gate passes: MD step in both
# build orders on SrTiO3 10x10x10 and NbCl5 3x3x3 (ab_pack v1).
export EXPECT_GPU=A100
W=fastmace.kernels.oeq_streaming.wprime_backward
export PACK="hf58|$W:HALF_FAR=True $W:HALF_FAR_RADIUS=5.8;hf55|$W:HALF_FAR=True $W:HALF_FAR_RADIUS=5.5;hf58b2|$W:HALF_FAR=True $W:HALF_FAR_RADIUS=5.8 $W:HALF_FAR_MIN_BLOCKS=2"
base=$OUT
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,power.limit,driver_version --format=csv > $base/gpu.csv
cat $base/gpu.csv
mkdir -p $base/gate $base/ab-srtio3 $base/ab-nbcl5
OUT=$base/gate bash tools/jobs/gate_pack.sh; g=$?; echo "gate $g"; cat $base/gate/run.log
if [ "$g" != 0 ]; then
  for f in $base/gate/gate-*.log; do echo "== $f"; tail -30 "$f"; done
  echo "gate failed: no A/Bs"; exit 0
fi
OUT=$base/ab-srtio3 bash tools/jobs/ab_pack.sh; echo "ab-srtio3 $?"; cat $base/ab-srtio3/run.log
OUT=$base/ab-nbcl5 STRUCTURE=NbCl5 CELL=3x3x3 bash tools/jobs/ab_pack.sh; echo "ab-nbcl5 $?"; cat $base/ab-nbcl5/run.log
