#!/bin/bash
# Colab A100 (12 Nov 2026, conductor follow-up): WEIGHTED_VECTOR against today's A100 defaults, alone and with
# SKIP_OUTSIDE + HOIST_DX. MD-step A/B in both build orders on Benchmark_loup SrTiO3 10x10x10 and NbCl5 3x3x3
# (ab_pack v1); then the 3-structure force/stress gate (gate_pack v1) for an entry with all four b/a under 1.
# Run by fastmace_colab.py ($OUT set); the VM is stopped after the job.
export EXPECT_GPU=A100
M=fastmace.kernels.oeq_streaming.wprime_backward
skw="skip-hoist-weighted|$M:SKIP_OUTSIDE=True $M:HOIST_DX=True $M:WEIGHTED_VECTOR=True"
w="weighted|$M:WEIGHTED_VECTOR=True"
export PACK="$skw;$w"
base=$OUT
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,power.limit,driver_version --format=csv > $base/gpu.csv
mkdir -p $base/ab-srtio3 $base/ab-nbcl5 $base/gate
OUT=$base/ab-srtio3 bash tools/jobs/ab_pack.sh; echo "ab-srtio3 $?"; cat $base/ab-srtio3/run.log
OUT=$base/ab-nbcl5 STRUCTURE=NbCl5 CELL=3x3x3 bash tools/jobs/ab_pack.sh; echo "ab-nbcl5 $?"; cat $base/ab-nbcl5/run.log
win=""
for entry in "$skw" "$w"; do
  name=${entry%%|*}
  ratios=$(grep -h "end ab-$name-" $base/ab-srtio3/run.log $base/ab-nbcl5/run.log | grep -oE "b/a [0-9.]+" | cut -d' ' -f2)
  n=$(echo "$ratios" | grep -c .)
  below=$(echo "$ratios" | awk '$1 < 1' | grep -c .)
  echo "$name: b/a $(echo $ratios) ($below of $n under 1)"
  [ "$n" = 4 ] && [ "$below" = 4 ] && win="${win:+$win;}$entry"
done
if [ -n "$win" ]; then
  OUT=$base/gate PACK="$win" bash tools/jobs/gate_pack.sh; echo "gate $?"; cat $base/gate/run.log
else
  echo "no entry under 1 in all four A/Bs: no gate"
fi
