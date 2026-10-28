#!/usr/bin/env bash
cd ~/tmp/fm-ideas
export PYTHONPATH=/mnt/d/Margraf/fm-ideas PYTHONDONTWRITEBYTECODE=1
P=~/environments/mace_cueq_oeq/bin/python
B=/mnt/d/Margraf/fm-ideas/tools/bench/ideas_d2_wtable.py
EMU="env CUBLAS_EMULATE_DOUBLE_PRECISION=1 CUBLAS_EMULATION_STRATEGY=eager"
for r in 1 2; do
  $EMU CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT=56 timeout 600 $P $B >> d2.jsonl 2>> d2.err
  $EMU CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT=56 timeout 600 $P $B --edges 84072 --pairs 9 --pieces 400 >> d2.jsonl 2>> d2.err
done
$EMU timeout 600 $P $B >> d2.jsonl 2>> d2.err
echo "$(date -u +%FT%TZ) P8DONE" >> progress8.txt
