#!/bin/bash
# Shortened anneal (0 -> 1000 K in 2 ps, 1000 -> 250 K in 25 ps; relaxation and every-20th-step output
# as --profile anneal), two seeds per engine, one run at a time, then --quench of each.
S=$(dirname "$0")/t17_gpu.sh
R=$HOME/runs/1a-T17-20261018
for seed in 20261001 20261101; do
  for e in gpu fastmace; do
    out=$R/anneal-$e-$seed
    [ -f "$out/NaCl/metadata.json" ] && grep -q '"status": "complete"' "$out/NaCl/metadata.json" && continue
    echo "$(date -u +%FT%TZ) start $e $seed $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
    bash "$S" "$e" anneal --profile anneal --heat-ps 2 --cool-ps 25 --structures NaCl --seed $seed \
      --output "$out" > "$out.log" 2>&1
    echo "$(date -u +%FT%TZ) md $e $seed exit $?"
    bash "$S" "$e" quench --quench "$out" > "$out-quench.log" 2>&1
    echo "$(date -u +%FT%TZ) quench $e $seed exit $?"
  done
done
echo "$(date -u +%FT%TZ) ALL DONE"
