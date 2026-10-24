#!/usr/bin/env bash
# OPT4-T06: Nsight Compute verdict on the OEq streaming forward, NbCl5, MH-1 hybrid float32, local RTX 5060 Ti.
# From a pinned export of main 13a1983. Run inside one tools/ci/local_gpu.py reservation (untimed).
# 1. probe_forward.py: kernel-only launch sweep on the real interaction-2 inputs (block sizes, register caps).
# 2. the same probe under ncu --set full: one launch per variant, real clocks, caches flushed per pass.
set -uo pipefail
src=$HOME/fastmace-opt4/src-13a1983
out=$HOME/fastmace-opt4/T06
NCU=/usr/local/cuda/bin/ncu
export PYTHONPATH=$src FASTMACE_MODELS=$HOME/models FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18
export PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=4
py=$HOME/environments/mace_cueq_oeq/bin/python
ckpt=$HOME/models/mace-mh-1.model
probe=$src/results/opt4-T06/probe_forward.py
cd "$src"
mkdir -p "$out"
echo "$(date -u +%FT%TZ) probe"
timeout 600 "$py" "$probe" --checkpoint "$ckpt" --structure NbCl5 --output "$out" > "$out/probe.log" 2>&1 < /dev/null \
  || { echo "probe failed: $(tail -5 "$out/probe.log")"; exit 1; }
echo "$(date -u +%FT%TZ) ncu"
timeout 600 "$NCU" --profile-from-start off --target-processes all --set full --clock-control none \
  --cache-control all -k regex:oeq_projected_forward --import-source no --export "$out/ncu" --force-overwrite \
  "$py" "$probe" --checkpoint "$ckpt" --structure NbCl5 --output "$out" --ncu-range --blocks 1 --launches 1 \
  > "$out/ncu.log" 2>&1 < /dev/null || { echo "ncu failed: $(tail -5 "$out/ncu.log")"; exit 1; }
"$NCU" --import "$out/ncu.ncu-rep" --page details --print-units base > "$out/ncu_details.txt" 2>&1
"$NCU" --import "$out/ncu.ncu-rep" --csv --page raw --print-units base > "$out/ncu_raw.csv" 2>&1
echo "$(date -u +%FT%TZ) done"
