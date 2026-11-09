#!/usr/bin/env bash
# SMEM_SPILL re-test on the H200 (9 November 2026). The earlier A/Bs (158399f2) ran uncapped kernels: with
# __launch_bounds__ present ptxas ignored --maxrregcount (fixed in 183e7d3d, the cap as minimum blocks per SM).
# 1. lab: w-prime with shared-memory spilling at caps 72 / 64 / 56 / 48 (registers and local bytes: does the pragma
#    work on sm_90?) and kernel time against the default;
# 2. ab: MD-step A/Bs (compiler_spill_ab.sh, SrTiO3 10x10x10, both build orders) for VARIANTS (default spill64 spill56).
# Environment: OUT (the job sets it), PARTS (default "lab ab").
set -uo pipefail
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
echo "$gpu" | grep -q H200 || { note "WRONG-GPU $gpu"; exit 3; }
note "h200_smem_spill on $gpu; source $(cat .source_commit 2>/dev/null)"
for part in ${PARTS:-lab ab}; do
  case $part in
    lab)
      python tools/omat0_kernel_lab.py --checkpoint "$ck" --kernels wprime \
        --variants base base-s72 base-s64 base-s56 base-s48 \
        --rounds 21 --ptxas-archs 9.0 --output "$out/lab" > "$out/lab.log" 2>&1 < /dev/null
      note "end lab exit=$?"
      grep -h -E "^wprime " "$out/lab.log" | cut -c1-240 >> "$log" ;;
    ab)
      mkdir -p "$out/ab"
      OUT=$out/ab VARIANTS="${VARIANTS:-spill64 spill56}" bash tools/modal/compiler_spill_ab.sh
      note "end ab $(grep -h 'end ab' "$out"/ab/*/run.log | cut -c1-300 | tr '\n' ' ')" ;;
  esac
done
note done
