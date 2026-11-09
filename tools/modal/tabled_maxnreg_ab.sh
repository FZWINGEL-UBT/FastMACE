#!/usr/bin/env bash
# tabled_outer_sum.FORWARD_MAXNREG on the A100 (9 November 2026, from tools/modal/step_limiters.sh: the tabled forward
# runs at 128 registers on sm_80, 16 blocks per SM, 398 us against the H200's 112 at 64 registers). MD-step A/Bs
# (tools/modal/round6_ab256.sh, both build orders, OMAT-0 medium, SrTiO3 10x10x10, 300 K) for CAPS (default "64 80").
# Environment: OUT (the job sets it), CAPS, GPU_MATCH (default A100).
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
echo "$gpu" | grep -q "${GPU_MATCH:-A100}" || { note "WRONG-GPU $gpu"; exit 3; }
note "tabled_maxnreg_ab on $gpu; source $(cat .source_commit 2>/dev/null)"
for cap in ${CAPS:-64 80}; do
  mkdir -p "$out/r$cap"
  OUT=$out/r$cap NAME=maxnreg$cap B_CONSTANTS="fastmace.kernels.triton.tabled_outer_sum:FORWARD_MAXNREG=$cap" \
    STRUCTURES=results/loup-a100-20261103/SrTiO3.cif@10x10x10 bash tools/modal/round6_ab256.sh \
    > "$out/r$cap.log" 2>&1 < /dev/null
  note "end r$cap exit=$? $(grep -h 'end ab' "$out/r$cap/run.log" | cut -c1-300 | tr '\n' ' ')"
done
note done
