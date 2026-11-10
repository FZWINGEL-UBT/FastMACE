#!/usr/bin/env bash
# Round 2 follow-ups on the H200 (9 November 2026), tools/spline_table_bound.py on SrTiO3 10x10x10 (crystal) and the
# 0.3 A disordered snapshot (20 MD steps):
# - forward: packed8 (S12) and packed8v2 (the same values, decoded by OR-ing the exponent into each mantissa field
#   and one exact subtraction, no integer-to-float conversions), the deferred-CG forward (76 phi accumulators, CG
#   once per receiver) and its two-thread split timed as two kernels (deferred-a 41, deferred-b 35 accumulators);
# - w': split (each FACTORED u / v sum of 4+ terms in two partial sums: are the multiply-add chains what it waits on).
# Environment: OUT (the job sets it), EXPECT_GPU (default H200), PARTS (default "lab rattle").
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$real" > "$out/gpu.txt"
if ! echo "$real" | grep -qE "${EXPECT_GPU:-H200}"; then note "WRONG-GPU $real"; exit 3; fi
ck=${CHECKPOINT:-$FASTMACE_MODELS/mace-omat-0-medium.model}
note "round2_lab_job on $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"
for part in ${PARTS:-lab rattle}; do
  extra=""
  [ "$part" = rattle ] && extra="--rattle 0.3 --equilibrate-steps 20"
  mkdir -p "$out/$part"
  # shellcheck disable=SC2086
  python tools/spline_table_bound.py --checkpoint "$ck" --output "$out/$part" --rounds 31 \
    --table-reads packed8 packed8v2 deferred deferred-a deferred-b --wprime-zero split $extra \
    > "$out/$part/bound.log" 2>&1 < /dev/null
  note "end $part exit=$? $(grep -hE '^(forward|wprime)|registers|against real' "$out/$part/bound.log" | tr '\n' ' ' | cut -c1-2500)"
done
note done
