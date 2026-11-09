#!/usr/bin/env bash
# The bspline-lab lane's second H200 container (9 November 2026, report 5 follow-ups):
# 1. the kernel lab on SrTiO3 10x10x10: ghoist and one-slab+ghoist (the forward's instruction part), w' zero-byte
#    variants (table, dout, sender row, all) and the locality probes;
# 2. the same on a disordered snapshot (0.3 A Gaussian position noise, 20 MD steps): does the table cost more where
#    the neighbour shells are not sharp?
# 3. MD-step A/Bs (tools/modal/compiler_spill_ab.sh): w' capped at 64 / 80 registers with shared-memory spilling,
#    NVRTC 13.5 against 13.0.
# Environment: OUT (the job sets it), EXPECT_GPU (default H200), PARTS (default "lab rattle ab").
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$real" > "$out/gpu.txt"
if ! echo "$real" | grep -qE "${EXPECT_GPU:-H200}"; then note "WRONG-GPU $real"; exit 3; fi
ck=${CHECKPOINT:-$FASTMACE_MODELS/mace-omat-0-medium.model}
note "bspline_night2_job on $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"
for part in ${PARTS:-lab rattle ab}; do
  case $part in
    lab|rattle)
      extra=""
      [ "$part" = rattle ] && extra="--rattle 0.3 --equilibrate-steps 20"
      mkdir -p "$out/$part"
      # shellcheck disable=SC2086
      python tools/spline_table_bound.py --checkpoint "$ck" --output "$out/$part" --rounds 31 --table-reads ghoist \
        --wprime-zero table dout x all $extra > "$out/$part/bound.log" 2>&1 < /dev/null
      note "end $part exit=$? $(grep -hE '^(forward|wprime)' "$out/$part/bound.log" | tr '\n' ' ' | cut -c1-1600)" ;;
    ab)
      mkdir -p "$out/ab"
      OUT=$out/ab VARIANTS="${VARIANTS:-spill64 spill80 nvrtc}" bash tools/modal/compiler_spill_ab.sh
      note "end ab $(grep -h 'end ab' "$out"/ab/*/run.log | cut -c1-300 | tr '\n' ' ')" ;;
  esac
done
note done
