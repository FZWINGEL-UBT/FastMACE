#!/usr/bin/env bash
# packed8 (8-byte SPLINE_W forward table) on a disordered snapshot (9 November 2026): on SrTiO3 with 0.3 A Gaussian
# position noise the forward's table share is about 304 us on the H200 (crystal 191), so halving the table bytes may
# pay there although it lost on the crystal (1.060). Environment: OUT (the job sets it), EXPECT_GPU (default H200).
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$real" > "$out/gpu.txt"
if ! echo "$real" | grep -qE "${EXPECT_GPU:-H200}"; then note "WRONG-GPU $real"; exit 3; fi
ck=${CHECKPOINT:-$FASTMACE_MODELS/mace-omat-0-medium.model}
note "packed8_rattle_job on $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"
python tools/spline_table_bound.py --checkpoint "$ck" --output "$out" --rounds 31 --table-reads packed8 \
  --rattle 0.3 --equilibrate-steps 20 > "$out/bound.log" 2>&1 < /dev/null
note "end bound exit=$? $(grep -hE '^forward' "$out/bound.log" | tr '\n' ' ' | cut -c1-1200)"
