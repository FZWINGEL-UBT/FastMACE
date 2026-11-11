#!/usr/bin/env bash
# The SPLINE_W forward's table-read lab (tools/spline_read_lab.py, 9 November 2026) on the H200 defaults: the real
# forward against one-slab and the 8-byte packed table (packed8). Environment: OUT (the job sets it), EXPECT_GPU
# (default H200), READS (default packed8), WPRIME (w' lab forms for --wprime-zero, default none).
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$real" > "$out/gpu.txt"
if ! echo "$real" | grep -qE "${EXPECT_GPU:-H200}"; then note "WRONG-GPU $real"; exit 3; fi
ck=${CHECKPOINT:-$FASTMACE_MODELS/mace-omat-0-medium.model}
note "spline_read_lab_job on $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"
# shellcheck disable=SC2086
python tools/spline_table_bound.py --checkpoint "$ck" --output "$out" --rounds 31 --table-reads ${READS:-packed8} \
  --wprime-zero ${WPRIME:-} \
  > "$out/bound.log" 2>&1 < /dev/null
note "end bound exit=$? $(grep -hE '^(forward|wprime)' "$out/bound.log" | tr '\n' ' ' | cut -c1-900)"
