#!/usr/bin/env bash
# The spline table-traffic bound (tools/spline_table_bound.py, 8 November 2026) on the H200 defaults: the SPLINE_W
# forward and w' with their real inputs against every edge on one table slab. Environment: OUT (the job sets it),
# EXPECT_GPU (default H200).
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$real" > "$out/gpu.txt"
if ! echo "$real" | grep -qE "${EXPECT_GPU:-H200}"; then note "WRONG-GPU $real"; exit 3; fi
ck=${CHECKPOINT:-$FASTMACE_MODELS/mace-omat-0-medium.model}
note "spline_table_bound_job on $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"
python tools/spline_table_bound.py --checkpoint "$ck" --output "$out" --rounds 31 > "$out/bound.log" 2>&1 < /dev/null
note "end bound exit=$? $(grep -hE '^(forward|wprime):' "$out/bound.log" | tr '\n' ' ' | cut -c1-600)"
