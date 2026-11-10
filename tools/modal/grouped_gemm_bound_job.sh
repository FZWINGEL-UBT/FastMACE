#!/usr/bin/env bash
# The grouped-GEMM bound (tools/grouped_gemm_bound.py, 9 November 2026) on the H200: four per-path CUTLASS 3xTF32
# launches against one launch of the same multiply-adds, per product group of OMAT-0 medium on 5,000 atoms.
# Environment: OUT (the job sets it), EXPECT_GPU (default H200).
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$real" > "$out/gpu.txt"
if ! echo "$real" | grep -qE "${EXPECT_GPU:-H200}"; then note "WRONG-GPU $real"; exit 3; fi
note "grouped_gemm_bound_job on $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"
python tools/grouped_gemm_bound.py --output "$out" --rounds 31 > "$out/bound.log" 2>&1 < /dev/null
note "end bound exit=$? $(tr '\n' ' ' < "$out/bound.log" | cut -c1-900)"
