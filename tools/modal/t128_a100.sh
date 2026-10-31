#!/usr/bin/env bash
# The 128-row tile rerun (docs/validation/ab-constants-audit-20261030.md) on the Modal A100: 128 rows and 8 warps
# against 64 and 4 in segment_gemm's Triton products, on the row-first order and on the tile order's rule, with the
# fixed A/B tool (results/segment-gemm-order-20261030/tiles128.sh).
# Usage (from WSL, Modal client venv; MODAL_IMAGE_BUILDER_VERSION=2025.06; the guard in fastmace_modal.py):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100 --timeout-min 75 --commit <sha> \
#     --command "bash tools/modal/t128_a100.sh" --out ~/fastmace-runs/tiles128-20261031/modal-A100
set -uo pipefail
out=${OUT:?the job sets OUT}
export FASTMACE_HYBRID_PYTHON=$(command -v python)
prefix=$(dirname "$(dirname "$(command -v python)")")
echo "$(date -u +%FT%TZ) Modal $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit)" \
  >> "$out/run.log"
bash results/segment-gemm-order-20261030/tiles128.sh "$out" "$PWD" "$prefix" "$FASTMACE_MODELS/mace-mh-1.model" \
  "$FASTMACE_STRUCTURES"
