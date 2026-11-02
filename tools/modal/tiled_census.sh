#!/usr/bin/env bash
# Memory census of the tiled call (docs/validation/tiled-ideas-20261102.md): 5M SrTiO3, OMAT-0 medium, wavefront,
# R = 8192; the build's device peak by source line (--memory-history) and the live tensors after the build and every
# call (--ledger).
# Usage (from WSL, Modal client venv; --repo the main checkout):
#   modal run tools/modal/fastmace_modal.py::job --gpu RTX-PRO-6000 --timeout-min 40 --repo /mnt/d/Margraf/FastMACE \
#     --commit <sha> --command "bash tools/modal/tiled_census.sh" --out ~/fastmace-runs/tiled-census-20261103/modal
set -uo pipefail
out=${OUT:?the job sets OUT}
echo "$(date -u +%FT%TZ) Modal $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit)" \
  >> "$out/run.log"
python tools/bench/tiled_capacity.py --structure SrTiO3 --model omat0-medium --repeat 100x100x100 --mode auto \
  --calls 3 --tile-list --host-state --alias-state --species-index --receivers 8192 --no-budget --wavefront on \
  --lanes off --memory-history --ledger --output "$out/census" > "$out/census.log" 2>&1
echo "census exit $?" >> "$out/run.log"
