#!/usr/bin/env bash
# Wavefront group A/B of the tiled call: 5M SrTiO3, OMAT-0 medium, wavefront on, R = 8192, TileConfig.wavefront_group
# 1, 16, 16, 1 (one process each).
# Usage (from WSL, Modal client venv; --repo the main checkout):
#   modal run tools/modal/fastmace_modal.py::job --gpu RTX-PRO-6000 --timeout-min 50 --repo /mnt/d/Margraf/FastMACE \
#     --commit <sha> --command "bash tools/modal/tiled_group.sh" --out ~/fastmace-runs/tiled-group-20261102/modal
set -uo pipefail
out=${OUT:?the job sets OUT}
echo "$(date -u +%FT%TZ) Modal $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit)" \
  >> "$out/run.log"
args="--structure SrTiO3 --model omat0-medium --repeat 100x100x100 --mode auto --calls 3 --tile-list --host-state
  --alias-state --species-index --receivers 8192 --no-budget --wavefront on --lanes off"
for run in group1-a group16-a group16-b group1-b; do
  group=${run%-?}
  # shellcheck disable=SC2086
  python tools/bench/tiled_capacity.py $args --wavefront-group ${group#group} --output "$out/$run" > "$out/$run.log" 2>&1
  echo "$run exit $?" >> "$out/run.log"
done
