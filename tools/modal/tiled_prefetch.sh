#!/usr/bin/env bash
# Prefetch A/B of the tiled call (docs/validation/tiled-ideas-20261102.md): 1.08M SrTiO3, OMAT-0 medium, wavefront,
# R = 8192, TileConfig.prefetch off, on, on, off (one process each); then the backward chunk scale 4.
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu L4 --timeout-min 60 --commit <sha> \
#     --command "bash tools/modal/tiled_prefetch.sh" --out ~/fastmace-runs/tiled-prefetch-20261102/modal-L4
set -uo pipefail
out=${OUT:?the job sets OUT}
echo "$(date -u +%FT%TZ) Modal $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit)" \
  >> "$out/run.log"
args="--structure SrTiO3 --model omat0-medium --repeat 60x60x60 --mode auto --calls 4 --tile-list --host-state
  --alias-state --species-index --receivers 8192 --no-budget --wavefront on --lanes off"
for run in prefetch-off-a prefetch-on-a prefetch-on-b prefetch-off-b chunk4-on; do
  case $run in
    prefetch-off-*) extra="--prefetch off" ;;
    prefetch-on-*) extra="--prefetch on" ;;
    chunk4-on) extra="--prefetch on --backward-chunk-scale 4" ;;
  esac
  # shellcheck disable=SC2086
  python tools/bench/tiled_capacity.py $args $extra --output "$out/$run" > "$out/$run.log" 2>&1
  echo "$run exit $?" >> "$out/run.log"
done
