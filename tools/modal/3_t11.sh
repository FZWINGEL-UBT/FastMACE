#!/usr/bin/env bash
# 3-T11 acceptance item 3 on Modal: float32 MC energy differences against float64 for OMOL-0 XL (all-electron
# E0), whose build does not fit a local 15-minute slot and whose NbCl5 float64 reference does not fit 16 GB
# (2-T05). tools/validation/move_energy_precision.py on NaCl, NaNbO3 and NbCl5, plain cuEq float32, eager
# (still captured), superset {Na, Nb, O, Cl}, 300 K then 1500 K. Every record is labelled "Modal <GPU>".
# Usage (from WSL, Modal client venv; budget $3, guard in fastmace_modal.py):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100 --timeout-min 50 --commit <sha> \
#     --command "bash tools/modal/3_t11.sh" --out ~/fastmace-runs/3-t11/modal-A100
set -uo pipefail
out=${OUT:?the job sets OUT}
checkpoint=$FASTMACE_MODELS/MACE-omol-0-extra-large-1024.model
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
measure() {  # structure md-steps
  note "start $1 gpu=$(nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader)"
  FASTMACE_SOURCE_COMMIT=$(cat .source_commit) python tools/validation/move_energy_precision.py \
    --model omol0-xl --checkpoint "$checkpoint" --backend cueq --compile-mode none --md-steps "$2" \
    --structures "$1" --output "$out/omol0-xl-$1" > "$out/omol0-xl-$1.log" 2>&1 < /dev/null
  note "end $1 exit=$?"
}
note "Modal $(nvidia-smi --query-gpu=name --format=csv,noheader); source $(cat .source_commit) host $(hostname)"
measure NaCl 1000
measure NaNbO3 1000
measure NbCl5 500
note done
tail -n 40 "$out"/omol0-xl-*.log 2>/dev/null | grep -E '^\{|^ |^\}|Error' | cut -c1-200
