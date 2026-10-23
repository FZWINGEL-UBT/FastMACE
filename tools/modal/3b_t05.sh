#!/usr/bin/env bash
# 3b-T05, the ASE side of the NPT + D3 comparison (conductor's choice 7) on Modal: ASE LangevinBAOAB with MACE's
# MACECalculator (cuEq float32, MH-1 matpes_r2scan) plus torch-dftd (float64), NbCl5 1x1x3, 0 GPa, 1 fs, P_tau 2 ps
# (tools/validation/npt_d3_comparison.py, SETTINGS). Each argument is one run, TEMPERATURE:SEED, run in order;
# EQUILIBRATION_PS and PRODUCTION_PS override the 5 + 20 ps (a calibration job: 0 and 1).
# Usage (from WSL, Modal client venv; guard in fastmace_modal.py; one container per run or two):
#   modal run tools/modal/fastmace_modal.py::job --gpu L4 --timeout-min 120 --commit <sha> \
#     --command "bash tools/modal/3b_t05.sh 300:101" --out ~/fastmace-runs/3b-t05/modal-L4-300K-s101
#   calibration: --command "EQUILIBRATION_PS=0 PRODUCTION_PS=1 bash tools/modal/3b_t05.sh 600:100" --timeout-min 20
set -uo pipefail
out=${OUT:?the job sets OUT}
checkpoint=$FASTMACE_MODELS/mace-mh-1.model
export FASTMACE_SOURCE_COMMIT=$(cat .source_commit)
log=$out/run.log
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "Modal $gpu; source $FASTMACE_SOURCE_COMMIT host $(hostname)"
status=0
for run in "$@"; do
  temperature=${run%%:*}
  seed=${run##*:}
  name=ase-${temperature}K-s${seed}
  note "start $name"
  python tools/validation/npt_d3_comparison.py run-ase --temperature-K "$temperature" --seed "$seed" \
    --output "$out/$name" --checkpoint "$checkpoint" --machine "Modal $gpu" \
    --equilibration-ps "${EQUILIBRATION_PS:-5}" --production-ps "${PRODUCTION_PS:-20}" \
    > "$out/$name.log" 2>&1 < /dev/null
  code=$?
  note "end $name exit=$code"
  [ $code -eq 0 ] || status=$code
  tail -n 1 "$out/$name.log"
done
note done
exit $status
