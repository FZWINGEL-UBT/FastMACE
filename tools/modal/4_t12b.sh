#!/usr/bin/env bash
# 4-T12b on Modal: tools/validation/polar_cell_probe.py's reference part (the float64 reference's forces at
# 4-T07's PolarMACE start, NbCl5 3x3x3) and its fixed-cell part (ASE LBFGS on fm.calc without the cell filter,
# and MH-1's forces at the same start), each in its own process.
# Usage (from WSL, Modal client venv; guard in fastmace_modal.py):
#   modal run tools/modal/fastmace_modal.py::job --gpu L4 --timeout-min 20 --commit <sha> \
#     --command "bash tools/modal/4_t12b.sh" --out ~/fastmace-runs/4-t12b/modal-L4 --limit-usd <limit>
set -uo pipefail
out=${OUT:?the job sets OUT}
export FASTMACE_SOURCE_COMMIT=$(cat .source_commit)
log=$out/run.log
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "Modal $gpu; source $FASTMACE_SOURCE_COMMIT host $(hostname)"
status=0
for part in reference fixed-cell; do
  note "start $part"
  python tools/validation/polar_cell_probe.py --part "$part" --output "$out/$part.json" > "$out/$part.log" 2>&1 < /dev/null
  code=$?
  note "end $part exit=$code"
  [ $code -eq 0 ] || status=$code
  tail -n 12 "$out/$part.log"
done
note done
exit $status
