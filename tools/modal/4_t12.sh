#!/usr/bin/env bash
# 4-T12 on Modal: the PolarMACE stress floor of SPEC decision 49 on the GPU. (1) the POLAR-1-S gate cases of
# tests/gpu/test_selfcheck_gpu.py (NaCl 3x3x3 NVT, NPT hydrostatic and full; NbCl5 3x3x3 NVT and NPT; a 6e-7
# eV/A^3 stress error that must fail); (2) the 1b-T23 in-library gate rerun, POLAR-1-S on NaCl 3x3x3, NVT and
# NPT (tools/library_gate.py; 1b-T23's 1.2e-7 case ran on murphy, another GPU).
# Usage (from WSL, Modal client venv; guard in fastmace_modal.py):
#   modal run tools/modal/fastmace_modal.py::job --gpu L4 --timeout-min 30 --commit <sha> \
#     --command "bash tools/modal/4_t12.sh" --out ~/fastmace-runs/4-t12/modal-L4 --limit-usd <metered + 1.00>
set -uo pipefail
out=${OUT:?the job sets OUT}
export FASTMACE_SOURCE_COMMIT=$(cat .source_commit)
log=$out/run.log
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "Modal $gpu; source $FASTMACE_SOURCE_COMMIT host $(hostname)"
status=0
note "start pytest polar"
python -m pytest -s -p no:cacheprovider tests/gpu/test_selfcheck_gpu.py -k polar --junitxml "$out/junit-polar.xml" \
  > "$out/pytest-polar.log" 2>&1 < /dev/null
code=$?
note "end pytest polar exit=$code"
[ $code -eq 0 ] || status=$code
tail -n 3 "$out/pytest-polar.log"
note "start library gate"
python tools/library_gate.py --output "$out/library-gate-polar1-s" --model polar1-s --structure NaCl --check gate \
  > "$out/library-gate.log" 2>&1 < /dev/null
code=$?
note "end library gate exit=$code"
[ $code -eq 0 ] || status=$code
tail -n 3 "$out/library-gate.log"
note done
exit $status
