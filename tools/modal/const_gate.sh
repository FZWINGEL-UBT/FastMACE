#!/usr/bin/env bash
# The force/stress gate of OMAT-0 medium (SrTiO3 3x3x3, NaCl, NbCl5) for module constants (decision 66's rounding
# rule: no worse than the GPU's default): the default plan and side b (CONST, space-separated MODULE:NAME=VALUE) in
# this job, as ws_forward_check.sh. ALIAS: the exact device name the job must land on.
# Usage (Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 60 --commit <sha> \
#     --command "ALIAS='NVIDIA A100-SXM4-80GB' NAME=estrin CONST='m:ESTRIN=True' bash tools/modal/const_gate.sh" ...
set -uo pipefail
out=${OUT:?the job sets OUT}
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
[ "$gpu" = "${ALIAS:?set ALIAS}" ] || { echo "WRONG-GPU: $gpu"; exit 3; }
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader); source $(cat .source_commit)"
O=$FASTMACE_MODELS/mace-omat-0-medium.model
S="results/loup-a100-20261103/SrTiO3.cif@3x3x3 NaCl NbCl5"
gate() {
  local name=$1; shift
  python tools/force_stress_gate.py --output "$out/gate-$name" --engine fastmace --model omat0-medium --checkpoint "$O" \
    --structures-root "$FASTMACE_STRUCTURES" --structures $S "$@" > "$out/gate-$name.log" 2>&1 < /dev/null
  note "gate-$name exit=$? $(grep -h -iE 'verdict' "$out/gate-$name.log" | tail -n 1 | cut -c1-200)"
}
read -r -a consts <<< "${CONST:?set CONST}"
gate default
gate "${NAME:-b}" --production-constant "${consts[@]}"
note done
