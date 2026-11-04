#!/usr/bin/env bash
# A100-SXM4-80GB: force/stress gate of wprime_backward.FACTORED on the A100's default plan (OMAT-0 medium: SrTiO3
# 3x3x3, NaCl, NbCl5; MH-1 r2SCAN: NaCl, NaNbO3, NbCl5), default against FACTORED, 4 November 2026.
# Usage (Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 30 --commit <sha> \
#     --command "bash tools/modal/tp_backward_gate_a100.sh" --out ~/fm-runs/tp-backward-gate-a100
set -uo pipefail
out=${OUT:?the job sets OUT}
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
[ "$gpu" = "NVIDIA A100-SXM4-80GB" ] || { echo "not the SXM4 card: $gpu"; exit 3; }
wp=fastmace.kernels.oeq_streaming.wprime_backward
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader); source $(cat .source_commit)"
gate() {
  local name=$1 model=$2 ck=$3 structs=$4; shift 4
  python tools/force_stress_gate.py --output "$out/gate-$name" --engine fastmace --model "$model" --checkpoint "$ck" \
    --structures-root "$FASTMACE_STRUCTURES" --structures $structs "$@" > "$out/gate-$name.log" 2>&1 < /dev/null
  note "gate-$name exit=$?"
}
O=$FASTMACE_MODELS/mace-omat-0-medium.model; M=$FASTMACE_MODELS/mace-mh-1.model
gate omat0-default omat0-medium "$O" "results/loup-a100-20261103/SrTiO3.cif@3x3x3 NaCl NbCl5"
gate omat0-factored omat0-medium "$O" "results/loup-a100-20261103/SrTiO3.cif@3x3x3 NaCl NbCl5" \
  --production-constant "$wp:FACTORED=True"
gate mh1-default mh1-r2scan "$M" "NaCl NaNbO3 NbCl5"
gate mh1-factored mh1-r2scan "$M" "NaCl NaNbO3 NbCl5" --production-constant "$wp:FACTORED=True"
note done
