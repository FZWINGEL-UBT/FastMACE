#!/usr/bin/env bash
# fused_forward.CG_FOLD (H200 ideas plan idea 24, round 12, 6 November 2026): the force/stress gate of OMAT-0 medium
# (SrTiO3 3x3x3, NaCl, NbCl5) with the GPU's default plan against CG_FOLD=True, then the MD-step A/B in both build
# orders (hopper_sweep.sh's AB). Usage (Modal client venv):
#   ALIAS="NVIDIA H200" modal run tools/modal/fastmace_modal.py::job --gpu H200 --timeout-min 45 --commit <sha> \
#     --command "ALIAS='NVIDIA H200' bash tools/modal/cg_fold_check.sh" --out ~/fm-runs/cg-fold-h200
set -uo pipefail
out=${OUT:?the job sets OUT}
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
[ "$gpu" = "${ALIAS:?set ALIAS}" ] || { echo "WRONG-GPU: $gpu"; exit 3; }
ff=fastmace.kernels.oeq_streaming.fused_forward
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader); source $(cat .source_commit)"
O=$FASTMACE_MODELS/mace-omat-0-medium.model
S="results/loup-a100-20261103/SrTiO3.cif@3x3x3 NaCl NbCl5"
gate() {
  local name=$1; shift
  python tools/force_stress_gate.py --output "$out/gate-$name" --engine fastmace --model omat0-medium --checkpoint "$O" \
    --structures-root "$FASTMACE_STRUCTURES" --structures $S "$@" > "$out/gate-$name.log" 2>&1 < /dev/null
  note "gate-$name exit=$?"
}
gate default
gate cgfold --production-constant "$ff:CG_FOLD=True"
TESTS=0 GATE=0 LOUP=0 FORWARD_VARIANTS= WPRIME_VARIANTS= AB="cgfold|$ff:CG_FOLD=True" bash tools/modal/hopper_sweep.sh
note done
