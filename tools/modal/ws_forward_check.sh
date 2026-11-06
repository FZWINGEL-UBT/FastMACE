#!/usr/bin/env bash
# fused_forward.WS_FORWARD (H200 ideas plan idea 30, 6 November 2026): on an H200,
# - LAB_VARIANTS (optional): the kernel lab's forward variants (tools/lab_idea30.py);
# - TEST=1: tests/gpu/test_ws_forward_cuda.py (sm_90 only);
# - GATE=1: the force/stress gate of OMAT-0 medium (SrTiO3 3x3x3, NaCl, NbCl5), the GPU's default plan against
#   WS_FORWARD=True, in this job (cg_fold_check.sh's pattern);
# - AB=1: the MD-step A/B in both build orders (hopper_sweep.sh's AB).
# Usage (Modal client venv):
#   ALIAS="NVIDIA H200" modal run tools/modal/fastmace_modal.py::job --gpu H200 --timeout-min 45 --commit <sha> \
#     --command "ALIAS='NVIDIA H200' TEST=1 GATE=1 AB=1 bash tools/modal/ws_forward_check.sh" --out ~/fm-runs/ws-h200
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
if [ -n "${LAB_VARIANTS:-}" ]; then
  TESTS=0 GATE=0 LOUP=0 AB= WPRIME_VARIANTS= FORWARD_VARIANTS="$LAB_VARIANTS" bash tools/modal/hopper_sweep.sh
fi
if [ "${TEST:-0}" = 1 ]; then
  python -m pytest -p no:cacheprovider -q -rs -s tests/gpu/test_ws_forward_cuda.py > "$out/test.log" 2>&1 < /dev/null
  note "test exit=$? $(tail -n 1 "$out/test.log" | cut -c1-200)"
fi
S="results/loup-a100-20261103/SrTiO3.cif@3x3x3 NaCl NbCl5"
gate() {
  local name=$1; shift
  python tools/force_stress_gate.py --output "$out/gate-$name" --engine fastmace --model omat0-medium --checkpoint "$O" \
    --structures-root "$FASTMACE_STRUCTURES" --structures $S "$@" > "$out/gate-$name.log" 2>&1 < /dev/null
  note "gate-$name exit=$?"
}
if [ "${GATE:-0}" = 1 ]; then
  gate default
  gate ws --production-constant "$ff:WS_FORWARD=True"
fi
if [ "${AB:-0}" = 1 ]; then
  TESTS=0 GATE=0 LOUP=0 FORWARD_VARIANTS= WPRIME_VARIANTS= AB="ws|$ff:WS_FORWARD=True" bash tools/modal/hopper_sweep.sh
fi
note done
