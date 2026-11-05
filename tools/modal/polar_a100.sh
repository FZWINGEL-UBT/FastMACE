#!/usr/bin/env bash
# POLAR-1-M on the A100-SXM4-80GB (polar-T01, docs/validation/polar-reuse.md): the force gate with MH-1's A100
# label-gated set (wprime_backward, fused_folded_forward, tf32x3_gemms) and the round-5 switches, then MD-step
# A/Bs in both build orders (tools/polar_reuse_ab.sh). Exits 3 on any other card.
# Usage (from WSL, Modal client venv; the checkpoint in the volume's /models):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 150 --commit <sha> \
#     --command "bash tools/modal/polar_a100.sh" --out ~/fm-runs/polar-a100
set -uo pipefail
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q SXM4 || exit 3
C="fastmace.rewrites.generic.modules:EDGE_SH=True fastmace.kernels.oeq_streaming.wprime_backward:TABLE_W=True"
C="$C fastmace.rewrites.generic.modules:FUSED_FIRST_BACKWARD=True"
MODEL=polar1-m CKPT=MACE-POLAR-1-M.model GATE_REWRITES="wprime_backward=on fused_folded_forward=on tf32x3_gemms=on" \
  GATE_CONSTANTS="$C" BASE="wprime_backward=on fused_folded_forward=on tf32x3_gemms=on" \
  SETS="pair pairtf epilogue species edgesh tablew fusedfirst" bash tools/polar_reuse_ab.sh
grep -h -E "end |done" "$OUT/run.log"
