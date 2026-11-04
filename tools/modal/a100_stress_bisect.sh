#!/usr/bin/env bash
# A100-SXM4-80GB: which default of the A100 plan pushes OMAT-0 medium's NbCl5 stress over the gate's limit
# (docs/validation/omat0-tp-backward.md section 3.1: 1.79e-7 eV/A^3 against 1.16e-7), and the full gate of the new
# FACTORED + SYMMETRIC_PAIRS defaults, 4 November 2026. NbCl5 (OMAT-0) with one A100 default off at a time; the
# default twice (run-to-run scatter: OEq atomic sums, and the float32 yardstick's own scatter).
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 90 --commit <sha> \
#     --command "bash tools/modal/a100_stress_bisect.sh" --out ~/fm-runs/a100-stress-bisect
set -uo pipefail
out=${OUT:?the job sets OUT}
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
[ "$gpu" = "NVIDIA A100-SXM4-80GB" ] || { echo "not the SXM4 card: $gpu"; exit 3; }
wp=fastmace.kernels.oeq_streaming.wprime_backward
ff=fastmace.kernels.oeq_streaming.fused_forward
rd=fastmace.rewrites.radial
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
# Bisection on OMAT-0 NbCl5.
gate nb-default omat0-medium "$O" NbCl5
gate nb-default-2 omat0-medium "$O" NbCl5
gate nb-main omat0-medium "$O" NbCl5 --production-constant "$wp:FACTORED=False" "$rd:SYMMETRIC_PAIRS=False"
gate nb-no-tf32x3 omat0-medium "$O" NbCl5 --rewrites tf32x3_gemms=off
gate nb-no-tc omat0-medium "$O" NbCl5 --production-constant "$ff:TENSOR_CORES=False"
gate nb-no-tf32x3-tc omat0-medium "$O" NbCl5 --rewrites tf32x3_gemms=off --production-constant "$ff:TENSOR_CORES=False"
gate nb-no-tablew omat0-medium "$O" NbCl5 --production-constant "$wp:TABLE_W=False"
gate nb-no-block omat0-medium "$O" NbCl5 --production-constant "$wp:BLOCK_MESSAGE=False"
gate nb-degree10 omat0-medium "$O" NbCl5 --production-constant "$rd:DEGREE=10"
gate nb-no-tables omat0-medium "$O" NbCl5 --rewrites radial_tables=off
# The new defaults on the other structures.
gate omat0-rest omat0-medium "$O" "results/loup-a100-20261103/SrTiO3.cif@3x3x3 NaCl"
gate mh1 mh1-r2scan "$M" "NaCl NaNbO3 NbCl5"
note done
