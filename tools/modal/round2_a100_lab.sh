#!/usr/bin/env bash
# Round-2 kernel screens on the A100-SXM4-80GB (5 November 2026, plans/ideas-20261105-round2.md): the default plan's
# kernels from the kernel lab, against variants that are not in any plan yet.
# - forward: base (the A100 default: 3xTF32 tensor cores, mask split), tc-fp16 (exponent-shifted 3xFP16 m16n8k16),
#   tc-fp16-1pass (its big*big product alone: the bound);
# - wprime: base (FACTORED, TABLE_W), Horner (power basis) with and without the l1 = 0 scalar paths, capped at 80
#   registers or not, and factored-maxreg80 (the old cap);
# - eos: tools/eos_backward_lab.py (edge_outer_sum backward: base, the run walk, falsifiers).
# ptxas registers and spills for sm_80 with every lab variant. Exits 3 at once on any other GPU unless ALLOW_ANY_GPU=1
# (the RTX 4000 Ada on leto: FORWARD_CONSTANTS="fastmace.kernels.oeq_streaming.fused_forward:TENSOR_CORES=True" puts
# the forward lab on tensor cores, and the default SIMT forward is timed first as forward-simt).
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 40 --commit <sha> \
#     --command "bash tools/modal/round2_a100_lab.sh" --out ~/fm-runs/round2-a100-lab
set -uo pipefail
[ "${ALLOW_ANY_GPU:-0}" = 1 ] || nvidia-smi --query-gpu=name --format=csv,noheader | grep -q SXM4 || exit 3
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$out/gpu.txt"
fc=()
if [ -n "${FORWARD_CONSTANTS:-}" ]; then
  python tools/omat0_kernel_lab.py --checkpoint "$ck" --kernels forward --rounds 21 --variants base \
    --output "$out/forward-simt" > "$out/forward-simt.log" 2>&1
  echo "forward-simt exit=$?" >> "$out/summary.txt"
  read -r -a fc <<< "--constant $FORWARD_CONSTANTS"
fi
python tools/omat0_kernel_lab.py --checkpoint "$ck" --kernels forward --rounds 21 --ptxas-archs 8.0 8.9 "${fc[@]}" \
  --variants base tc-fp16 tc-fp16-1pass --output "$out/forward" > "$out/forward.log" 2>&1
echo "forward exit=$?" >> "$out/summary.txt"
python tools/omat0_kernel_lab.py --checkpoint "$ck" --kernels wprime --rounds 21 --ptxas-archs 8.0 8.9 \
  --variants base horner scalar-l1 horner+scalar-l1 horner-maxreg80 horner+scalar-l1-maxreg80 factored-maxreg80 \
  --output "$out/wprime" > "$out/wprime.log" 2>&1
echo "wprime exit=$?" >> "$out/summary.txt"
python tools/eos_backward_lab.py --rounds 51 --output "$out/eos" > "$out/eos.log" 2>&1
echo "eos exit=$?" >> "$out/summary.txt"
grep -h -E "^(forward|wprime) |rule|ptxas" "$out"/forward*.log "$out/wprime.log" | cut -c1-240 | tee -a "$out/summary.txt"
grep -v -i warn "$out/eos.log" | tail -20 | tee -a "$out/summary.txt"
