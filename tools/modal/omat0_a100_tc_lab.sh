#!/usr/bin/env bash
# Where the 3xTF32 tensor-core forward loses on the A100 (4 November 2026): the kernel lab's forward kernel captured
# from the default plan (FP32 SIMT product) and from the plan with fused_forward.TENSOR_CORES, with falsifiers:
# F-tc-1pass (one mma instead of three), F-tc-no-mma (no mma), F-tc-no-wsplit (W fragments unsplit),
# F-tc-no-redistribute (w read back from the accumulators, not the shared tile), F-tc-no-product (no k-loop).
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 30 --commit <sha> \
#     --command "bash tools/modal/omat0_a100_tc_lab.sh" --out ~/fm-runs/omat0-a100-tc-lab
set -uo pipefail
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
nvidia-smi --query-gpu=name --format=csv,noheader > "$out/gpu.txt"
python tools/omat0_kernel_lab.py --checkpoint "$ck" --kernels forward \
  --variants base stcs F-no-store F-no-gemm F-no-tp F-gemm-only --output "$out/simt" > "$out/simt.log" 2>&1
echo "simt exit=$?" >> "$out/summary.txt"
python tools/omat0_kernel_lab.py --checkpoint "$ck" --kernels forward \
  --constant fastmace.kernels.oeq_streaming.fused_forward:TENSOR_CORES=True \
  --variants base F-no-store F-no-tp F-tc-1pass F-tc-no-mma F-tc-no-wsplit F-tc-no-redistribute F-tc-no-product \
  --output "$out/tc" > "$out/tc.log" 2>&1
echo "tc exit=$?" >> "$out/summary.txt"
grep -h -E "^forward " "$out/simt.log" "$out/tc.log" | cut -c1-200 | tee -a "$out/summary.txt"
