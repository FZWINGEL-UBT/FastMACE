#!/usr/bin/env bash
# Where the 3xTF32 tensor-core forward loses on the A100 (4 November 2026): the kernel lab's forward kernel captured
# from the default plan (FP32 SIMT product) and from the plan with fused_forward.TENSOR_CORES, with falsifiers:
# F-tc-1pass (one mma instead of three), F-tc-no-mma (no mma), F-tc-no-wsplit (W fragments unsplit),
# F-tc-no-redistribute (w read back from the accumulators, not the shared tile), F-tc-no-product (no k-loop).
# Since 4 November 2026 (TC_SPLIT): only the bases, the SIMT product and the splits rna, round and mask.
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 30 --commit <sha> \
#     --command "bash tools/modal/omat0_a100_tc_lab.sh" --out ~/fm-runs/omat0-a100-tc-lab
set -uo pipefail
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q SXM4 || exit 3
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
nvidia-smi --query-gpu=name --format=csv,noheader > "$out/gpu.txt"
python tools/omat0_kernel_lab.py --checkpoint "$ck" --kernels forward \
  --variants base --output "$out/simt" > "$out/simt.log" 2>&1
echo "simt exit=$?" >> "$out/summary.txt"
ff=fastmace.kernels.oeq_streaming.fused_forward
for split in rna round mask; do
  python tools/omat0_kernel_lab.py --checkpoint "$ck" --kernels forward \
    --constant $ff:TENSOR_CORES=True "$ff:TC_SPLIT='$split'" --variants base \
    --output "$out/tc-$split" > "$out/tc-$split.log" 2>&1
  echo "tc-$split exit=$?" >> "$out/summary.txt"
done
grep -H -E "^forward " "$out"/*.log | cut -c1-200 | tee -a "$out/summary.txt"
