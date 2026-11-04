#!/usr/bin/env bash
# The W slice split into its TF32 parts once per block (fused_forward.TC_PRESPLIT, 4 November 2026) on the
# A100-SXM4-80GB: the kernel lab's forward kernel from the default plan (tensor cores, mask split), the pre-split
# variant re-rendered with its own launch rule (bitwise), and F-tc-no-wsplit (no W split at all: the bound).
# Exits 3 at once on any other GPU.
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 30 --commit <sha> \
#     --command "bash tools/modal/omat0_a100_presplit_lab.sh" --out ~/fm-runs/omat0-a100-presplit-lab
set -uo pipefail
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q SXM4 || exit 3
out=${OUT:?the job sets OUT}
nvidia-smi --query-gpu=name --format=csv,noheader > "$out/gpu.txt"
python tools/omat0_kernel_lab.py --checkpoint "$FASTMACE_MODELS/mace-omat-0-medium.model" --kernels forward \
  --rounds 21 --variants base tc-presplit F-tc-no-wsplit --output "$out/lab" > "$out/lab.log" 2>&1
echo "lab exit=$?" >> "$out/summary.txt"
grep -E "^forward |rule" "$out/lab.log" | cut -c1-240 | tee -a "$out/summary.txt"
