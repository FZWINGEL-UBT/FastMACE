#!/usr/bin/env bash
# Species skip lab (tools/species_skip_lab.py) on the A100-SXM4-80GB: SpeciesSkip's product against rows grouped by
# species and the gather-free bound, with tf32x3_matmul (the A100 plan) and plain FP32 matmul.
# Exits 3 at once on any other GPU.
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 20 --commit <sha> \
#     --command "bash tools/modal/species_skip_lab.sh" --out ~/fm-runs/species-skip-lab-a100
set -uo pipefail
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q SXM4 || exit 3
out=${OUT:?the job sets OUT}
nvidia-smi --query-gpu=name --format=csv,noheader > "$out/gpu.txt"
python tools/species_skip_lab.py --tf32x3 --output "$out/tf32x3" > "$out/tf32x3.log" 2>&1
python tools/species_skip_lab.py --output "$out/fp32" > "$out/fp32.log" 2>&1
grep -H "^size" "$out"/*.log | tee "$out/summary.txt"
