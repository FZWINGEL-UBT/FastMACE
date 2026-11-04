#!/usr/bin/env bash
# A100-SXM4-80GB: the a'-path bound of the wprime backward (tools/wprime_da_bound.py), 4 November 2026.
# Exits 3 at once on any other GPU.
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 15 --commit <sha> \
#     --command "bash tools/modal/wprime_da_bound_a100.sh" --out ~/fm-runs/wprime-da-bound-a100
set -uo pipefail
out=${OUT:?the job sets OUT}
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
[ "$gpu" = "NVIDIA A100-SXM4-80GB" ] || { echo "not the SXM4 card: $gpu"; exit 3; }
python tools/wprime_da_bound.py --output "$out/bound" > "$out/bound.log" 2>&1
echo "exit=$?" >> "$out/bound.log"
grep -E "^J |exit" "$out/bound.log"
