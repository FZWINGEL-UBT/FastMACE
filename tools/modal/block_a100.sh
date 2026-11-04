#!/usr/bin/env bash
# A100-SXM4-80GB: wprime_backward.BLOCK_MESSAGE (the conv message in linear_1's block layout) against the default
# (TABLE_W on), 4 November 2026: the MD step A/B on Benchmark_loup in both orders, 8 blocks
# (omat0_a100_launch_screen.sh, case block). Exits 3 at once on any other GPU.
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 60 --commit <sha> \
#     --command "bash tools/modal/block_a100.sh" --out ~/fm-runs/block-a100
set -uo pipefail
out=${OUT:?the job sets OUT}
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
[ "$gpu" = "NVIDIA A100-SXM4-80GB" ] || { echo "not the SXM4 card: $gpu"; exit 3; }
echo "$(date -u +%FT%TZ) block screen start" >> "$out/run.log"
SCREEN=block SCREEN_BOTH=1 bash tools/modal/omat0_a100_launch_screen.sh
