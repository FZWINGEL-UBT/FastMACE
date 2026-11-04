#!/usr/bin/env bash
# A100-SXM4-80GB: wprime_backward.TABLE_W (w and its slope from the value table; the forward stores no w), 4 November
# 2026. The kernel lab bound (base, B-gslab-f2, F-no-wload, F-no-store), then the MD step A/B on Benchmark_loup in both
# orders, 8 blocks (omat0_a100_launch_screen.sh, case tablew). Exits 3 at once on any other GPU.
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 60 --commit <sha> \
#     --command "bash tools/modal/tablew_a100.sh" --out ~/fm-runs/tablew-a100
set -uo pipefail
out=${OUT:?the job sets OUT}
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
[ "$gpu" = "NVIDIA A100-SXM4-80GB" ] || { echo "not the SXM4 card: $gpu"; exit 3; }
echo "$(date -u +%FT%TZ) lab start" >> "$out/run.log"
python tools/omat0_kernel_lab.py --checkpoint "$FASTMACE_MODELS/mace-omat-0-medium.model" \
  --variants base B-gslab-f2 F-no-wload F-no-store --output "$out/lab" > "$out/lab.log" 2>&1 < /dev/null
echo "$(date -u +%FT%TZ) lab exit=$? $(grep -h -E "^(wprime|forward) " "$out/lab.log" | tr '\n' ';' | cut -c1-600)" \
  >> "$out/run.log"
SCREEN=tablew SCREEN_BOTH=1 bash tools/modal/omat0_a100_launch_screen.sh
