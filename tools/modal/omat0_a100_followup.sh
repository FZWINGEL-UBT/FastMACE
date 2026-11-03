#!/usr/bin/env bash
# OMAT-0 A100 follow-up of the kernel lab (results/omat0-kernel-lab-20261103): the lab itself on this GPU (variants
# and falsifiers of the wprime backward and the fused forward; the falsifiers size the 3xTF32 tensor-core forward
# idea), then the screen of the lab's switches: one order for the single switches (SCREEN_ONE), both orders for the
# table degree and the bundle (SCREEN_TWO). Case names as in omat0_a100_launch_screen.sh; keep the two lists apart.
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 60 --commit <sha> \
#     --command "bash tools/modal/omat0_a100_followup.sh" --out ~/fm-runs/omat0-a100-followup
set -uo pipefail
out=${OUT:?the job sets OUT}
python tools/omat0_kernel_lab.py --checkpoint "$FASTMACE_MODELS/mace-omat-0-medium.model" --output "$out/lab" \
  > "$out/lab.log" 2>&1 < /dev/null
echo "exit=$?" >> "$out/lab.log"
SCREEN=${SCREEN_ONE:-"skipzero stcs slopes2 streamw sympairs strided deg6 d5-skipzero"} \
  bash tools/modal/omat0_a100_launch_screen.sh
SCREEN_BOTH=1 SCREEN=${SCREEN_TWO:-"deg5 all5"} bash tools/modal/omat0_a100_launch_screen.sh
