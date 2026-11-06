#!/usr/bin/env bash
# The x gathers against the atom order and the lab's L2 flush (round 9, 6 November 2026). The fused forward's lab
# (base, F-x-row0) on SrTiO3 10x10x10 in ASE repeat, Morton and seeded random order
# (results/round4-staging-20261105/orders), each with the lab's 256 MB L2 flush between launches and without it (in MD,
# x was just written by the previous kernel and is L2-resident).
# Usage: modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 30 --commit <sha> \
#   --command "ALIAS='NVIDIA A100-SXM4-80GB' bash tools/modal/order_lab.sh" --out ~/fm-runs/<tag>
set -uo pipefail
out=${OUT:?the job sets OUT}
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
export PYTHONPATH=$PWD/tools/bench/gpu_alias:${PYTHONPATH:-} FASTMACE_GPU_ALIAS="$real=${ALIAS:-$real}"
log=$out/run.log
echo "$(date -u +%FT%TZ) Modal $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); alias $FASTMACE_GPU_ALIAS; source $(cat .source_commit 2>/dev/null)" >> "$log"
for order in repeat morton random; do
  for flush in 256 0; do
    tag=$order-flush$flush
    python tools/omat0_kernel_lab.py --checkpoint "$FASTMACE_MODELS/mace-omat-0-medium.model" --kernels forward \
      --structure "results/round4-staging-20261105/orders/SrTiO3-10x10x10-$order.extxyz@1x1x1" --flush-l2-mb "$flush" \
      --rounds 5 --variants base F-x-row0 --output "$out/lab-$tag" > "$out/$tag.log" 2>&1 < /dev/null
    echo "$(date -u +%FT%TZ) $tag exit=$?" >> "$log"
    grep -h -E "^forward  " "$out/$tag.log" | sed "s/^/$tag /" >> "$log"
  done
done
