#!/usr/bin/env bash
# Where the wprime backward's time goes on the A100-SXM4-80GB at the degree-5 tables (4 November 2026; it is 5.2 ms,
# 35 % of the Benchmark_loup force call, once the forward runs on tensor cores): the kernel lab's wprime kernel with
# the bitwise variants and the falsifiers. Exits 3 on any other card.
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 30 --commit <sha> \
#     --command "bash tools/modal/omat0_a100_wprime_lab.sh" --out ~/fm-runs/omat0-a100-wprime-lab
set -uo pipefail
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q SXM4 || exit 3
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
nvidia-smi --query-gpu=name --format=csv,noheader > "$out/gpu.txt"
python tools/omat0_kernel_lab.py --checkpoint "$ck" --kernels wprime \
  --variants base ldcs-w unordered F-no-slope F-half-terms F-one-slab F-no-atomics F-no-wload \
  --output "$out/wprime" > "$out/wprime.log" 2>&1
echo "wprime exit=$?" >> "$out/summary.txt"
grep -h -E "^wprime " "$out/wprime.log" | cut -c1-200 | tee -a "$out/summary.txt"
