#!/usr/bin/env bash
# Round 5 on the A100-SXM4-80GB (5 November 2026, plans/ideas-20261105-round2.md):
# - forward lab: base (the A100 default, 3xTF32) against tc-hybrid16 (W_hi a_hi in TF32, both small products in FP16
#   m16n8k16: 4 MMAs per 16 k instead of 6);
# - tools/bench/rebuild_cost.py at skin 0.5 (the default) and 1.0: the neighbour-list rebuild's added time per step
#   (the profile's 11.35 ms per rebuild runs beside the force call).
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 40 --commit <sha> \
#     --command "bash tools/modal/round5_hybrid_rebuild.sh" --out ~/fm-runs/round5-a100-hybrid16
set -uo pipefail
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q SXM4 || exit 3
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif@10x10x10
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$out/gpu.txt"
python tools/omat0_kernel_lab.py --checkpoint "$ck" --kernels forward --rounds 21 --ptxas-archs 8.0 \
  --variants ${FORWARD_VARIANTS:-base tc-hybrid16} --output "$out/forward" > "$out/forward.log" 2>&1
echo "forward exit=$?" >> "$out/summary.txt"
for skin in 0.5 1.0; do
  python tools/bench/rebuild_cost.py --structure "$st" --checkpoint "$ck" --skin "$skin" \
    --output "$out/rebuild-skin$skin.json" > "$out/rebuild-skin$skin.log" 2>&1
  echo "rebuild skin $skin exit=$?" >> "$out/summary.txt"
  tail -1 "$out/rebuild-skin$skin.log" >> "$out/summary.txt"
done
grep -h -E "^forward |ptxas" "$out/forward.log" | cut -c1-240 | tee -a "$out/summary.txt"
