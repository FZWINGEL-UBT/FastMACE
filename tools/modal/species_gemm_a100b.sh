#!/usr/bin/env bash
# Second A100-SXM4-80GB job of 4 November 2026 (species_gemm_a100.sh measured the first species GEMM 1.040 / 1.036):
# 1. tools/species_skip_lab.py with the Triton species GEMM (tf32x3 and FP32): kernel level against CUTLASS;
# 2. MD step A/B default vs SPECIES_GEMM=True (the count-based tiles), both orders, 8 blocks;
# 3. MD step A/B default vs FIRST_TABLE_COLUMNS=True, both orders, 8 blocks.
# Exits 3 at once on any other GPU.
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 60 --commit <sha> \
#     --command "bash tools/modal/species_gemm_a100b.sh" --out ~/fm-runs/species-gemm-a100b
set -uo pipefail
out=${OUT:?the job sets OUT}
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
[ "$gpu" = "NVIDIA A100-SXM4-80GB" ] || { echo "not the SXM4 card: $gpu"; exit 3; }
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif@10x10x10
m=fastmace.rewrites.generic.modules
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "$(hostname) $gpu; source $(cat .source_commit)"
python tools/species_skip_lab.py --tf32x3 --output "$out/lab-tf32x3" > "$out/lab-tf32x3.log" 2>&1
python tools/species_skip_lab.py --output "$out/lab-fp32" > "$out/lab-fp32.log" 2>&1
note "lab $(grep -h '^size' "$out"/lab-*.log | cut -c1-150 | tr '\n' ' ')"
for name in sg narrow; do
  c="$m:SPECIES_GEMM=True"
  [ "$name" = narrow ] && c="$m:FIRST_TABLE_COLUMNS=True"
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    note "start ab-$name-$order"
    python tools/ab_rewrites.py --output "$out/ab-$name-$order" --checkpoint "$ck" --model omat0-medium \
      --structures "$st" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 --warmup 200 --block 200 \
      --blocks 8 --order "$order" --pattern "$pattern" --b-constant "$c" > "$out/ab-$name-$order.log" 2>&1 < /dev/null
    note "end ab-$name-$order exit=$? $(grep -h -E "b/a|Error" "$out/ab-$name-$order.log" | tail -n 2 | tr '\n' ' ' | cut -c1-300)"
  done
done
note done
grep -h -E "lab|end ab" "$log" | cut -c1-400
