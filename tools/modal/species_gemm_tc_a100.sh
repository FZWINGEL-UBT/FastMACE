#!/usr/bin/env bash
# A100-SXM4-80GB job of 4 November 2026: the species GEMM in our own mma.sync 3xTF32 kernel
# (species_gemm.TENSOR_CORES), after Triton's tf32x3 version only tied CUTLASS there (958 vs 949 us, MD 0.992).
# 1. tools/species_skip_lab.py --tf32x3 with tensor-core variants (chunk 16/32/64, plain and presplit);
# 2. MD step A/B default vs SPECIES_GEMM=True with TENSOR_CORES=True and the lab's fastest variant, both orders,
#    8 blocks.
# Exits 3 at once on any other GPU.
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 50 --commit <sha> \
#     --command "bash tools/modal/species_gemm_tc_a100.sh" --out ~/fm-runs/species-gemm-tc-a100
set -uo pipefail
out=${OUT:?the job sets OUT}
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
[ "$gpu" = "NVIDIA A100-SXM4-80GB" ] || { echo "not the SXM4 card: $gpu"; exit 3; }
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif@10x10x10
m=fastmace.rewrites.generic.modules
g=fastmace.kernels.triton.species_gemm
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "$(hostname) $gpu; source $(cat .source_commit)"
python tools/species_skip_lab.py --tf32x3 --tc 16,32,64,16p,32p,64p --output "$out/lab-tf32x3" \
  > "$out/lab-tf32x3.log" 2>&1
note "lab $(grep -h '^size' "$out/lab-tf32x3.log" | sed 's/diff.*//' | tr '\n' ' ')"
best=$(python -c "
import json, sys
med = json.load(open(sys.argv[1]))['sizes']['16']['median_us']
v = min((t, n) for n, t in med.items() if n.startswith('tc'))[1][2:]
print(v.rstrip('p'), v.endswith('p'))" "$out/lab-tf32x3/species_skip_lab.json")
chunk=${best% *}; presplit=${best#* }
note "fastest tensor-core variant: chunk $chunk presplit $presplit"
for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  note "start ab-tc-$order"
  python tools/ab_rewrites.py --output "$out/ab-tc-$order" --checkpoint "$ck" --model omat0-medium \
    --structures "$st" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 --warmup 200 --block 200 \
    --blocks 8 --order "$order" --pattern "$pattern" --b-constant "$m:SPECIES_GEMM=True" "$g:TENSOR_CORES=True" \
    "$g:TC_CHUNK=$chunk" "$g:TC_PRESPLIT=$presplit" > "$out/ab-tc-$order.log" 2>&1 < /dev/null
  note "end ab-tc-$order exit=$? $(grep -h -E "b/a|Error" "$out/ab-tc-$order.log" | tail -n 2 | tr '\n' ' ' | cut -c1-300)"
done
note done
grep -h -E "lab|fastest|end ab" "$log" | cut -c1-500
