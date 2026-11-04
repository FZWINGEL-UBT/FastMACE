#!/usr/bin/env bash
# A100-SXM4-80GB job of 4 November 2026: the species GEMM as one gathered CUTLASS 3xTF32 launch
# (species_gemm.CUTLASS, fm_tf32x3_species_rr_64x64), after the Triton (0.992) and our own mma.sync (1.003) versions.
# 1. GPU tests of species_gemm (CUTLASS cases);
# 2. tools/species_skip_lab.py --tf32x3: current, Triton, CUTLASS, bound;
# 3. MD step A/B default vs SPECIES_GEMM=True with CUTLASS=True, both orders, 8 blocks.
# Exits 3 at once on any other GPU.
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 50 --commit <sha> \
#     --command "bash tools/modal/species_gemm_cutlass_a100.sh" --out ~/fm-runs/species-gemm-cutlass-a100
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
python -m pytest -q -p no:cacheprovider tests/gpu/test_species_gemm_cuda.py > "$out/pytest-gpu.log" 2>&1
note "gpu tests exit=$? $(grep -E 'passed|failed' "$out/pytest-gpu.log" | tail -n 1)"
python tools/species_skip_lab.py --tf32x3 --tc "" --output "$out/lab-tf32x3" > "$out/lab-tf32x3.log" 2>&1
note "lab $(grep -h '^size' "$out/lab-tf32x3.log" | sed 's/diff.*//' | tr '\n' ' ')"
for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  note "start ab-cutlass-$order"
  python tools/ab_rewrites.py --output "$out/ab-cutlass-$order" --checkpoint "$ck" --model omat0-medium \
    --structures "$st" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 --warmup 200 --block 200 \
    --blocks 8 --order "$order" --pattern "$pattern" --b-constant "$m:SPECIES_GEMM=True" "$g:CUTLASS=True" \
    > "$out/ab-cutlass-$order.log" 2>&1 < /dev/null
  note "end ab-cutlass-$order exit=$? $(grep -h -E "b/a|Error" "$out/ab-cutlass-$order.log" | tail -n 2 | tr '\n' ' ' | cut -c1-300)"
done
note done
grep -h -E "tests|lab|end ab" "$log" | cut -c1-500
