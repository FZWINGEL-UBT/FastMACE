#!/usr/bin/env bash
# SpeciesSkip with the grouped species GEMM (modules.SPECIES_GEMM, 4 November 2026) on the A100-SXM4-80GB, on top of
# the A100 defaults (tensor-core forward, tf32x3_gemms, the generic wprime set, the skipped zero column):
# 1. MD step A/B default vs SPECIES_GEMM=True, both orders, 8 blocks (Benchmark_loup, OMAT-0 medium);
# 2. tools/bench/loup_md.py, NVE and Langevin, 2 processes each, alternating default / species GEMM.
# Exits 3 at once on any other GPU.
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 60 --commit <sha> \
#     --command "bash tools/modal/species_gemm_a100.sh" --out ~/fm-runs/species-gemm-a100
set -uo pipefail
out=${OUT:?the job sets OUT}
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
[ "$gpu" = "NVIDIA A100-SXM4-80GB" ] || { echo "not the SXM4 card: $gpu"; exit 3; }
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif@10x10x10
sg=fastmace.rewrites.generic.modules:SPECIES_GEMM=True
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "$(hostname) $gpu; source $(cat .source_commit)"
for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  note "start ab-sg-$order"
  python tools/ab_rewrites.py --output "$out/ab-sg-$order" --checkpoint "$ck" --model omat0-medium \
    --structures "$st" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 --warmup 200 --block 200 \
    --blocks 8 --order "$order" --pattern "$pattern" --b-constant "$sg" > "$out/ab-sg-$order.log" 2>&1 < /dev/null
  note "end ab-sg-$order exit=$? $(grep -h -E "b/a|Error" "$out/ab-sg-$order.log" | tail -n 2 | tr '\n' ' ' | cut -c1-300)"
done
mkdir -p "$out/loup"
for t in 1 2; do
  for ens in nve langevin; do
    for set in default sg; do
      c=()
      [ "$set" = sg ] && c+=(--constant "$sg")
      note "start loup-$set-$ens-$t"
      python tools/bench/loup_md.py --structure "$st" --checkpoint "$ck" --ensemble "$ens" "${c[@]}" \
        --output "$out/loup/$set-$ens-$t.json" > "$out/loup-$set-$ens-$t.log" 2>&1 < /dev/null
      note "end loup-$set-$ens-$t exit=$? $(tail -n 1 "$out/loup-$set-$ens-$t.log" | cut -c1-300)"
    done
  done
done
note done
grep -h -E "end (ab|loup)" "$log" | cut -c1-260
