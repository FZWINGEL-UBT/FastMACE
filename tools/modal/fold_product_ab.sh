#!/usr/bin/env bash
# generic_fold_product_linear (fold_product_linear on MACE-OMAT-0 medium, 4 November 2026): the force/stress gate with
# the fold on, then tools/ab_rewrites.py MD-step A/Bs in both build orders, side b with the fold (side a the default
# plan, where decision 66 keeps it off), SrTiO3 10x10x10 (5,000 atoms, Benchmark_loup) and NbCl5. Any GPU; the result
# names it. On a workstation: PY=<environment python> OUT=<directory> bash tools/modal/fold_product_ab.sh
# On Modal (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 60 --commit <sha> \
#     --command "bash tools/modal/fold_product_ab.sh" --out ~/fm-runs/fold-product-a100
set -uo pipefail
out=${OUT:?set OUT}
py=${PY:-python}
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "$(hostname) $gpu; source $(cat .source_commit 2>/dev/null || git rev-parse --short HEAD 2>/dev/null)"
"$py" tools/force_stress_gate.py --output "$out/gate-fold" --engine fastmace --model omat0-medium --checkpoint "$ck" \
  --structures-root "$FASTMACE_STRUCTURES" --structures NaCl NbCl5 "$st@3x3x3" --rewrites fold_product_linear=on \
  > "$out/gate-fold.log" 2>&1 < /dev/null
note "gate-fold exit=$? $(grep -h -iE 'verdict|pass|fail' "$out/gate-fold.log" | tail -n 4 | tr '\n' ' ' | cut -c1-400)"
for structure in "$st@10x10x10" NbCl5; do
  tag=$(echo "$structure" | sed 's#.*/##; s#\.cif##; s#@#-#')
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    note "start ab-$tag-$order"
    "$py" tools/ab_rewrites.py --output "$out/ab-$tag-$order" --checkpoint "$ck" --model omat0-medium \
      --structures "$structure" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 --warmup 200 \
      --block 200 --blocks 8 --order "$order" --pattern "$pattern" --b fold_product_linear=on \
      > "$out/ab-$tag-$order.log" 2>&1 < /dev/null
    note "end ab-$tag-$order exit=$? $(grep -h -E "b/a|Error" "$out/ab-$tag-$order.log" | tail -n 2 | tr '\n' ' ' | cut -c1-300)"
  done
done
note done
grep -h -E "gate-fold|end ab" "$log" | cut -c1-500
