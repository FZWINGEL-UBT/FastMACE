#!/usr/bin/env bash
# Round 6 (5 November 2026): tools/modal/round2_ab.sh with blocks of 256 steps (each holds one Langevin noise refill, so
# the draw-ahead A/B is not aliased by the block length) and no gate unless SKIP_GATE=0: MD-step A/Bs (ab_rewrites.py)
# in both build orders, side a the default plan, side b with B_CONSTANTS, on SrTiO3 10x10x10 (5,000 atoms,
# Benchmark_loup) and NbCl5. Any GPU; the result names it. NAME labels the outputs. TEMPERATURE_K: default 300.
# On a workstation or the local PC: PY=<environment python> OUT=<directory> NAME=... B_CONSTANTS=... bash \
#   tools/modal/round6_ab256.sh
# On Modal (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 60 --commit <sha> \
#     --command "NAME=... B_CONSTANTS='...' bash tools/modal/round6_ab256.sh" --out ~/fm-runs/round6-ab-a100
set -uo pipefail
out=${OUT:?set OUT}
name=${NAME:?set NAME}
read -r -a bc <<< "${B_CONSTANTS:?set B_CONSTANTS}"
py=${PY:-python}
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "$(hostname) $gpu; source $(cat .source_commit 2>/dev/null || git rev-parse --short HEAD 2>/dev/null); $name: ${bc[*]}"
if [ "${SKIP_GATE:-1}" != 1 ]; then
  "$py" tools/force_stress_gate.py --output "$out/gate-$name" --engine fastmace --model omat0-medium \
    --checkpoint "$ck" --structures-root "$FASTMACE_STRUCTURES" --structures NaCl NbCl5 "$st@3x3x3" \
    --production-constant "${bc[@]}" > "$out/gate-$name.log" 2>&1 < /dev/null
  note "gate-$name exit=$? $(grep -h -iE 'verdict|pass|fail' "$out/gate-$name.log" | tail -n 4 | tr '\n' ' ' | cut -c1-400)"
fi
for structure in ${STRUCTURES:-"$st@10x10x10" NbCl5}; do
  tag=$(echo "$structure" | sed 's#.*/##; s#\.cif##; s#@#-#')
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    note "start ab-$name-$tag-$order"
    "$py" tools/ab_rewrites.py --output "$out/ab-$name-$tag-$order" --checkpoint "$ck" --model omat0-medium \
      --structures "$structure" --timestep-fs 1 --temperature-K "${TEMPERATURE_K:-300}" --a-neighbours skin=0.5 --b-neighbours skin=0.5 --warmup 200 \
      --block 256 --blocks 8 --order "$order" --pattern "$pattern" --b-constant "${bc[@]}" \
      > "$out/ab-$name-$tag-$order.log" 2>&1 < /dev/null
    note "end ab-$name-$tag-$order exit=$? $(grep -h -E "b/a|Error" "$out/ab-$name-$tag-$order.log" | tail -n 2 | tr '\n' ' ' | cut -c1-300)"
  done
done
note done
grep -h -E "gate-|end ab" "$log" | cut -c1-500
