#!/usr/bin/env bash
# generic_fused_folded_forward and generic_wprime_backward (3 November 2026), OMAT-0 medium: the force gate with both
# on (NaCl, NbCl5), then same-process MD-step A/Bs in both build orders on the Benchmark_loup SrTiO3 cell (5,000
# atoms, 1 fs, skin 0.5): b = a + fused forward, + wprime, + both. CHUNKED=on adds chunked_conv_backward=on to every
# b (the A100, where decision 66 leaves the generic chunked path off; wprime and the fused forward need it).
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 70 --commit <sha> \
#     --command "CHUNKED=on bash tools/modal/generic_wprime_ab.sh" --out ~/fm-runs/generic-wprime-a100
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
omat=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif@10x10x10
extra=()
[ "${CHUNKED:-}" = on ] && extra=(chunked_conv_backward=on)
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
job() {
  local name=$1; shift
  note "start $name"
  "$@" > "$out/$name.log" 2>&1 < /dev/null
  local code=$?
  note "end $name exit=$code $(grep -h -E "b/a|passed|failed|Error" "$out/$name.log" | tail -n 2 | tr '\n' ' ' | cut -c1-300)"
  return $code
}
note "$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader); source $(cat .source_commit 2>/dev/null || echo "$FASTMACE_SOURCE_COMMIT")"
job gate-omat0 python tools/force_stress_gate.py --output "$out/gate-omat0" --engine fastmace --checkpoint "$omat" \
  --model omat0-medium --structures-root "$FASTMACE_STRUCTURES" --structures NaCl NbCl5 \
  --rewrites "${extra[@]}" wprime_backward=on fused_folded_forward=on
ab() {  # name b-fields...
  local name=$1; shift
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    job "ab-$name-$order" python tools/ab_rewrites.py --output "$out/ab-$name-$order" --checkpoint "$omat" \
      --model omat0-medium --structures "$st" --a-neighbours skin=0.5 --b-neighbours skin=0.5 --timestep-fs 1 \
      --warmup 200 --block 200 --blocks 8 --order "$order" --pattern "$pattern" --b "${extra[@]}" "$@"
  done
}
ab ffwd fused_folded_forward=on
ab wprime wprime_backward=on
ab both wprime_backward=on fused_folded_forward=on
note done
grep -h -E "b/a" "$out"/ab-*.log 2>/dev/null | cut -c1-200
