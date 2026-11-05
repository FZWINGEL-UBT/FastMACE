#!/usr/bin/env bash
# PolarMACE with the MH-1 and OMAT-0 label-gated rewrites (task polar-T01, docs/validation/polar-reuse.md): the
# force gate with the candidate set on (NaCl, NbCl5 3x3x3), then same-process MD-step A/Bs in both build orders.
# Side a is the default POLAR-1 plan; each b adds explicit "on" fields: the fused forward + wprime pair (decision 66
# for MH-1 and OMAT-0), wprime alone, the gate epilogue, and all three.
#   MODEL=polar1-s CKPT=MACE-POLAR-1-S.model; on leto through fm-tiled/run.sh (idle check, NAS caches).
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
model=${MODEL:-polar1-s}
ckpt=$FASTMACE_MODELS/${CKPT:-MACE-POLAR-1-S.model}
structs=(${STRUCTS:-NaCl NbCl5})
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
job() {
  local name=$1; shift
  note "start $name"
  "$@" > "$out/$name.log" 2>&1 < /dev/null
  local code=$?
  note "end $name exit=$code $(grep -h -E "b/a|passed|failed|Error" "$out/$name.log" | tail -n 4 | tr '\n' ' ' | cut -c1-400)"
  return $code
}
note "$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader); source $(cat .source_commit 2>/dev/null || echo "$FASTMACE_SOURCE_COMMIT"); model $model"
if [ "${GATE:-on}" = on ]; then
  job "gate-$model" python tools/force_stress_gate.py --output "$out/gate-$model" --engine fastmace --checkpoint "$ckpt" \
    --model "$model" --structures-root "$FASTMACE_STRUCTURES" --structures "${structs[@]}" \
    --rewrites wprime_backward=on fused_folded_forward=on gate_epilogue=on
fi
ab() {  # name b-fields...
  local name=$1; shift
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    job "ab-$name-$order" python tools/ab_rewrites.py --output "$out/ab-$name-$order" --checkpoint "$ckpt" \
      --model "$model" --structures "${structs[@]}" --warmup "${WARMUP:-300}" --block "${BLOCK:-200}" \
      --blocks "${BLOCKS:-8}" --order "$order" --pattern "$pattern" --b "$@"
  done
}
for set in ${SETS:-pair wprime epilogue all}; do
  case $set in
    pair) ab pair wprime_backward=on fused_folded_forward=on ;;
    wprime) ab wprime wprime_backward=on ;;
    ffwd) ab ffwd fused_folded_forward=on ;;
    epilogue) ab epilogue gate_epilogue=on ;;
    all) ab all wprime_backward=on fused_folded_forward=on gate_epilogue=on ;;
  esac
done
note done
