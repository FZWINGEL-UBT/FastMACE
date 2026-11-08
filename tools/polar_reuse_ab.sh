#!/usr/bin/env bash
# PolarMACE with the MH-1 and OMAT-0 label-gated rewrites (task polar-T01, docs/validation/polar/polar-reuse.md): the
# force gate with the candidate set on (NaCl, NbCl5 3x3x3), then same-process MD-step A/Bs in both build orders.
# Side a is the default plan; each b adds explicit "on" fields: the fused forward + wprime pair (decision 66
# for MH-1 and OMAT-0), wprime alone, the gate epilogue, and all three; or an OMAT-0 kernel switch as a module
# constant (edgesh: EDGE_SH, tablew: TABLE_W, both: the two, fusedfirst: FUSED_FIRST_BACKWARD); tf32 and pairtf
# add tf32x3_gemms (the A100's MH-1 default); species: side a without the species tables.
#   MODEL=polar1-s CKPT=MACE-POLAR-1-S.model; REPEAT="2 2 2" for the gate's cell (POLAR-1-L's float64 reference
#   does not fit 20 GB at 3x3x3); GATE_REWRITES / GATE_CONSTANTS for the gate's production side.
#   PROFILE=on: tools/module_profile.py per structure first. On leto through fm-tiled/run_bash.sh (idle check,
#   NAS caches).
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
    --repeat ${REPEAT:-3 3 3} --rewrites ${GATE_REWRITES:-wprime_backward=on fused_folded_forward=on gate_epilogue=on} \
    ${GATE_CONSTANTS:+--production-constant $GATE_CONSTANTS}
fi
if [ "${PROFILE:-off}" = on ]; then  # where a force call's kernels come from (local part against long range)
  for s in "${structs[@]}"; do
    job "profile-$model-$s" python tools/module_profile.py --output "$out/profile-$model-$s" --checkpoint "$ckpt" \
      --model "$model" --structure "$s"
  done
fi
ab() {  # name side-arguments...
  local name=$1; shift
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    job "ab-$name-$order" python tools/ab_rewrites.py --output "$out/ab-$name-$order" --checkpoint "$ckpt" \
      --model "$model" --structures "${structs[@]}" --warmup "${WARMUP:-300}" --block "${BLOCK:-200}" \
      --blocks "${BLOCKS:-8}" --order "$order" --pattern "$pattern" "$@"
  done
}
EDGE_SH=fastmace.rewrites.generic.modules:EDGE_SH
TABLE_W=fastmace.kernels.oeq_streaming.wprime_backward:TABLE_W
FUSED_FIRST=fastmace.rewrites.generic.modules:FUSED_FIRST_BACKWARD
# BASE: RewriteConfig fields both sides of the constant and species sets take (a GPU where the label-gated
# rewrites are not yet default, e.g. BASE="wprime_backward=on fused_folded_forward=on").
BASE_A=${BASE:+--a $BASE}
BASE_B=${BASE:+--b $BASE}
for set in ${SETS:-pair wprime epilogue all}; do
  case $set in
    pair) ab pair --b wprime_backward=on fused_folded_forward=on ;;
    wprime) ab wprime --b wprime_backward=on ;;
    ffwd) ab ffwd --b fused_folded_forward=on ;;
    epilogue) ab epilogue --b gate_epilogue=on ;;
    all) ab all --b wprime_backward=on fused_folded_forward=on gate_epilogue=on ;;
    edgesh) ab edgesh $BASE_A $BASE_B --a-constant $EDGE_SH=False --b-constant $EDGE_SH=True ;;
    tablew) ab tablew $BASE_A $BASE_B --a-constant $TABLE_W=False --b-constant $TABLE_W=True ;;
    tf32) ab tf32 --b tf32x3_gemms=on ;;
    pairtf) ab pairtf --b wprime_backward=on fused_folded_forward=on tf32x3_gemms=on ;;
    species) ab species --a species_tables=off $BASE $BASE_B ;;
    fusedfirst) ab fusedfirst $BASE_A $BASE_B --a-constant $FUSED_FIRST=False --b-constant $FUSED_FIRST=True ;;
    both) ab both $BASE_A $BASE_B --a-constant $EDGE_SH=False $TABLE_W=False --b-constant $EDGE_SH=True $TABLE_W=True ;;
  esac
done
note done
