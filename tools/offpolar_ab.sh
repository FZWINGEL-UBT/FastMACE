#!/usr/bin/env bash
# MD-step A/B screens for MACE-OFF and POLAR-1 (branch fastmace/off-polar-alex, NHR@FAU Alex A100s).
# Each line of $SETS_FILE is "name<TAB>arguments for tools/ab_rewrites.py" (the b side's switch, e.g.
# "--b tf32x3_gemms=on" or "--b-constant fastmace.rewrites.generic.modules:EDGE_SH=True"); every set runs in both
# build orders (ab with abab, ba with abba). Optional gate first: GATE_ARGS = force_stress_gate.py production args.
#   OUT=runs/<id> MODEL=off23-medium CKPT=MACE-OFF23_medium.model STRUCTS="$ORG/water_216.extxyz@1x1x1" \
#     SETS_FILE=sets.tsv bash tools/offpolar_ab.sh
set -uo pipefail
out=${OUT:?the job sets OUT}
mkdir -p "$out"
log=$out/run.log
model=${MODEL:?}
ckpt=$FASTMACE_MODELS/${CKPT:?}
structs=(${STRUCTS:?})
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
job() {
  local name=$1; shift
  note "start $name"
  "$@" > "$out/$name.log" 2>&1 < /dev/null
  local code=$?
  note "end $name exit=$code $(grep -h -E "b/a|passed|failed|Traceback|[A-Za-z]Error:|Refus" "$out/$name.log" | tail -n 4 | tr '\n' ' ' | cut -c1-500)"
  return $code
}
note "$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader); node ${SLURM_JOB_NODELIST:-?}; job ${SLURM_JOB_ID:-?}; source $(cat .source_commit 2>/dev/null); model $model"
if [ -n "${GATE_ARGS:-}" ]; then
  job "gate-${GATE_NAME:-$model}" python tools/force_stress_gate.py --output "$out/gate-${GATE_NAME:-$model}" \
    --engine fastmace --checkpoint "$ckpt" --model "$model" --structures-root "$FASTMACE_STRUCTURES" \
    --structures ${GATE_STRUCTS:-${structs[@]}} --repeat ${REPEAT:-3 3 3} ${GATE_CONFIGS:+--configs $GATE_CONFIGS} $GATE_ARGS
fi
[ -n "${SETS_FILE:-}" ] || exit 0
while IFS=$'\t' read -r name args; do
  [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    eval "set -- $args"
    job "ab-$name-$order" python tools/ab_rewrites.py --output "$out/ab-$name-$order" --checkpoint "$ckpt" \
      --model "$model" --structures "${structs[@]}" --warmup "${WARMUP:-300}" --block "${BLOCK:-200}" \
      --blocks "${BLOCKS:-8}" --order "$order" --pattern "$pattern" "$@"
  done
done < "$SETS_FILE"
note "done"
