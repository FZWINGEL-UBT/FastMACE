#!/usr/bin/env bash
# gate_pack v2: several tools/force_stress_gate.py gates in one Modal container (resource framework §3.4). The
# default plan is gated once, then each switch (decision 66's rounding rule: no worse than the GPU's default).
#
# Environment (tools/jobs/launch.py sets it):
#   OUT          results directory (fastmace_modal.py's job sets it)
#   PACK         "name|MODULE:NAME=VALUE ...;name|..."  one entry per switch (production constants); a body
#                starting with "--" goes to tools/force_stress_gate.py as it is (v2; e.g. "--rewrites tf32x3_gemms=on")
#   EXPECT_GPU   grep -E pattern the GPU name must match; else the job exits 3 (WRONG-GPU)
#   MODEL        model label (default omat0-medium), CHECKPOINT (default $FASTMACE_MODELS/mace-omat-0-medium.model)
#   STRUCTURES   gate structures, space- or comma-separated (v2: launch.py --env takes no spaces; default
#                "results/loup-a100-20261103/SrTiO3.cif@3x3x3 NaCl NbCl5")
# Results: $OUT/run.log (one verdict line per gate), $OUT/gate-<name>/ and .log.
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$real" > "$out/gpu.txt"
if ! echo "$real" | grep -qE "${EXPECT_GPU:?set EXPECT_GPU}"; then echo "WRONG-GPU: $real"; note "WRONG-GPU $real"; exit 3; fi
ck=${CHECKPOINT:-$FASTMACE_MODELS/mace-omat-0-medium.model}
structures_list=${STRUCTURES:-results/loup-a100-20261103/SrTiO3.cif@3x3x3 NaCl NbCl5}
read -r -a structures <<< "${structures_list//,/ }"
note "gate_pack v2 on $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"
failed=0
gate() {
  local name=$1; shift
  note "start gate-$name"
  python tools/force_stress_gate.py --output "$out/gate-$name" --engine fastmace --model "${MODEL:-omat0-medium}" \
    --checkpoint "$ck" --structures-root "$FASTMACE_STRUCTURES" --structures "${structures[@]}" "$@" \
    > "$out/gate-$name.log" 2>&1 < /dev/null
  local code=$?
  [ "$code" = 0 ] || failed=$((failed + 1))
  note "end gate-$name exit=$code $(grep -h -iE 'verdict' "$out/gate-$name.log" | tail -n 1 | cut -c1-200)"
}
gate default
IFS=';' read -r -a entries <<< "${PACK:?set PACK}"
for entry in "${entries[@]}"; do
  [ -n "$entry" ] || continue
  read -r -a consts <<< "${entry#*|}"
  if [ "${consts[0]:0:2}" = "--" ]; then gate "${entry%%|*}" "${consts[@]}"
  else gate "${entry%%|*}" --production-constant "${consts[@]}"; fi
done
note "done failed=$failed"
[ "$failed" = 0 ]
