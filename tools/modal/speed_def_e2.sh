#!/usr/bin/env bash
# Speed item E2 of plans/capacity-ideas-20261030.md's fifth list on Modal (the A100 leg; decision 66 decides between a
# global default and a per-GPU one): the force/stress gate with gate_epilogue on, then tools/ab_rewrites.py
# same-process A/Bs, a = the default plan (gate_epilogue off), b = gate_epilogue on (the K-outer gated_products
# kernel), both build orders: NaCl, NbCl5 3x3x3 and NaNbO3 4x4x3 (12 blocks of 300 steps), NbCl5 4x4x8 and NaNbO3
# 6x6x6 (8 blocks of 100 steps after 100 warm-up steps), as results/speed-def-20261030/local.sh. Every record is
# labelled "Modal <GPU>, not murphy". Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100 --timeout-min 60 --commit <sha> \
#     --command "bash tools/modal/speed_def_e2.sh" --out ~/fastmace-runs/speed-def/modal-A100
set -uo pipefail
out=${OUT:?the job sets OUT}
checkpoint=$FASTMACE_MODELS/mace-mh-1.model
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
job() {
  local name=$1; shift
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $name "* ]]; then return 0; fi
  note "start $name gpu=$(nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  "$@" > "$out/$name.log" 2>&1 < /dev/null
  note "end $name exit=$?"
}
ab() {  # structure-spec name order [extra...]
  local spec=$1 name=$2 order=$3; shift 3
  job "$name" python tools/ab_rewrites.py --output "$out/ab/$name" --checkpoint "$checkpoint" --model mh1-r2scan \
    --structures "$spec" --order "$order" --pattern "$([ "$order" = ab ] && echo abab || echo abba)" --prepare \
    --b gate_epilogue=on "$@"
}
mkdir -p "$out/ab"
note "Modal $(nvidia-smi --query-gpu=name --format=csv,noheader), not murphy; source $(cat .source_commit) host $(hostname)"
job gate-epi python tools/force_stress_gate.py --output "$out/gate-epi" --engine fastmace --checkpoint "$checkpoint" \
  --model mh1-r2scan --structures NaCl NbCl5 NaNbO3 --structures-root "$FASTMACE_STRUCTURES" --rewrites gate_epilogue=on
for order in ab ba; do
  ab NaCl "ab-e2-NaCl-small-$order" "$order"
  ab NbCl5 "ab-e2-NbCl5-small-$order" "$order"
  ab NaNbO3@4x4x3 "ab-e2-NaNbO3-small-$order" "$order"
  ab NbCl5@4x4x8 "ab-e2-NbCl5-big-$order" "$order" --warmup 100 --block 100 --blocks 8
  ab NaNbO3@6x6x6 "ab-e2-NaNbO3-big-$order" "$order" --warmup 100 --block 100 --blocks 8
done
note done
grep -h "summary\|b/a" "$out"/gate-epi.log "$out"/ab-*.log 2>/dev/null | cut -c1-200
