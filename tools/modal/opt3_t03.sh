#!/usr/bin/env bash
# OPT3-T03's A/B on Modal (MODAL-T01): EngineConfig.fused_integrator off (a) against on (b), tools/ab_rewrites.py
# same-process A/Bs, 12 blocks of 300 steps, both build orders, NaCl and NbCl5 at 300 K, OPT-T08's records on
# both sides (round 3's baseline). Every record is labelled "Modal <GPU>, not murphy". Usage (from WSL, Modal
# client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu L4 --timeout-min 60 --commit <sha> \
#     --command "bash tools/modal/opt3_t03.sh" --out ~/fastmace-acceptance/modal-opt3-T03
set -uo pipefail
out=${OUT:?the job sets OUT}
checkpoint=$FASTMACE_MODELS/mace-mh-1.model
t08=(species_tables=on fold_readout_tail=on fold_product_linear=on)
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
ab() {  # name order pattern args...
  local name=$1 order=$2 pattern=$3; shift 3
  note "start $name gpu=$(nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader)"
  python tools/ab_rewrites.py --output "$out/$name" --checkpoint "$checkpoint" --order "$order" --pattern "$pattern" \
    "$@" > "$out/$name.log" 2>&1 < /dev/null
  note "end $name exit=$?"
}
note "Modal $(nvidia-smi --query-gpu=name --format=csv,noheader), not murphy; source $(cat .source_commit) host $(hostname)"
ab fused-ab ab abab --temperature-K 300 --a "${t08[@]}" --b "${t08[@]}" --b-engine fused_integrator=true
ab fused-ba ba abba --temperature-K 300 --a "${t08[@]}" --b "${t08[@]}" --b-engine fused_integrator=true
note done
grep -h "b/a" "$out"/*.log 2>/dev/null | cut -c1-200
