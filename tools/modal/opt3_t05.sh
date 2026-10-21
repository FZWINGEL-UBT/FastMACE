#!/usr/bin/env bash
# OPT3-T05 (layout audit) A/B on Modal (MODAL-T01): tools/ab_rewrites.py same-process A/Bs, 12 blocks of 300 steps,
# both build orders, NaCl and NbCl5, side b = the default plan + direct_node_feats + folded_constants. Every record
# is labelled "Modal <GPU>, not murphy". Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu L4 --timeout-min 60 --commit <sha> \
#     --command "bash tools/modal/opt3_t05.sh" --out ~/fastmace-runs/opt3-T05/modal-L4
set -uo pipefail
out=${OUT:?the job sets OUT}
checkpoint=$FASTMACE_MODELS/mace-mh-1.model
records=(direct_node_feats=on folded_constants=on)
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
ab() {  # name order pattern
  local name=$1 order=$2 pattern=$3
  note "start $name gpu=$(nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader)"
  python tools/ab_rewrites.py --output "$out/$name" --checkpoint "$checkpoint" --order "$order" --pattern "$pattern" \
    --b "${records[@]}" > "$out/$name.log" 2>&1 < /dev/null
  note "end $name exit=$?"
}
note "Modal $(nvidia-smi --query-gpu=name --format=csv,noheader), not murphy; source $(cat .source_commit) host $(hostname)"
ab layout-ab ab abab
ab layout-ba ba abba
note done
grep -h "b/a" "$out"/*.log 2>/dev/null | cut -c1-200
