#!/usr/bin/env bash
# OPT3-T04 on Modal (native Linux, where launch costs differ from WSL's): Inductor's pointwise combo kernels
# (RewriteConfig.combo_kernels) against the default plan, tools/ab_rewrites.py same-process A/Bs, 12 blocks of 300
# steps, NaCl and NbCl5, both build orders, inside one container from the root of a pinned source tarball; then the
# CUDA no-value-change test. Records are labelled "Modal <GPU>, not murphy". Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu L4 --timeout-min 60 --commit <sha> \
#     --command "bash tools/modal/opt3_t04.sh" --out ~/fastmace-acceptance/modal-opt3-T04-<sha>
set -uo pipefail
out=${OUT:?the job sets OUT}
checkpoint=$FASTMACE_MODELS/mace-mh-1.model
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
ab() {  # name order pattern
  note "start $1 gpu=$(nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader)"
  python tools/ab_rewrites.py --output "$out/$1" --checkpoint "$checkpoint" --order "$2" --pattern "$3" \
    --a combo_kernels=off --b combo_kernels=on > "$out/$1.log" 2>&1 < /dev/null
  note "end $1 exit=$?"
}
note "Modal $(nvidia-smi --query-gpu=name --format=csv,noheader), not murphy; source $(cat .source_commit) host $(hostname)"
ab combo-ab ab abab
ab combo-ba ba abba
note "start check"
python -m pytest -p no:cacheprovider -q -rf -s tests/gpu/test_combo_kernels_cuda.py > "$out/check.log" 2>&1 < /dev/null
note "end check exit=$?"
note done
grep -h "b/a" "$out"/*.log 2>/dev/null | cut -c1-200
