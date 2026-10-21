#!/usr/bin/env bash
# OPT-T08 on Modal (MODAL-T01): results/opt-T08-linear-audit/run.sh's force gate and both same-process A/B
# orders, run inside one Modal container from the root of a pinned source tarball. Labelled "Modal <GPU>, not
# murphy" in every record. Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu L4 --timeout-min 75 --commit <sha> \
#     --command "bash tools/modal/opt_t08.sh" --out ~/fastmace-acceptance/modal-opt-T08-<sha>
# ONLY="gate ab-off-first ab-on-first" selects jobs. One image Python serves as reference and production (the
# research machines use mace_cueq_latest for the float64 reference); the hybrid runs need no launcher, since
# FastMACE prepares the OpenEquivariance runtime in-process.
set -uo pipefail
out=${OUT:?the job sets OUT}
here=results/opt-T08-linear-audit
checkpoint=$FASTMACE_MODELS/mace-mh-1.model
baseline=$FASTMACE_STRUCTURES
all=(species_tables=on fold_readout_tail=on fold_product_linear=on)
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
job() {
  local name=$1; shift
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $name "* ]]; then return 0; fi
  note "start $name gpu=$(nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  "$@" > "$out/$name.log" 2>&1 < /dev/null
  note "end $name exit=$?"
}
note "Modal $(nvidia-smi --query-gpu=name --format=csv,noheader), not murphy; source $(cat .source_commit) host $(hostname)"
job gate python tools/force_stress_gate.py --output "$out/gate" --engine fastmace --checkpoint "$checkpoint" \
  --model mh1-r2scan --structures NaCl NbCl5 --structures-root "$baseline" --rewrites "${all[@]}"
job ab-off-first python "$here/ab_species.py" --output "$out/ab-off-first" --checkpoint "$checkpoint" \
  --order off on --pattern abab --rewrites "${all[@]}"
job ab-on-first python "$here/ab_species.py" --output "$out/ab-on-first" --checkpoint "$checkpoint" \
  --order on off --pattern abba --rewrites "${all[@]}"
note done
grep -h "summary\|b/a" "$out"/gate.log "$out"/ab-*.log 2>/dev/null | cut -c1-200
