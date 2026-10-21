#!/usr/bin/env bash
# Round-2 A/Bs and the noise floor on Modal (MODAL-T01): tools/ab_rewrites.py same-process A/Bs, 12 blocks of
# 300 steps, both build orders, inside one Modal container from the root of a pinned source tarball. Every record
# is labelled "Modal <GPU>, not murphy". Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu L4 --timeout-min 90 --commit <sha> \
#     --command "ONLY='noise-ab noise-ba' bash tools/modal/round2.sh" --out ~/fastmace-acceptance/modal-<run>
# Jobs:
# - noise-ab, noise-ba: a = b = the default plan (the noise floor), NaCl and NbCl5. The tool refuses identical
#   fields, so side a asks gated_rows=auto and side b gated_rows=on: both resolve to on in the default plan;
# - overlap-ab/ba (NaCl, NbCl5 at 300 K) and overlap1000-ab/ba (NaCl at 1000 K): EngineConfig.overlap_rebuild
#   (OPT2-T04) on side b;
# - tiles-ab/ba: RewriteConfig.folded_backward_tiles (OPT2-T03) on side b, NaCl and NbCl5;
# - t08-ab/ba: OPT-T08's three records off (a) against on (b), NaCl and NbCl5.
# The candidate A/Bs keep OPT-T08's records on both sides (round 2's baseline, plans/opt-2.md).
set -uo pipefail
out=${OUT:?the job sets OUT}
checkpoint=$FASTMACE_MODELS/mace-mh-1.model
t08=(species_tables=on fold_readout_tail=on fold_product_linear=on)
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
ab() {  # name order pattern args...
  local name=$1 order=$2 pattern=$3; shift 3
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $name "* ]]; then return 0; fi
  note "start $name gpu=$(nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader)"
  python tools/ab_rewrites.py --output "$out/$name" --checkpoint "$checkpoint" --order "$order" --pattern "$pattern" \
    "$@" > "$out/$name.log" 2>&1 < /dev/null
  note "end $name exit=$?"
}
note "Modal $(nvidia-smi --query-gpu=name --format=csv,noheader), not murphy; source $(cat .source_commit) host $(hostname)"
ab noise-ab ab abab --a gated_rows=auto --b gated_rows=on
ab noise-ba ba abba --a gated_rows=auto --b gated_rows=on
ab overlap-ab ab abab --a "${t08[@]}" --b "${t08[@]}" --b-engine overlap_rebuild=true
ab overlap-ba ba abba --a "${t08[@]}" --b "${t08[@]}" --b-engine overlap_rebuild=true
ab overlap1000-ab ab abab --structures NaCl --temperature-K 1000 --a "${t08[@]}" --b "${t08[@]}" \
  --b-engine overlap_rebuild=true
ab overlap1000-ba ba abba --structures NaCl --temperature-K 1000 --a "${t08[@]}" --b "${t08[@]}" \
  --b-engine overlap_rebuild=true
ab tiles-ab ab abab --a "${t08[@]}" --b "${t08[@]}" folded_backward_tiles=on
ab tiles-ba ba abba --a "${t08[@]}" --b "${t08[@]}" folded_backward_tiles=on
ab t08-ab ab abab --b "${t08[@]}"
ab t08-ba ba abba --b "${t08[@]}"
note done
grep -h "b/a" "$out"/*.log 2>/dev/null | cut -c1-200
