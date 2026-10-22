#!/usr/bin/env bash
# 3-T20 (phase 3 benchmarks, items 1-3) on Modal: one container, MH-1 hybrid float32, in priority order:
#   item 1: tools/bench/event_speed.py, species-swap events every 100 steps against fixed species, 1500 K, 1 fs,
#           12 blocks of 300 steps after 300 warm-up steps and --settle 5, both build orders, NaCl and NbCl5;
#   item 2: tools/bench/mc_cost.py, swaps (float32 and float64 MC energies) and displacements (with and without a
#           trial rebuild), NaCl and NbCl5, 1500 K, 8 rounds;
#   item 3: tools/validation/move_energy_precision.py, MH-1 matpes_r2scan, NaCl, NbCl5 and NaNbO3.
# Every record is labelled "Modal <GPU>". Usage (from WSL, Modal client venv; budget $2, guard in fastmace_modal.py):
#   modal run tools/modal/fastmace_modal.py::job --gpu L4 --timeout-min 70 --commit <sha> \
#     --command "bash tools/modal/3_t20.sh" --out ~/fastmace-runs/3-t20/modal-L4-<sha>
set -uo pipefail
out=${OUT:?the job sets OUT}
checkpoint=$FASTMACE_MODELS/mace-mh-1.model
export FASTMACE_SOURCE_COMMIT=$(cat .source_commit)
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
run() {  # name command...
  local name=$1; shift
  note "start $name gpu=$(nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader)"
  "$@" > "$out/$name.log" 2>&1 < /dev/null
  note "end $name exit=$?"
}
note "Modal $(nvidia-smi --query-gpu=name --format=csv,noheader); source $FASTMACE_SOURCE_COMMIT host $(hostname)"
for structure in NaCl NbCl5; do
  run events-$structure-ab python tools/bench/event_speed.py --output "$out/events-$structure-ab" \
    --checkpoint "$checkpoint" --structures "$structure" --order ab --pattern abab --settle 5
  run events-$structure-ba python tools/bench/event_speed.py --output "$out/events-$structure-ba" \
    --checkpoint "$checkpoint" --structures "$structure" --order ba --pattern abba --settle 5
done
run mc-NaCl-swap python tools/bench/mc_cost.py --output "$out/mc-NaCl-swap" --checkpoint "$checkpoint" \
  --structure NaCl --temperature-K 1500 --block 100 --rounds 8
run mc-NbCl5-swap python tools/bench/mc_cost.py --output "$out/mc-NbCl5-swap" --checkpoint "$checkpoint" \
  --structure NbCl5 --temperature-K 1500 --block 50 --rounds 8
for structure in NaCl NbCl5; do
  run mc-$structure-displace python tools/bench/mc_cost.py --output "$out/mc-$structure-displace" \
    --checkpoint "$checkpoint" --structure "$structure" --temperature-K 1500 --displace-delta-A 0.1 --block 20 --rounds 8
done
run de-matpes python tools/validation/move_energy_precision.py --model mh1 --head matpes_r2scan --checkpoint "$checkpoint" \
  --structures NaCl NbCl5 NaNbO3 --output "$out/de-matpes"
note done
grep -h "rate fixed/events" "$out"/*.log 2>/dev/null | cut -c1-200
