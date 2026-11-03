#!/usr/bin/env bash
# Benchmark_loup follow-up (owner, 3 November 2026) on a Modal A100 80GB: MACE-OMAT-0 medium, SrTiO3 10x10x10
# (5,000 atoms), float32. On the A100 80GB PCIe of that benchmark, decision 66 left generic_radial_tables and
# generic_chunked_conv_backward off (the A100 was never measured). This job measures
# - the benchmark's protocol (tools/bench/loup_md.py: from rest, skin 0.5, 1 fs, 5 + 20 steps, 3 fresh processes), the
#   default plan against radial_tables=on chunked_conv_backward=on, NVE and Langevin;
# - same-process whole-step A/Bs (tools/ab_rewrites.py, 300 K, skin 0.5, both build orders): both records, and
#   radial_tables alone;
# - the step breakdown (tools/step_breakdown.py, kernels by category) of both plans.
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 75 --commit <sha> \
#     --command "bash tools/modal/loup_a100.sh" --out ~/fm-runs/loup-a100
set -uo pipefail
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif@10x10x10
both="radial_tables=on chunked_conv_backward=on"
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
job() {
  local name=$1; shift
  note "start $name"
  "$@" > "$out/$name.log" 2>&1 < /dev/null
  note "end $name exit=$? $(tail -n 1 "$out/$name.log" | cut -c1-300)"
}
note "Modal $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader); source $(cat .source_commit)"
for ens in nve langevin; do
  for t in 1 2 3; do
    job "loup-$ens-default-$t" python tools/bench/loup_md.py --structure "$st" --checkpoint "$ck" --ensemble "$ens" \
      --output "$out/loup/$ens-default-$t.json"
    job "loup-$ens-tables-$t" python tools/bench/loup_md.py --structure "$st" --checkpoint "$ck" --ensemble "$ens" \
      --rewrites $both --output "$out/loup/$ens-tables-$t.json"
  done
done
ab() {  # name b-fields...
  local name=$1; shift
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    job "ab-$name-$order" python tools/ab_rewrites.py --output "$out/ab-$name-$order" --checkpoint "$ck" \
      --model omat0-medium --structures "$st" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
      --warmup 200 --block 200 --blocks 8 --order "$order" --pattern "$pattern" --b "$@"
  done
}
ab both $both
ab tables radial_tables=on
job breakdown-default python tools/step_breakdown.py --output "$out/breakdown-default" --checkpoint "$ck" \
  --model omat0-medium --structure "$st" --temperature-K 300 --timestep-fs 1 --equilibrate-steps 100 --steps 200
job breakdown-tables python tools/step_breakdown.py --output "$out/breakdown-tables" --checkpoint "$ck" \
  --model omat0-medium --structure "$st" --temperature-K 300 --timestep-fs 1 --equilibrate-steps 100 --steps 200 \
  --rewrites $both
rm -f "$out"/breakdown-*/trace.json
note done
grep -h "b/a" "$out"/ab-*.log 2>/dev/null | cut -c1-300
