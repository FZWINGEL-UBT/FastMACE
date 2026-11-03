#!/usr/bin/env bash
# Benchmark_loup follow-up, second job (3 November 2026), Modal A100 80GB PCIe: the first job (loup_a100.sh) found
# generic_radial_tables alone 0.90 in both build orders on SrTiO3 5,000 atoms and the chunked backward 1.12 with it.
# This job measures the benchmark's protocol with radial_tables=on alone (3 fresh processes, NVE and Langevin) and
# the radial_tables A/B on a second crystal, NaCl 10x10x10 (8,000 atoms), both build orders.
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 45 --commit <sha> \
#     --command "bash tools/modal/loup_a100_b.sh" --out ~/fm-runs/loup-a100-b
set -uo pipefail
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif@10x10x10
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
    job "loup-$ens-radial-$t" python tools/bench/loup_md.py --structure "$st" --checkpoint "$ck" --ensemble "$ens" \
      --rewrites radial_tables=on --output "$out/loup/$ens-radial-$t.json"
  done
done
for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  job "ab-tables-NaCl-$order" python tools/ab_rewrites.py --output "$out/ab-tables-NaCl-$order" --checkpoint "$ck" \
    --model omat0-medium --structures NaCl@10x10x10 --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
    --warmup 200 --block 200 --blocks 8 --order "$order" --pattern "$pattern" --b radial_tables=on
done
note done
grep -h "b/a" "$out"/ab-*.log 2>/dev/null | cut -c1-300
