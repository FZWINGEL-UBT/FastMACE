#!/usr/bin/env bash
# OMAT-0 medium on the A100 after decision 66's OMAT-0 defaults (owner, 3 November 2026: "just profile for now on
# a100 ... I just want to be faster on the Benchmark_loup"): Benchmark_loup SrTiO3 10x10x10 (5,000 atoms), float32,
# the default plan (radial tables, tf32x3_gemms on CUTLASS, generic fused forward + wprime_backward).
# - the benchmark's protocol (tools/bench/loup_md.py, NVE and Langevin, 3 fresh processes each);
# - the step breakdown (tools/step_breakdown.py, kernels by category, 300 K, 1 fs);
# - the kernel sources (tools/module_profile.py: compiled graph, eager frame or autograd node per kernel).
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 60 --commit <sha> \
#     --command "bash tools/modal/omat0_a100_profile.sh" --out ~/fm-runs/omat0-a100-profile
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
    job "loup-$ens-$t" python tools/bench/loup_md.py --structure "$st" --checkpoint "$ck" --ensemble "$ens" \
      --output "$out/loup/$ens-$t.json"
  done
done
job breakdown python tools/step_breakdown.py --output "$out/breakdown" --checkpoint "$ck" --model omat0-medium \
  --structure "$st" --temperature-K 300 --timestep-fs 1 --equilibrate-steps 100 --steps 200
job sources python tools/module_profile.py --output "$out/sources" --checkpoint "$ck" --model omat0-medium \
  --structure "$st" --temperature-K 300 --equilibrate-steps 50 --calls 5
rm -f "$out"/breakdown/trace.json "$out"/sources/trace.json
note done
