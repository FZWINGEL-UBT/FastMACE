#!/usr/bin/env bash
# CUTLASS 3xTF32 follow-up on the A100 (3 November 2026): the op check and GPU tests with the tile rule
# (cutlass.tf32x3.takes: MIN_TILES, LONG_K), and the OMAT-0 force gate with and without tf32x3_gemms on the same GPU
# (the control for the first job's energy ratios).
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 40 --commit <sha> \
#     --command "bash tools/modal/a100_cutlass_followup.sh" --out ~/fm-runs/a100-cutlass-followup
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
omat=$FASTMACE_MODELS/mace-omat-0-medium.model
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
job() {
  local name=$1; shift
  note "start $name"
  "$@" > "$out/$name.log" 2>&1 < /dev/null
  local code=$?
  note "end $name exit=$code $(grep -h -E "passed|failed|worst" "$out/$name.log" | tail -n 1 | cut -c1-300)"
  return $code
}
note "Modal $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader); source $(cat .source_commit)"
job check python tools/bench/cutlass_tf32x3_check.py --output "$out/check.json"
job gpu-tests python -m pytest -q -p no:cacheprovider tests/gpu/test_tf32x3_gemm_cuda.py
gate() {  # name rewrites...
  local name=$1; shift
  job "$name" python tools/force_stress_gate.py --output "$out/$name" --engine fastmace --checkpoint "$omat" \
    --model omat0-medium --structures-root "$FASTMACE_STRUCTURES" --structures NaCl NbCl5 --rewrites "$@"
}
gate gate-omat0-control grouped_launch=off
gate gate-omat0-tf32x3 tf32x3_gemms=on grouped_launch=off
note done
