#!/usr/bin/env bash
# CUTLASS 3xTF32 GEMMs on the A100 (3 November 2026; fastmace/kernels/cutlass, owner: "build it for a100").
# 1. The op against Triton's tf32x3 and cuBLAS IEEE (tools/bench/cutlass_tf32x3_check.py): errors, repeat, graph,
#    times. Stops here if CUTLASS does not load.
# 2. tests/gpu/test_tf32x3_gemm_cuda.py (now on the CUTLASS path) and tools/tf32x3_accuracy.py (real operands).
# 3. The force gate with tf32x3_gemms=on: MH-1 and OMAT-0 medium on NaCl and NbCl5.
# 4. Same-process MD-step A/Bs, both build orders, b = a + tf32x3_gemms=on: OMAT-0 SrTiO3 5,000 atoms (the
#    Benchmark_loup cell, 1 fs, skin 0.5) and MH-1 NaCl, NbCl5 (3x3x3); then b with CUTLASS against b with Triton.
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 80 --commit <sha> \
#     --command "bash tools/modal/a100_cutlass_tf32x3.sh" --out ~/fm-runs/a100-cutlass-tf32x3
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
omat=$FASTMACE_MODELS/mace-omat-0-medium.model
mh1=$FASTMACE_MODELS/mace-mh-1.model
st=results/loup-a100-20261103/SrTiO3.cif@10x10x10
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
job() {
  local name=$1; shift
  note "start $name"
  "$@" > "$out/$name.log" 2>&1 < /dev/null
  local code=$?
  note "end $name exit=$code $(grep -h -E "b/a|summary|passed|failed|worst" "$out/$name.log" | tail -n 2 | tr '\n' ' ' | cut -c1-300)"
  return $code
}
note "Modal $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader); source $(cat .source_commit)"
job check python tools/bench/cutlass_tf32x3_check.py --output "$out/check.json" || { note "CUTLASS check failed"; exit 1; }
job gpu-tests python -m pytest -q -p no:cacheprovider tests/gpu/test_tf32x3_gemm_cuda.py
job accuracy-mh1 python tools/tf32x3_accuracy.py --output "$out/accuracy-mh1" --model mh1-r2scan
job gate-mh1 python tools/force_stress_gate.py --output "$out/gate-mh1" --engine fastmace --checkpoint "$mh1" \
  --model mh1-r2scan --structures-root "$FASTMACE_STRUCTURES" --structures NaCl NbCl5 --rewrites tf32x3_gemms=on grouped_launch=off
job gate-omat0 python tools/force_stress_gate.py --output "$out/gate-omat0" --engine fastmace --checkpoint "$omat" \
  --model omat0-medium --structures-root "$FASTMACE_STRUCTURES" --structures NaCl NbCl5 --rewrites tf32x3_gemms=on grouped_launch=off
ab() {  # name model checkpoint structures... -- extra
  local name=$1 model=$2 ck=$3; shift 3
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    job "ab-$name-$order" python tools/ab_rewrites.py --output "$out/ab-$name-$order" --checkpoint "$ck" \
      --model "$model" --timestep-fs 1 --warmup 200 --block 200 --blocks 8 --order "$order" --pattern "$pattern" "$@"
  done
}
# tf32x3_gemms conflicts with grouped_launch (both reroute the block products): b turns grouped_launch off, and a
# control pair (a = grouped_launch=off) separates the two.
nb=(--a-neighbours skin=0.5 --b-neighbours skin=0.5)
ab omat0 omat0-medium "$omat" --structures "$st" "${nb[@]}" --b tf32x3_gemms=on grouped_launch=off
ab omat0-control omat0-medium "$omat" --structures "$st" "${nb[@]}" --a grouped_launch=off   --b tf32x3_gemms=on grouped_launch=off
ab mh1 mh1-r2scan "$mh1" --structures NaCl NbCl5 --a grouped_launch=off --b tf32x3_gemms=on grouped_launch=off
ab omat0-backend omat0-medium "$omat" --structures "$st" "${nb[@]}" --a tf32x3_gemms=on grouped_launch=off   --a-constant fastmace.kernels.triton.tf32x3_gemm:CUTLASS=False --b tf32x3_gemms=on grouped_launch=off
note done
grep -h -E "b/a" "$out"/ab-*.log 2>/dev/null | cut -c1-300
