#!/usr/bin/env bash
# BF16x3 (FACTO, arXiv 2611.03276) on the Modal A100 (sm_80, BF16 dense 16x FP32): the GEMM probe (cuBLAS IEEE,
# Triton ieee/tf32x3/bf16x3/bf16x6), the per-product accuracy on real operands (tools/tf32x3_accuracy.py with the
# split products in bf16x3), the force gate (bf16x3 and the default plan, NaCl and NbCl5) and the same-process
# whole-step A/B, default plan (a) against tf32x3_gemms=on grouped_launch=off in bf16x3 (b), both build orders.
# As results/opt4-T03/bf16x3_speed.sh. Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100 --timeout-min 60 --commit <sha> \
#     --command "bash tools/modal/bf16x3_a100.sh" --out ~/fm-runs/bf16x3-modal-A100
set -uo pipefail
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-mh-1.model
baseline=$FASTMACE_STRUCTURES
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
job() {
  local name=$1; shift
  note "start $name"
  "$@" > "$out/$name.log" 2>&1 < /dev/null
  note "end $name exit=$?"
}
gate() {
  local name=$1; shift
  job "$name" python tools/force_stress_gate.py --output "$out/$name" --engine fastmace --checkpoint "$ck" \
    --model mh1-r2scan --structures NaCl NbCl5 --structures-root "$baseline" "$@"
}
note "Modal $(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader); source $(cat .source_commit)"
export FASTMACE_SPLIT_PRECISION=bf16x3
job probe python results/opt-T01-gemm-precision-20261020/colab-bf16x6/a100/probe_bf16x6.py "$out/probe.json"
job accuracy python tools/tf32x3_accuracy.py --output "$out/accuracy" --structures NaCl NbCl5
gate gate-bf16x3 --rewrites tf32x3_gemms=on grouped_launch=off
gate gate-default
for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  job "ab-$order" python tools/ab_rewrites.py --output "$out/ab-$order" --checkpoint "$ck" --timestep-fs 1 \
    --order "$order" --pattern "$pattern" --b tf32x3_gemms=on grouped_launch=off
done
note done
grep -h "summary\|b/a" "$out"/gate-*.log "$out"/ab-*.log 2>/dev/null | cut -c1-300
