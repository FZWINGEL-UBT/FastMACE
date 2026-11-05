#!/usr/bin/env bash
# H200 exploration (owner, 5 November 2026: "Try to make fastmace as fast as possible on h200"). OMAT-0 medium,
# Benchmark_loup SrTiO3 10x10x10, float32, on the GPU's own plan (no alias unless ALIAS is set):
# - Benchmark_loup Langevin, LOUP fresh processes (default 2);
# - the force/stress gate (GATE=1);
# - the step breakdown (tools/step_breakdown.py; kernels by category);
# - kernel lab: the fused forward's launch shapes (grid1/3/4, rc1/3/4) and its falsifier bounds (how much of the
#   kernel the TF32 product, the redistribution, the loads and the tensor product take: the ceilings of a wgmma
#   rewrite and of on-chip sharing), and the w' backward's falsifier bounds.
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu H200 --timeout-min 30 --commit <sha> \
#     --command "bash tools/modal/h200_explore.sh" --out ~/fm-runs/h200-explore
set -uo pipefail
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif
if [ -n "${ALIAS:-}" ]; then
  real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
  export PYTHONPATH=$PWD/tools/bench/gpu_alias:${PYTHONPATH:-} FASTMACE_GPU_ALIAS="$real=$ALIAS"
fi
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
step() {
  local name=$1; shift
  note "start $name"
  "$@" > "$out/$name.log" 2>&1 < /dev/null
  note "end $name exit=$? $(tail -n 1 "$out/$name.log" | cut -c1-300)"
}
note "Modal $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); alias ${FASTMACE_GPU_ALIAS:-none}; source $(cat .source_commit 2>/dev/null)"
for t in $(seq 1 "${LOUP:-2}"); do
  step "loup-$t" python tools/bench/loup_md.py --structure "$st@10x10x10" --checkpoint "$ck" --ensemble langevin \
    --output "$out/loup/langevin-$t.json"
done
[ "${GATE:-1}" = 0 ] || step gate python tools/force_stress_gate.py --output "$out/gate" --engine fastmace \
  --model omat0-medium --checkpoint "$ck" --structures-root "$FASTMACE_STRUCTURES" --structures NaCl NbCl5 "$st@3x3x3"
[ "${GATE:-1}" = 0 ] || grep -h -iE "verdict|pass|fail" "$out/gate.log" | tail -n 6 >> "$log"
[ "${BREAKDOWN:-1}" = 0 ] || step breakdown python tools/step_breakdown.py --output "$out/breakdown" --checkpoint "$ck" \
  --model omat0-medium --structure "$st@10x10x10" --temperature-K 300 --timestep-fs 1 --equilibrate-steps 100 \
  --steps 200
rm -f "$out"/breakdown/trace.json
[ -z "${FORWARD_VARIANTS-base grid1 grid3 grid4 rc1 rc3 rc4 F-tc-1pass F-tc-no-mma F-tc-no-product F-tc-no-redistribute F-gemm-only F-no-tp F-no-store F-no-xload}" ] ||
  step lab-forward python tools/omat0_kernel_lab.py --checkpoint "$ck" --kernels forward --rounds 15 \
    --variants ${FORWARD_VARIANTS-base grid1 grid3 grid4 rc1 rc3 rc4 F-tc-1pass F-tc-no-mma F-tc-no-product F-tc-no-redistribute F-gemm-only F-no-tp F-no-store F-no-xload} \
    --output "$out/lab-forward"
grep -h -E "^forward " "$out/lab-forward.log" 2>/dev/null | cut -c1-200 >> "$log"
[ -z "${WPRIME_VARIANTS-base F-no-atomics F-no-wload F-no-tp F-tp-no-coef F-no-reduce F-one-slab}" ] ||
  step lab-wprime python tools/omat0_kernel_lab.py --checkpoint "$ck" --kernels wprime --rounds 15 \
    --variants ${WPRIME_VARIANTS-base F-no-atomics F-no-wload F-no-tp F-tp-no-coef F-no-reduce F-one-slab} \
    --output "$out/lab-wprime"
grep -h -E "^wprime " "$out/lab-wprime.log" 2>/dev/null | cut -c1-200 >> "$log"
note done
