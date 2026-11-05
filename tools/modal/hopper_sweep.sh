#!/usr/bin/env bash
# Hopper sweep (owner, 5 November 2026: "do the sweep ... then try out the ideas that might work that didnt work on
# a100"). OMAT-0 medium, Benchmark_loup SrTiO3 10x10x10, on an H100 or H200 run under the A100-SXM4 plan
# (tools/bench/gpu_alias: FASTMACE_GPU_ALIAS, the GPU's name -> "NVIDIA A100-SXM4-80GB") with the sm_90 CUTLASS cubin:
# - GPU tests of the CUTLASS 3xTF32 GEMMs on sm_90;
# - the force/stress gate (NaCl, NbCl5, SrTiO3 3x3x3);
# - Benchmark_loup Langevin, two fresh processes;
# - kernel lab: the fused forward and the w' backward, the A100 default against the launch shapes and the variants
#   the A100 rejected (FORWARD_VARIANTS, WPRIME_VARIANTS);
# - MD-step A/Bs on SrTiO3 10x10x10, both build orders, for each entry of AB ("name|MODULE:NAME=VALUE ...").
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu H200 --timeout-min 45 --commit <sha> \
#     --command "bash tools/modal/hopper_sweep.sh" --out ~/fm-runs/hopper-sweep
set -uo pipefail
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
export PYTHONPATH=$PWD/tools/bench/gpu_alias:${PYTHONPATH:-} FASTMACE_GPU_ALIAS="$real=${ALIAS:-NVIDIA A100-SXM4-80GB}"
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
step() {
  local name=$1; shift
  note "start $name"
  "$@" > "$out/$name.log" 2>&1 < /dev/null
  note "end $name exit=$? $(tail -n 1 "$out/$name.log" | cut -c1-300)"
}
note "Modal $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); alias $FASTMACE_GPU_ALIAS; source $(cat .source_commit 2>/dev/null)"
[ "${TESTS:-1}" = 0 ] || step tests python -m pytest -p no:cacheprovider -q -rs tests/gpu/test_cutlass_tf32x3_tiles_cuda.py \
  tests/gpu/test_tf32x3_gemm_cuda.py
[ "${GATE:-1}" = 0 ] || step gate python tools/force_stress_gate.py --output "$out/gate" --engine fastmace --model omat0-medium \
  --checkpoint "$ck" --structures-root "$FASTMACE_STRUCTURES" --structures NaCl NbCl5 "$st@3x3x3"
[ "${GATE:-1}" = 0 ] || grep -h -iE "verdict|pass|fail" "$out/gate.log" | tail -n 6 >> "$log"
for t in $(seq 1 "${LOUP:-2}"); do
  step "loup-$t" python tools/bench/loup_md.py --structure "$st@10x10x10" --checkpoint "$ck" --ensemble langevin \
    --output "$out/loup/langevin-$t.json"
done
[ -z "${FORWARD_VARIANTS-base tc-warps8 tc-warps10 tc-prefetch tc-prefetch-x tc-prefetch-w10 tc-presplit tc-hybrid16 tc-fp16 simt}" ] ||
  step lab-forward python tools/omat0_kernel_lab.py --checkpoint "$ck" --kernels forward --rounds 15 --ptxas-archs 9.0 \
    --variants ${FORWARD_VARIANTS-base tc-warps8 tc-warps10 tc-prefetch tc-prefetch-x tc-prefetch-w10 tc-presplit tc-hybrid16 tc-fp16 simt} \
    --output "$out/lab-forward"
grep -h -E "^forward |rule" "$out/lab-forward.log" 2>/dev/null | cut -c1-240 >> "$log"
[ -z "${WPRIME_VARIANTS-base maxreg-none maxreg80 maxreg72 redux butterfly F-split2 pair}" ] ||
  step lab-wprime python tools/omat0_kernel_lab.py --checkpoint "$ck" --kernels wprime --rounds 15 --ptxas-archs 9.0 \
    --variants ${WPRIME_VARIANTS-base maxreg-none maxreg80 maxreg72 redux butterfly F-split2 pair} --output "$out/lab-wprime"
grep -h -E "^wprime |rule" "$out/lab-wprime.log" 2>/dev/null | cut -c1-240 >> "$log"
IFS=';' read -r -a abs <<< "${AB-triton|fastmace.kernels.triton.tf32x3_gemm:CUTLASS_CAPABILITIES=();tile32x64|fastmace.kernels.cutlass.tf32x3:TILE='32x64'}"
for entry in "${abs[@]}"; do
  [ -n "$entry" ] || continue
  name=${entry%%|*}; read -r -a bc <<< "${entry#*|}"
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    step "ab-$name-$order" python tools/ab_rewrites.py --output "$out/ab-$name-$order" --checkpoint "$ck" \
      --model omat0-medium --structures "$st@10x10x10" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
      --warmup 200 --block 256 --blocks 6 --order "$order" --pattern "$pattern" --b-constant "${bc[@]}"
    grep -h -E "b/a" "$out/ab-$name-$order.log" | tail -n 1 >> "$log"
  done
done
note done
