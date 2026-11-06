#!/usr/bin/env bash
# Ideas queue 2 (6 November 2026), one Modal job per GPU: G1's ceiling bench (tools/bench/g1_ceiling_bench.py on this
# plan's GEMM shapes) and, with RC=1 (the A100-SXM4), idea 28's rc-all in the forward lab: once on round 13's base
# (A_LD128 and TP_NOFLUSH off, which rc-all adds back) and once on today's default, in the same job.
# ALIAS: the exact device name the job must land on.
set -uo pipefail
out=${OUT:?the job sets OUT}
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
[ "$gpu" = "${ALIAS:?set ALIAS}" ] || { echo "WRONG-GPU: $gpu"; exit 3; }
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "Modal $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); source $(cat .source_commit)"
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d .)
g1=results/ideas-queue2-20261106/g1
step() { local name=$1; shift; "$@" > "$out/$name.log" 2>&1 < /dev/null; note "$name exit=$? $(tail -n 1 "$out/$name.log" | cut -c1-300)"; }
step shapes python tools/bench/gemm_shapes.py --structure "$st@10x10x10" --checkpoint "$ck" --output "$out/shapes.json"
step g1 python tools/bench/g1_ceiling_bench.py --shapes "$out/shapes.json" --cubin "plain=$g1/plain_sm$cap.cubin" \
  --cubin "ceiling=$g1/ceiling_sm$cap.cubin" --output "$out/g1.json"
if [ "${RC:-0}" = 1 ]; then
  ff=fastmace.kernels.oeq_streaming.fused_forward
  step lab-r13base python tools/omat0_kernel_lab.py --checkpoint "$ck" --kernels forward --rounds 15 \
    --constant "$ff:A_LD128=False" "$ff:TP_NOFLUSH=False" --variants base fold-tpnf fold-a-ld128 rcfold rc-all \
    --output "$out/lab-r13base"
  grep -h -E "^forward " "$out/lab-r13base.log" | cut -c1-240 >> "$log"
  step lab-default python tools/omat0_kernel_lab.py --checkpoint "$ck" --kernels forward --rounds 15 --variants base \
    --output "$out/lab-default"
  grep -h -E "^forward " "$out/lab-default.log" | cut -c1-240 >> "$log"
fi
note done
