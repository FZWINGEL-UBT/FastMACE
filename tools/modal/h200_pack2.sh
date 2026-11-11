#!/usr/bin/env bash
# H200 pack 2 (11 November 2026, owner: "run it now"): the A100 switches never measured on the H200, in ONE short
# container (profile fourth, about $2 of credit left):
#   1. OMAT-0 medium MD-step A/B, a = the H200's default plan, b = one switch on, both build orders, SrTiO3 10x10x10
#      and NbCl5 3x3x3 (as h200_hybrid_geom.sh, 4 blocks of 256 steps): fused_forward TP_NOFLUSH and WFRAG_REGION.
#   2. MACE-OFF23-medium, water_216: the A100's three 7 November switches together (offall: radial tables with the
#      stock first block, STOCK_FIRST_SKIP, RADIAL_TF32X3), both orders (tools/offpolar_ab.sh, STOP_AFTER 4).
#   3. The force/stress gate of offall on OFF23-medium (default and candidate), then of TP_NOFLUSH and WFRAG_REGION.
# Steps run in that order and the job's timeout ends whatever is left. Stop rule per switch: not faster in both orders
# on every structure, or the gate fails.
# Environment: OUT (the job sets it), EXPECT_GPU (default H200), BLOCKS (4).
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" | tee -a "$log"; }
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$real" > "$out/gpu.txt"
if ! echo "$real" | grep -qE "${EXPECT_GPU:-H200}"; then note "WRONG-GPU $real"; exit 3; fi
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif
ORG=$FASTMACE_STRUCTURES/Structures/Organic
noflush="fastmace.kernels.oeq_streaming.fused_forward:TP_NOFLUSH=True"
wfrag="fastmace.kernels.oeq_streaming.fused_forward:WFRAG_REGION=True"
note "h200_pack2 on $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"
failed=0

for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  note "start omat-ab-$order"
  python tools/ab_rewrites.py --output "$out/omat-ab-$order" --checkpoint "$ck" --model omat0-medium \
    --structures "$st@10x10x10" "NbCl5@3x3x3" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
    --warmup 200 --block 256 --blocks "${BLOCKS:-4}" --order "$order" --pattern "$pattern" \
    --pack "noflush|$noflush;wfrag|$wfrag" > "$out/omat-ab-$order.log" 2>&1 < /dev/null || failed=$((failed + 1))
  grep -h -E 'b/a' "$out/omat-ab-$order.log" | sed "s/^/  $order /" | tee -a "$log"
done

offall="--b radial_tables=on --b-constant fastmace.rewrites.generic.modules:STOCK_FIRST_SKIP=True fastmace.kernels.triton.skinny_linear:RADIAL_TF32X3=True"
sets=$out/offall.tsv
printf '%s\t%s\n' offall "$offall" > "$sets"
note "start off-ab"
OUT=$out/off-ab MODEL=off23-medium CKPT=MACE-OFF23_medium.model STRUCTS="$ORG/water_216.extxyz@1x1x1" \
  SETS_FILE=$sets WARMUP=500 BLOCK=1000 BLOCKS=8 STOP_AFTER=4 bash tools/offpolar_ab.sh || failed=$((failed + 1))
grep -h "b/a" "$out"/off-ab/ab-*.log 2>/dev/null | sed "s/^/  off /" | tee -a "$log"

gs="$ORG/urea_cod1008787.cif $ORG/naphthalene_cod2311090.cif $ORG/paracetamol_cod2104364.cif $ORG/water_216.extxyz@1x1x1"
note "start off-gate"
OUT=$out/off-gate MODEL=off23-medium CKPT=MACE-OFF23_medium.model STRUCTS=x GATE_STRUCTS="$gs" GATE_NAME=default \
  GATE_ARGS="--seed 20261015" bash tools/offpolar_ab.sh || failed=$((failed + 1))
OUT=$out/off-gate MODEL=off23-medium CKPT=MACE-OFF23_medium.model STRUCTS=x GATE_STRUCTS="$gs" GATE_NAME=candidate \
  GATE_ARGS="--seed 20261015 --rewrites radial_tables=on --production-constant fastmace.rewrites.generic.modules:STOCK_FIRST_SKIP=True fastmace.kernels.triton.skinny_linear:RADIAL_TF32X3=True" \
  bash tools/offpolar_ab.sh || failed=$((failed + 1))
grep -H " summary " "$out"/off-gate/gate-*.log 2>/dev/null | cut -c1-400 | tee -a "$log"

for name in noflush wfrag; do
  c=$([ "$name" = noflush ] && echo "$noflush" || echo "$wfrag")
  note "start gate-$name"
  python tools/force_stress_gate.py --output "$out/gate-$name" --engine fastmace --model omat0-medium \
    --checkpoint "$ck" --structures-root "$FASTMACE_STRUCTURES" --structures "$st@3x3x3" NaCl NbCl5 \
    --production-constant "$c" > "$out/gate-$name.log" 2>&1 < /dev/null || failed=$((failed + 1))
  note "end gate-$name $(tail -n 4 "$out/gate-$name/summary.txt" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
done
note "done failed=$failed"
[ "$failed" = 0 ]
