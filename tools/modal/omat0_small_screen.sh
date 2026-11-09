#!/usr/bin/env bash
# OMAT-0 small on the H200: medium's tuned per-GPU switches screened on small (8 November 2026; owner: "do you think
# we could improve omat 0 small specifically?" -> "Yes, run the screen"). Every H200 kernel switch was chosen on
# OMAT-0 medium; small (128x0e hidden features) runs the same kernels on another shape. Benchmark_loup (SrTiO3
# 10x10x10, 300 K, 1 fs, skin 0.5), a = small's defaults (label omat0-small, no constants), b = a plus one change,
# tools/ab_rewrites.py --pack in both build orders. b/a below 1: the change is faster on small.
# Two packs of 7-8 entries, so that order ba (every b built before a) holds at most nine engines at once.
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$real" > "$out/gpu.txt"
if ! echo "$real" | grep -qE "${EXPECT_GPU:-H200}"; then note "WRONG-GPU $real"; exit 3; fi
ck=${CHECKPOINT:-$FASTMACE_MODELS/mace-omat-0-small.model}
st=results/loup-a100-20261103/SrTiO3.cif
sf=fastmace.kernels.oeq_streaming.spline_forward
wp=fastmace.kernels.oeq_streaming.wprime_backward
cs=fastmace.kernels.triton.compose_skip
tf=fastmace.kernels.cutlass.tf32x3
to=fastmace.kernels.triton.tabled_outer_sum
note "omat0_small_screen on $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit 2>/dev/null); checkpoint $(sha256sum "$ck" | cut -c1-16)"

declare -a packs
# 1: the spline forward's launch shape and staging, the w' backward
p="r128off|$sf:R128=False"
p="$p;mb7|$sf:R128=False $sf:THREADS=128 $sf:MIN_BLOCKS=7"
p="$p;mb8|$sf:R128=False $sf:THREADS=128 $sf:MIN_BLOCKS=8"
p="$p;fstageoff|$sf:STAGE_EDGES=False;prefetch|$sf:PREFETCH_AHEAD=True"
p="$p;vec2|$wp:SPLINE_VECTOR=2;wshoff|$wp:STAGE_SH=False;wedges|$wp:STAGE_EDGES=True"
packs[1]=$p
# 2: more w', the composed skip, the GEMM tile, the tabled backward
p="doutoff|$wp:STAGE_DOUT=False;alphaoff|$wp:FOLD_ALPHA=False"
p="$p;joined|$cs:JOINED_LAYOUT=True;countoff|$cs:COUNT_PLAN=False;fastoff|$cs:COMPOSE_FAST=False"
p="$p;tile3264|$tf:TILE='32x64';tile6432|$tf:TILE='64x32';poweroff|$to:POWER_BACKWARD=False"
packs[2]=$p

for half in 1 2; do
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    note "start pack$half-$order"
    python tools/ab_rewrites.py --output "$out/ab$half-$order" --checkpoint "$ck" --model "$ck" \
      --structures "$st@10x10x10" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
      --warmup 200 --block 256 --blocks 6 --order "$order" --pattern "$pattern" \
      --pack "${packs[$half]}" > "$out/ab$half-$order.log" 2>&1 < /dev/null
    note "end pack$half-$order exit=$?"
    grep -h -E 'b/a' "$out/ab$half-$order.log" | sed "s/^/  $order /" >> "$log"
  done
done
note done
