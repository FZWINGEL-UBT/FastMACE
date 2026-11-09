#!/usr/bin/env bash
# OMAT-0 small on the H200, 9 November 2026: small's defaults after main c8d08b5b (JOINED_LAYOUT, BLOCK_ORDER,
# SORT_BY_LENGTH on the H200 for every model; w' STAGE_EDGES for small) screened against the H200 switches chosen on
# medium. a = small's defaults (no constants), b = a plus one change; b/a below 1: the change is faster on small.
# Benchmark_loup (SrTiO3 10x10x10, 300 K, 1 fs, skin 0.5), tools/ab_rewrites.py --pack in both build orders.
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
nb=fastmace.neighbors.skin
note "omat0_small_rescreen on $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit 2>/dev/null); checkpoint $(sha256sum "$ck" | cut -c1-16)"

p="noblockorder|$sf:BLOCK_ORDER=False;nosort|$nb:SORT_BY_LENGTH=False;nojoined|$cs:JOINED_LAYOUT=False"
p="$p;prefetch|$sf:PREFETCH_AHEAD=True;mb8|$sf:R128=False $sf:THREADS=128 $sf:MIN_BLOCKS=8"
p="$p;wshoff|$wp:STAGE_SH=False;alphaoff|$wp:FOLD_ALPHA=False;tile6432|$tf:TILE='64x32'"

for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  note "start pack-$order"
  python tools/ab_rewrites.py --output "$out/ab-$order" --checkpoint "$ck" --model "$ck" \
    --structures "$st@10x10x10" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
    --warmup 200 --block 256 --blocks 6 --order "$order" --pattern "$pattern" \
    --pack "$p" > "$out/ab-$order.log" 2>&1 < /dev/null
  note "end pack-$order exit=$?"
  grep -h -E 'b/a' "$out/ab-$order.log" | sed "s/^/  $order /" >> "$log"
done

note "start breakdown"
python tools/step_breakdown.py --output "$out/breakdown" --checkpoint "$ck" --model omat0-small \
  --structure "$st@10x10x10" --temperature-K 300 --timestep-fs 1 --skin 0.5 --equilibrate-steps 100 --steps 200 \
  > "$out/breakdown.log" 2>&1 < /dev/null
note "end breakdown exit=$? $(grep -hoE '"(wall_ms_per_step|force_graph_ms_per_replay)": [0-9.]+' "$out/breakdown.log" | tr '\n' ' ')"
rm -f "$out"/breakdown/trace.json
note done
