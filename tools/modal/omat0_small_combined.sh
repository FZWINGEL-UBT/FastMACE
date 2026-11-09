#!/usr/bin/env bash
# OMAT-0 small on the H200: the screen's four small-specific wins combined (owner, 9 November 2026: "Yes, combine
# them"). a = small's defaults with JOINED_LAYOUT on (being adopted for the H200 on fastmace/joined-layout);
# b = a plus: w' STAGE_EDGES on, spline forward R128 off with 128 threads and MIN_BLOCKS 8, w' FOLD_ALPHA off,
# tf32x3 TILE 64x32. Benchmark_loup (SrTiO3 10x10x10, 300 K, 1 fs, skin 0.5).
#   1. pack, both orders: all four, and each leave-one-out (all but one);
#   2. the 3x force/stress gate with all of them (and JOINED_LAYOUT) on;
#   3. tools/step_breakdown.py with all of them on.
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
note "omat0_small_combined on $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit 2>/dev/null); checkpoint $(sha256sum "$ck" | cut -c1-16)"

joined="$cs:JOINED_LAYOUT=True"
wedges="$wp:STAGE_EDGES=True"
mb8="$sf:R128=False $sf:THREADS=128 $sf:MIN_BLOCKS=8"
alpha="$wp:FOLD_ALPHA=False"
tile="$tf:TILE='64x32'"
p="all|$wedges $mb8 $alpha $tile"
p="$p;nowedges|$mb8 $alpha $tile;nomb8|$wedges $alpha $tile;noalpha|$wedges $mb8 $tile;notile|$wedges $mb8 $alpha"

for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  note "start pack-$order"
  # shellcheck disable=SC2086
  python tools/ab_rewrites.py --output "$out/ab-$order" --checkpoint "$ck" --model "$ck" \
    --structures "$st@10x10x10" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
    --warmup 200 --block 256 --blocks 6 --order "$order" --pattern "$pattern" \
    --a-constant $joined --b-constant $joined --pack "$p" > "$out/ab-$order.log" 2>&1 < /dev/null
  note "end pack-$order exit=$?"
  grep -h -E 'b/a' "$out/ab-$order.log" | sed "s/^/  $order /" >> "$log"
done

note "start gate"
# shellcheck disable=SC2086
python tools/force_stress_gate.py --output "$out/gate" --engine fastmace --model omat0-small \
  --checkpoint "$ck" --structures-root "$FASTMACE_STRUCTURES" --structures "$st@3x3x3" NaCl NbCl5 \
  --production-constant $joined $wedges $mb8 $alpha $tile > "$out/gate.log" 2>&1 < /dev/null
note "end gate exit=$?"
grep -hE "summary" "$out/gate.log" | tail -n 3 | sed 's/^/  /' >> "$log"

note "start breakdown"
# shellcheck disable=SC2086
python tools/step_breakdown.py --output "$out/breakdown" --checkpoint "$ck" --model omat0-small \
  --structure "$st@10x10x10" --temperature-K 300 --timestep-fs 1 --skin 0.5 --equilibrate-steps 100 --steps 200 \
  --constant $joined $wedges $mb8 $alpha $tile > "$out/breakdown.log" 2>&1 < /dev/null
note "end breakdown exit=$? $(grep -hoE '"(wall_ms_per_step|force_graph_ms_per_replay)": [0-9.]+' "$out/breakdown.log" | tr '\n' ' ')"
rm -f "$out"/breakdown/trace.json
note done
