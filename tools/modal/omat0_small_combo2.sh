#!/usr/bin/env bash
# OMAT-0 small on the H200, 9 November 2026: the rescreen's three wins on small's defaults (main c8d08b5b + w'
# STAGE_EDGES), combined. a = small's defaults; b = MIN_BLOCKS 8 (spline forward at 128 threads, R128 off), with
# w' FOLD_ALPHA off and / or tf32x3 TILE 64x32. Benchmark_loup, both orders; then the 3x force/stress gate and the
# step breakdown with all three on.
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
tf=fastmace.kernels.cutlass.tf32x3
note "omat0_small_combo2 on $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit 2>/dev/null); checkpoint $(sha256sum "$ck" | cut -c1-16)"

mb8="$sf:R128=False $sf:THREADS=128 $sf:MIN_BLOCKS=8"
alpha="$wp:FOLD_ALPHA=False"
tile="$tf:TILE='64x32'"
p="mb8|$mb8;mb8alpha|$mb8 $alpha;mb8tile|$mb8 $tile;all3|$mb8 $alpha $tile;alphatile|$alpha $tile"

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

note "start gate"
# shellcheck disable=SC2086
python tools/force_stress_gate.py --output "$out/gate" --engine fastmace --model omat0-small \
  --checkpoint "$ck" --structures-root "$FASTMACE_STRUCTURES" --structures "$st@3x3x3" NaCl NbCl5 \
  --production-constant $mb8 $alpha $tile > "$out/gate.log" 2>&1 < /dev/null
note "end gate exit=$?"
grep -hE "summary" "$out/gate.log" | tail -n 3 | sed 's/^/  /' >> "$log"

note "start breakdown"
# shellcheck disable=SC2086
python tools/step_breakdown.py --output "$out/breakdown" --checkpoint "$ck" --model omat0-small \
  --structure "$st@10x10x10" --temperature-K 300 --timestep-fs 1 --skin 0.5 --equilibrate-steps 100 --steps 200 \
  --constant $mb8 $alpha $tile > "$out/breakdown.log" 2>&1 < /dev/null
note "end breakdown exit=$? $(grep -hoE '"(wall_ms_per_step|force_graph_ms_per_replay)": [0-9.]+' "$out/breakdown.log" | tr '\n' ' ')"
rm -f "$out"/breakdown/trace.json
note done
