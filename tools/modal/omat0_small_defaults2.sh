#!/usr/bin/env bash
# OMAT-0 small on the H200, 9 November 2026: confirm its new per-model defaults (the spline forward's R128 shape with
# MIN_BLOCKS 8, R128_MIN_BLOCKS_MODEL_GPUS; w' FOLD_ALPHA off, FOLD_ALPHA_MODEL_OFF_GPUS). a = small's defaults (no
# constants); b = each one back to the shared default, both, and w' ESTRIN on (closed while FOLD_ALPHA was on).
# b/a above 1: the default is faster. Benchmark_loup, both orders; then the 3x gate and the step breakdown on the
# plan's defaults.
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
note "omat0_small_defaults2 on $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit 2>/dev/null); checkpoint $(sha256sum "$ck" | cut -c1-16)"
python -c "import torch; from fastmace.kernels.oeq_streaming import spline_forward as s, wprime_backward as w; d = torch.device('cuda'); print('defaults small', s.launch_shape(d, 'omat0-small'), w.fold_alpha_default(d, 'omat0-small'), w.stage_edges_default(d, 'omat0-small'), 'medium', s.launch_shape(d, 'omat0-medium'), w.fold_alpha_default(d, 'omat0-medium'))" 2>&1 | grep defaults >> "$log"

mb6="$sf:R128_MIN_BLOCKS_MODEL_GPUS={}"
alpha="$wp:FOLD_ALPHA_MODEL_OFF_GPUS={}"
p="mb6|$mb6;alphaon|$alpha;both|$mb6 $alpha;estrin|$wp:ESTRIN=True"

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
python tools/force_stress_gate.py --output "$out/gate" --engine fastmace --model omat0-small \
  --checkpoint "$ck" --structures-root "$FASTMACE_STRUCTURES" --structures "$st@3x3x3" NaCl NbCl5 \
  > "$out/gate.log" 2>&1 < /dev/null
note "end gate exit=$?"
grep -hE "summary" "$out/gate.log" | tail -n 3 | sed 's/^/  /' >> "$log"

note "start breakdown"
python tools/step_breakdown.py --output "$out/breakdown" --checkpoint "$ck" --model omat0-small \
  --structure "$st@10x10x10" --temperature-K 300 --timestep-fs 1 --skin 0.5 --equilibrate-steps 100 --steps 200 \
  > "$out/breakdown.log" 2>&1 < /dev/null
note "end breakdown exit=$? $(grep -hoE '"(wall_ms_per_step|force_graph_ms_per_replay)": [0-9.]+' "$out/breakdown.log" | tr '\n' ' ')"
rm -f "$out"/breakdown/trace.json
note done
