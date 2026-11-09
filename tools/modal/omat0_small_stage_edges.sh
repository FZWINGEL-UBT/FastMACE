#!/usr/bin/env bash
# OMAT-0 small on the H200: confirm the new per-model default w' STAGE_EDGES (STAGE_EDGES_MODEL_GPUS, 9 November
# 2026; owner: "Yes, add it"). a = small's defaults (STAGE_EDGES on through the label), b = a with STAGE_EDGES=False.
# b/a above 1: the default is faster. Benchmark_loup (SrTiO3 10x10x10, 300 K, 1 fs, skin 0.5), both build orders,
# plus tools/step_breakdown.py with no constants (the plan's defaults).
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$real" > "$out/gpu.txt"
if ! echo "$real" | grep -qE "${EXPECT_GPU:-H200}"; then note "WRONG-GPU $real"; exit 3; fi
ck=${CHECKPOINT:-$FASTMACE_MODELS/mace-omat-0-small.model}
st=results/loup-a100-20261103/SrTiO3.cif
wp=fastmace.kernels.oeq_streaming.wprime_backward
note "omat0_small_stage_edges on $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit 2>/dev/null); checkpoint $(sha256sum "$ck" | cut -c1-16)"
python -c "import torch; from fastmace.kernels.oeq_streaming import wprime_backward as w; print('stage_edges_default', w.stage_edges_default(torch.device('cuda'), 'omat0-small'), w.stage_edges_default(torch.device('cuda'), 'omat0-medium'))" >> "$log" 2>&1

for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  note "start pack-$order"
  python tools/ab_rewrites.py --output "$out/ab-$order" --checkpoint "$ck" --model "$ck" \
    --structures "$st@10x10x10" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
    --warmup 200 --block 256 --blocks 6 --order "$order" --pattern "$pattern" \
    --pack "edgesoff|$wp:STAGE_EDGES=False" > "$out/ab-$order.log" 2>&1 < /dev/null
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
