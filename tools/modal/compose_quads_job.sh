#!/usr/bin/env bash
# H200 ideas after SPLINE_W (7 November 2026), ideas 1 and 2 in one container (resource framework: packed):
#   0. tools/spline_w_check.py with STAGE_QUADS on (Benchmark_loup build, SrTiO3 10x10x10): the spline and the
#      quad reads must be in use (the job stops otherwise); it dumps the spline kernels' sources, the w' kernel's
#      registers / local / shared bytes, and compares the forces with a build without STAGE_QUADS (bitwise expected);
#   1. the MD-step A/B a = H200 defaults, b = STAGE_QUADS on, both build orders (ab_rewrites; skin 0.5, 1 fs);
#   2. tools/compose_lab.py: interaction 1's projection + species skip, production chain against the composed
#      per-species product (species_mm launches) and the species-sorted floor, forward + backward in CUDA graphs.
# Environment: OUT (the job sets it), EXPECT_GPU (default H200), BLOCKS (default 6).
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$real" > "$out/gpu.txt"
if ! echo "$real" | grep -qE "${EXPECT_GPU:-H200}"; then note "WRONG-GPU $real"; exit 3; fi
ck=${CHECKPOINT:-$FASTMACE_MODELS/mace-omat-0-medium.model}
st=results/loup-a100-20261103/SrTiO3.cif
w=fastmace.kernels.oeq_streaming.wprime_backward
note "compose_quads_job on $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"

note "start check-quads"
python tools/spline_w_check.py --checkpoint "$ck" --structure "$st@10x10x10" --output "$out/check-quads.json" \
  --constant "$w:STAGE_QUADS=True" --sources "$out/sources" --compare-constant "$w:STAGE_QUADS=False" \
  > "$out/check-quads.log" 2>&1 < /dev/null
code=$?
note "end check-quads exit=$code $(grep -hoE '"(spline_in_use|wprime_stage_quads|bitwise|max_abs)": [^,]+' "$out/check-quads.log" | tr '\n' ' ' | cut -c1-240)"
if [ "$code" != 0 ]; then note "stop: spline not in use"; exit 4; fi
if ! grep -q '"wprime_stage_quads": true' "$out/check-quads.log"; then note "stop: STAGE_QUADS not in use"; exit 5; fi

for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  note "start ab-quads-$order"
  python tools/ab_rewrites.py --output "$out/ab-quads-$order" --checkpoint "$ck" --model omat0-medium \
    --structures "$st@10x10x10" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
    --warmup 200 --block 256 --blocks "${BLOCKS:-6}" --order "$order" --pattern "$pattern" \
    --b-constant "$w:STAGE_QUADS=True" > "$out/ab-quads-$order.log" 2>&1 < /dev/null
  note "end ab-quads-$order exit=$? $(grep -h -E 'b/a' "$out/ab-quads-$order.log" | tail -n 1 | cut -c1-240)"
done

note "start compose-lab"
python tools/compose_lab.py --output "$out/compose-lab.json" > "$out/compose-lab.log" 2>&1 < /dev/null
note "end compose-lab exit=$? $(grep -hE '^(old|composed-[a-z]+|sorted) ' "$out/compose-lab.log" | cut -c1-90 | tr '\n' ' ')"
note done
