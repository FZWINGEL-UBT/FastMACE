#!/usr/bin/env bash
# BLOCK_ORDER (8 November 2026): the SPLINE_W forward walks the receivers grouped by species, bound by the force field
# (lab on the H200 defaults with SORT_BY_LENGTH: forward 717 -> 624 us, bitwise; results/slab-sorted-h200-20261108).
#   1. tools/spline_w_check.py: forces with BLOCK_ORDER against the defaults next to the same build's spread, and
#      that the order is bound;
#   2. MD-step A/B on Benchmark_loup (SrTiO3 10x10x10, 300 K, 1 fs, skin 0.5), a = H200 defaults, b = BLOCK_ORDER,
#      both build orders.
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
s=fastmace.kernels.oeq_streaming.spline_forward
note "block_order_job on $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"

note "start check"
python tools/spline_w_check.py --checkpoint "$ck" --structure "$st@10x10x10" --output "$out/check.json" \
  --constant "$s:BLOCK_ORDER=True" --rattle 0.05 --compare-constant "$s:BLOCK_ORDER=False" > "$out/check.log" 2>&1 < /dev/null
code=$?
note "end check exit=$code $(grep -hoE '"(forward_block_order|forward_order_bound|max_abs|same_build_max_abs)": [^,]+' "$out/check.log" | tr -d '\n ' | cut -c1-300)"
[ "$code" = 0 ] || { note "check failed: no A/B"; exit 1; }

for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  note "start ab-order-$order"
  python tools/ab_rewrites.py --output "$out/ab-order-$order" --checkpoint "$ck" --model omat0-medium \
    --structures "$st@10x10x10" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
    --warmup 200 --block 256 --blocks "${BLOCKS:-6}" --order "$order" --pattern "$pattern" \
    --b-constant "$s:BLOCK_ORDER=True" > "$out/ab-order-$order.log" 2>&1 < /dev/null
  note "end ab-order-$order exit=$? $(grep -h -E 'b/a' "$out/ab-order-$order.log" | tail -n 1 | cut -c1-240)"
done
note done
