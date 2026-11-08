#!/usr/bin/env bash
# SORT_BY_LENGTH (8 November 2026): the skin list orders each receiver's candidates (within its sender group) by length
# at the build, so the SPLINE_W forward's neighbouring receivers share table slabs (lab: forward 792 -> 680 us,
# results/omat0-breakthroughs-assessment-20261108). Rounding-class (summation order within a receiver).
#   1. MD-step A/B on Benchmark_loup (SrTiO3 10x10x10, 300 K, 1 fs, skin 0.5), a = H200 defaults, b = SORT_BY_LENGTH,
#      both build orders;
#   2. tools/force_stress_gate.py with the switch as the production constant (SrTiO3 3x3x3, NaCl, NbCl5).
# Environment: OUT (the job sets it), EXPECT_GPU (default H200), BLOCKS (default 6), MODEL (default omat0-medium).
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$real" > "$out/gpu.txt"
if ! echo "$real" | grep -qE "${EXPECT_GPU:-H200}"; then note "WRONG-GPU $real"; exit 3; fi
model=${MODEL:-omat0-medium}
ck=${CHECKPOINT:-$FASTMACE_MODELS/mace-omat-0-medium.model}
st=results/loup-a100-20261103/SrTiO3.cif
on="fastmace.neighbors.skin:SORT_BY_LENGTH=True"
note "slab_sorted_job on $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"

for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  note "start ab-sorted-$order"
  python tools/ab_rewrites.py --output "$out/ab-sorted-$order" --checkpoint "$ck" --model "$model" \
    --structures "$st@10x10x10" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
    --warmup 200 --block 256 --blocks "${BLOCKS:-6}" --order "$order" --pattern "$pattern" \
    --b-constant "$on" > "$out/ab-sorted-$order.log" 2>&1 < /dev/null
  note "end ab-sorted-$order exit=$? $(grep -h -E 'b/a' "$out/ab-sorted-$order.log" | tail -n 1 | cut -c1-240)"
done

note "start gate-sorted"
python tools/force_stress_gate.py --output "$out/gate-sorted" --engine fastmace --model "$model" \
  --checkpoint "$ck" --structures-root "$FASTMACE_STRUCTURES" --structures "$st@3x3x3" NaCl NbCl5 \
  --production-constant "$on" > "$out/gate-sorted.log" 2>&1 < /dev/null
note "end gate-sorted exit=$?"
grep -hE "verdict|PASS|FAIL|ratio" "$out/gate-sorted.log" | tail -n 20 | sed 's/^/  /' >> "$log"
note done
