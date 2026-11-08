#!/usr/bin/env bash
# COMPOSE_SKIP on the H200 against the H200 defaults (7 November 2026): one container (resource framework: packed) with
#   0. tools/spline_w_check.py with COMPOSE_SKIP on (Benchmark_loup build, SrTiO3 10x10x10): interaction 1 must
#      compose its skip (the job stops otherwise); forces on a copy rattled by 0.05 A against a build without it, and
#      the same build twice (the spread from atomics);
#   1. the force/stress gate of COMPOSE_SKIP=True (rounding-class change; the job stops if it fails);
#   2. the MD-step A/B a = H200 defaults, b = COMPOSE_SKIP on, both build orders (skin 0.5, 1 fs).
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
switch=fastmace.kernels.triton.compose_skip:COMPOSE_SKIP=True
note "compose_skip_job on $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"

note "start check-compose"
python tools/spline_w_check.py --checkpoint "$ck" --structure "$st@10x10x10" --output "$out/check-compose.json" \
  --constant "$switch" --require-compose --rattle 0.05 \
  --compare-constant fastmace.kernels.triton.compose_skip:COMPOSE_SKIP=False > "$out/check-compose.log" 2>&1 < /dev/null
code=$?
note "end check-compose exit=$code $(grep -hoE '"(compose_skip_in_use|spline_in_use|max_abs|same_build_max_abs|max_force)": [^,]+' "$out/check-compose.log" | tr '\n' ' ' | cut -c1-240)"
if [ "$code" != 0 ]; then note "stop: not in use"; exit 4; fi

note "start gate-compose"
python tools/force_stress_gate.py --output "$out/gate-compose" --engine fastmace --model omat0-medium \
  --checkpoint "$ck" --structures-root "$FASTMACE_STRUCTURES" --structures "$st@3x3x3" NaCl NbCl5 \
  --production-constant "$switch" > "$out/gate-compose.log" 2>&1 < /dev/null
code=$?
note "end gate-compose exit=$code $(grep -h -iE 'verdict' "$out/gate-compose.log" | tail -n 1 | cut -c1-200)"
if [ "$code" != 0 ]; then note "stop: gate failed"; exit 4; fi

for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  note "start ab-compose-$order"
  python tools/ab_rewrites.py --output "$out/ab-compose-$order" --checkpoint "$ck" --model omat0-medium \
    --structures "$st@10x10x10" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
    --warmup 200 --block 256 --blocks "${BLOCKS:-6}" --order "$order" --pattern "$pattern" \
    --b-constant "$switch" > "$out/ab-compose-$order.log" 2>&1 < /dev/null
  note "end ab-compose-$order exit=$? $(grep -h -E 'b/a' "$out/ab-compose-$order.log" | tail -n 1 | cut -c1-240)"
done
note done
