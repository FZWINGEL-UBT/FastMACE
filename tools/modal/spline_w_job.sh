#!/usr/bin/env bash
# SPLINE_W on the H200 (owner, 7 November 2026): one container (resource framework: packed) with
#   1. the force/stress gate of SPLINE_W=True (gate_pack's command; the job stops here if it fails, or if the plan
#      does not use the spline: a broken build then costs one gate);
#   2. the MD-step A/B, a = default, b = SPLINE_W=True, both build orders (ab_pack's command);
#   3. Benchmark_loup itself (tools/bench/loup_md.py: Langevin 300 K, friction 0.002 / fs, 1 fs, skin 0.5, seed 1,
#      from rest, SrTiO3 10x10x10), fresh processes in the order default, spline, spline, default;
#   4. step breakdowns (tools/step_breakdown.py main pass, 300 K, 1 fs, skin 0.5) of the default plan and of
#      SPLINE_W=True.
# Environment: OUT (the job sets it), EXPECT_GPU (default H200), BLOCKS (default 8), STEPS (breakdown, default 300).
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$real" > "$out/gpu.txt"
if ! echo "$real" | grep -qE "${EXPECT_GPU:-H200}"; then note "WRONG-GPU $real"; exit 3; fi
ck=${CHECKPOINT:-$FASTMACE_MODELS/mace-omat-0-medium.model}
st=results/loup-a100-20261103/SrTiO3.cif
switch=fastmace.kernels.oeq_streaming.spline_forward:SPLINE_W=True
note "spline_w_job on $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"

note "start gate-spline"
python tools/force_stress_gate.py --output "$out/gate-spline" --engine fastmace --model omat0-medium \
  --checkpoint "$ck" --structures-root "$FASTMACE_STRUCTURES" --structures "$st@3x3x3" NaCl NbCl5 \
  --production-constant "$switch" > "$out/gate-spline.log" 2>&1 < /dev/null
code=$?
note "end gate-spline exit=$code $(grep -h -iE 'verdict' "$out/gate-spline.log" | tail -n 1 | cut -c1-200)"
used=$(grep -rhoE "spline [0-9]+ intervals, certificate [0-9.]+" "$out/gate-spline" "$out/gate-spline.log" 2>/dev/null | sort -u | head -n 3 | tr '\n' ' ')
note "spline in the plan: ${used:-NONE}"
if [ "$code" != 0 ] || [ -z "$used" ]; then note "stop: gate failed or spline not used"; exit 4; fi

for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  note "start ab-spline-$order"
  python tools/ab_rewrites.py --output "$out/ab-spline-$order" --checkpoint "$ck" --model omat0-medium \
    --structures "$st@10x10x10" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
    --warmup 200 --block 256 --blocks "${BLOCKS:-8}" --order "$order" --pattern "$pattern" --b-constant "$switch" \
    > "$out/ab-spline-$order.log" 2>&1 < /dev/null
  note "end ab-spline-$order exit=$? $(grep -h -E 'b/a' "$out/ab-spline-$order.log" | tail -n 1 | cut -c1-240)"
done

run=0
for side in default spline spline default; do
  run=$((run + 1))
  extra=()
  [ "$side" = spline ] && extra=(--constant "$switch")
  note "start loup-$run-$side"
  python tools/bench/loup_md.py --structure "$st@10x10x10" --checkpoint "$ck" --ensemble langevin --skin 0.5 \
    --warmup 300 --steps 256 --blocks 8 --output "$out/loup-$run-$side.json" "${extra[@]}" \
    > "$out/loup-$run-$side.log" 2>&1 < /dev/null
  note "end loup-$run-$side exit=$? $(tail -n 3 "$out/loup-$run-$side.log" | tr '\n' ' ' | cut -c1-240)"
done

for side in default spline; do
  extra=()
  [ "$side" = spline ] && extra=(--constant "$switch")
  note "start profile-$side"
  python tools/step_breakdown.py --output "$out/profile-$side" --checkpoint "$ck" --model omat0-medium \
    --backend hybrid --timestep-fs 1 --temperature-K 300 --skin 0.5 --steps "${STEPS:-300}" --structure "$st@10x10x10" "${extra[@]}" \
    > "$out/profile-$side.log" 2>&1 < /dev/null
  code=$?
  rm -f "$out/profile-$side/trace.json"
  note "end profile-$side exit=$code"
done
note done
