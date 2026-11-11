#!/usr/bin/env bash
# H200 pack (11 November 2026): tf32x3.HYBRID and spline_forward.GEOM, neither measured on the H200 yet, in ONE
# container on OMAT-0 medium:
#   1. MD-step A/B, a = the H200's default plan, b = one switch on, both build orders (ab: abab, ba: abba), SrTiO3
#      10x10x10 and NbCl5 3x3x3, 1 fs, skin 0.5, warm-up 200, 6 blocks of 256 steps per side. One
#      tools/ab_rewrites.py --pack process per order builds side a once for both switches (as alex_ab_pack v2).
#   2. The force/stress gate (tools/force_stress_gate.py, 3x the float32 error) of each switch; SrTiO3 3x3x3, NaCl,
#      NbCl5. The default plan's own gate is left out (decision 66 needs the switch's gate to pass; it keeps the pack
#      within the month's remaining Modal credit).
# Every step runs whatever the others give (one pack, no relaunch). Stop rule per switch: not faster in both orders on
# both structures, or the gate fails.
#   modal run tools/modal/fastmace_modal.py::job --gpu H200 --timeout-min 50 --tarball <slim tar> \
#     --command "bash tools/modal/h200_hybrid_geom.sh" --out <dir>
# Environment: OUT (the job sets it), EXPECT_GPU (default H200), BLOCKS (6), DRY (1: print the commands only).
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" | tee -a "$log"; }
run() { if [ "${DRY:-0}" = 1 ]; then echo "DRY $*" >> "$log"; else "$@"; fi; }
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$real" > "$out/gpu.txt"
if ! echo "$real" | grep -qE "${EXPECT_GPU:-H200}"; then note "WRONG-GPU $real"; exit 3; fi
ck=${CHECKPOINT:-$FASTMACE_MODELS/mace-omat-0-medium.model}
st=results/loup-a100-20261103/SrTiO3.cif
hybrid="fastmace.kernels.cutlass.tf32x3:HYBRID=True"
geom="fastmace.kernels.oeq_streaming.spline_forward:GEOM=True"
note "h200_hybrid_geom on $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"
failed=0

for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  note "start ab-$order"
  t0=$(date +%s)
  run python tools/ab_rewrites.py --output "$out/ab-$order" --checkpoint "$ck" --model omat0-medium \
    --structures "$st@10x10x10" "NbCl5@3x3x3" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
    --warmup 200 --block 256 --blocks "${BLOCKS:-6}" --order "$order" --pattern "$pattern" \
    --pack "hybrid|$hybrid;geom|$geom" > "$out/ab-$order.log" 2>&1 < /dev/null
  code=$?
  [ "$code" = 0 ] || failed=$((failed + 1))
  note "end ab-$order exit=$code wall $(( $(date +%s) - t0 )) s"
  grep -h -E 'b/a' "$out/ab-$order.log" | sed "s/^/  $order /" | tee -a "$log"
done

gate() {
  local name=$1; shift
  note "start gate-$name"
  run python tools/force_stress_gate.py --output "$out/gate-$name" --engine fastmace --model omat0-medium \
    --checkpoint "$ck" --structures-root "$FASTMACE_STRUCTURES" --structures "$st@3x3x3" NaCl NbCl5 "$@" \
    > "$out/gate-$name.log" 2>&1 < /dev/null
  local code=$?
  [ "$code" = 0 ] || failed=$((failed + 1))
  note "end gate-$name exit=$code $(tail -n 4 "$out/gate-$name/summary.txt" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
}
gate hybrid --production-constant "$hybrid"
gate geom --production-constant "$geom"
note "done failed=$failed"
[ "$failed" = 0 ]
