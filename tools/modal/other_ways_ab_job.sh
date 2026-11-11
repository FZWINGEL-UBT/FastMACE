#!/usr/bin/env bash
# GEOM (spline_forward) and SC_FACTORED (rewrites.sc_generated) against the A100 defaults (results/other-ways-20261109
# sections 3 and 4), one container:
#   1. tools/sc_generated_check.py: the Benchmark_loup build (SrTiO3 10x10x10) with both on routes product 0 to the
#      factored kernels and the spline forwards to GEOM; energy, forces and stress on a copy rattled by 0.05 A against
#      both off, next to the same build's run-to-run spread;
#   2. the force/stress gate (rounding class: another summation order) with both on, SrTiO3 3x3x3, NaCl, NbCl5;
#   3. the MD-step A/B a = the defaults, b = one switch on, both build orders (skin 0.5, 1 fs), GEOM then
#      SC_FACTORED.
# Steps 2 and 3 are skipped when 1 fails; step 3 when 2 fails. Environment: OUT (the job sets it), EXPECT_GPU (default
# A100), BLOCKS (default 6), GATE (default 1), SWITCHES (default "geom fac").
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$real" > "$out/gpu.txt"
if ! echo "$real" | grep -qE "${EXPECT_GPU:-A100}"; then note "WRONG-GPU $real"; exit 3; fi
ck=${CHECKPOINT:-$FASTMACE_MODELS/mace-omat-0-medium.model}
st=results/loup-a100-20261103/SrTiO3.cif
geom="fastmace.kernels.oeq_streaming.spline_forward:GEOM"
fac="fastmace.rewrites.sc_generated:SC_FACTORED"
note "other_ways_ab_job on $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"

note "start check"
python tools/sc_generated_check.py --checkpoint "$ck" --structure "$st@10x10x10" --output "$out/check.json" \
  --constant "$geom=True" "$fac=True" --compare-constant "$geom=False" "$fac=False" --rattle 0.05 \
  --source "$out/sc_factored.cu" > "$out/check.log" 2>&1 < /dev/null
code=$?
note "end check exit=$code $(grep -hoE '"(in_use|factored|geom|spline_forwards|forward_registers|backward_registers|backward_local_bytes|max_abs|same_build_max_abs)": [^,]+' "$out/check.log" | tr -d '\n ' | cut -c1-500)"
if [ "$code" != 0 ] || ! grep -q '"factored": true' "$out/check.log" || grep -q '"geom": 0' "$out/check.log"; then
  note "stop: not in use"; exit 1
fi

if [ "${GATE:-1}" != 0 ]; then
  note "start gate"
  python tools/force_stress_gate.py --output "$out/gate" --engine fastmace --model omat0-medium \
    --checkpoint "$ck" --structures-root "$FASTMACE_STRUCTURES" --structures "$st@3x3x3" NaCl NbCl5 \
    --production-constant "$geom=True" "$fac=True" > "$out/gate.log" 2>&1 < /dev/null
  code=$?
  note "end gate exit=$code $(tail -n 3 "$out/gate/summary.txt" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
  if [ "$code" != 0 ]; then note "stop: gate failed"; exit 1; fi
fi

for sw in ${SWITCHES:-geom fac}; do
  constant=$([ "$sw" = geom ] && echo "$geom=True" || echo "$fac=True")
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    note "start ab-$sw-$order"
    python tools/ab_rewrites.py --output "$out/ab-$sw-$order" --checkpoint "$ck" --model omat0-medium \
      --structures "$st@10x10x10" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
      --warmup 200 --block 256 --blocks "${BLOCKS:-6}" --order "$order" --pattern "$pattern" \
      --b-constant "$constant" > "$out/ab-$sw-$order.log" 2>&1 < /dev/null
    note "end ab-$sw-$order exit=$? $(grep -h -E 'b/a' "$out/ab-$sw-$order.log" | tail -n 1 | cut -c1-240)"
  done
done
note done
