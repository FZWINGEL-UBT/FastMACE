#!/usr/bin/env bash
# rewrites.sc_generated (Codex v3 proposal 5, 9 November 2026) against the H200 defaults: one container with
#   0. tests/gpu/test_sc_generated_cuda.py (the generated pair against cuEq on the divided features, eager and
#      compiled);
#   1. tools/sc_generated_check.py: the Benchmark_loup build (SrTiO3 10x10x10) routes product 0 to the generated
#      kernels; energy, forces and stress on a copy rattled by 0.05 A against SC_GENERATED off, next to the same
#      build's run-to-run spread;
#   2. the force/stress gate (rounding class: another summation order), SrTiO3 3x3x3, NaCl, NbCl5;
#   3. the MD-step A/B a = H200 defaults, b = SC_GENERATED on, both build orders (skin 0.5, 1 fs).
# Steps 2 and 3 are skipped when 0 or 1 fails. Environment: OUT (the job sets it), EXPECT_GPU (default H200),
# BLOCKS (default 6), GATE (default 1).
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$real" > "$out/gpu.txt"
if ! echo "$real" | grep -qE "${EXPECT_GPU:-H200}"; then note "WRONG-GPU $real"; exit 3; fi
ck=${CHECKPOINT:-$FASTMACE_MODELS/mace-omat-0-medium.model}
st=results/loup-a100-20261103/SrTiO3.cif
on="fastmace.rewrites.sc_generated:SC_GENERATED=True"
off="fastmace.rewrites.sc_generated:SC_GENERATED=False"
note "sc_generated_job on $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"

note "start pytest"
python -m pytest -q -p no:cacheprovider tests/gpu/test_sc_generated_cuda.py > "$out/pytest.log" 2>&1 < /dev/null
code=$?
note "end pytest exit=$code $(tail -n 1 "$out/pytest.log" | cut -c1-200)"
if [ "$code" != 0 ]; then note "stop: test failed"; exit 1; fi

note "start check"
python tools/sc_generated_check.py --checkpoint "$ck" --structure "$st@10x10x10" --output "$out/check.json" \
  --constant "$on" --compare-constant "$off" --rattle 0.05 --source "$out/sc_generated.cu" \
  > "$out/check.log" 2>&1 < /dev/null
code=$?
note "end check exit=$code $(grep -hoE '"(in_use|forward_registers|backward_registers|forward_local_bytes|backward_local_bytes|max_abs|same_build_max_abs)": [^,]+' "$out/check.log" | tr -d '\n ' | cut -c1-400)"
if [ "$code" != 0 ]; then note "stop: not in use"; exit 1; fi

if [ "${GATE:-1}" != 0 ]; then
  note "start gate"
  python tools/force_stress_gate.py --output "$out/gate" --engine fastmace --model omat0-medium \
    --checkpoint "$ck" --structures-root "$FASTMACE_STRUCTURES" --structures "$st@3x3x3" NaCl NbCl5 \
    --production-constant "$on" > "$out/gate.log" 2>&1 < /dev/null
  code=$?
  note "end gate exit=$code $(tail -n 3 "$out/gate/summary.txt" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
  if [ "$code" != 0 ]; then note "stop: gate failed"; exit 1; fi
fi

for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  note "start ab-$order"
  python tools/ab_rewrites.py --output "$out/ab-$order" --checkpoint "$ck" --model omat0-medium \
    --structures "$st@10x10x10" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
    --warmup 200 --block 256 --blocks "${BLOCKS:-6}" --order "$order" --pattern "$pattern" \
    --b-constant "$on" > "$out/ab-$order.log" 2>&1 < /dev/null
  note "end ab-$order exit=$? $(grep -h -E 'b/a' "$out/ab-$order.log" | tail -n 1 | cut -c1-240)"
done
note done
