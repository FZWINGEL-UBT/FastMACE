#!/usr/bin/env bash
# VIRIAL_ELEMENTWISE (rewrites.engine) against the defaults, one container:
#   1. the strain self-check and layering tests (CPU, pytest);
#   2. the force/stress gate (rounding class: another summation order of the strain gradient) with the switch on,
#      SrTiO3 3x3x3, NaCl, NbCl5;
#   3. tools/virial_ab.py, the force call with stress, a = off, b = on, both build orders, SrTiO3 10x10x10 and
#      18x18x18 (VIRIAL_SIZES).
# Step 3 is skipped when 2 fails. Environment: OUT (the job sets it), EXPECT_GPU (default A100), GATE (default 1),
# VIRIAL_SIZES (default "10x10x10 18x18x18"), BLOCKS (default 6).
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$real" > "$out/gpu.txt"
if ! echo "$real" | grep -qE "${EXPECT_GPU:-A100}"; then note "WRONG-GPU $real"; exit 3; fi
ck=${CHECKPOINT:-$FASTMACE_MODELS/mace-omat-0-medium.model}
st=results/loup-a100-20261103/SrTiO3.cif
sw="fastmace.rewrites.engine:VIRIAL_ELEMENTWISE"
note "virial_ab_job on $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"

note "start tests"
CUDA_VISIBLE_DEVICES= python -m pytest -q -p no:cacheprovider tests/cpu/test_strain_per_model.py tests/cpu/test_layering.py \
  > "$out/tests.log" 2>&1 < /dev/null
note "end tests exit=$? $(tail -n 1 "$out/tests.log")"

if [ "${GATE:-1}" != 0 ]; then
  note "start gate"
  python tools/force_stress_gate.py --output "$out/gate" --engine fastmace --model omat0-medium \
    --checkpoint "$ck" --structures-root "$FASTMACE_STRUCTURES" --structures "$st@3x3x3" NaCl NbCl5 \
    --production-constant "$sw=True" > "$out/gate.log" 2>&1 < /dev/null
  code=$?
  note "end gate exit=$code"
  if [ "$code" != 0 ]; then note "stop: gate failed"; exit 1; fi
fi

for size in ${VIRIAL_SIZES:-10x10x10 18x18x18}; do
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    note "start virial-$size-$order"
    python tools/virial_ab.py --checkpoint "$ck" --structure "$st@$size" --output "$out/virial-$size-$order.json" \
      --order "$order" --pattern "$pattern" --blocks "${BLOCKS:-6}" --calls 20 > "$out/virial-$size-$order.log" 2>&1 < /dev/null
    note "end virial-$size-$order exit=$? $(tail -n 1 "$out/virial-$size-$order.log" | cut -c1-600)"
  done
done
note done
