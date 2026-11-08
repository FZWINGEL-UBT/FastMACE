#!/usr/bin/env bash
# Codex H200 follow-up items 3-5 against the H200 defaults (7 November 2026): one container (resource framework:
# packed) with
#   0. per variant tools/spline_w_check.py with the variant on (Benchmark_loup build, SrTiO3 10x10x10): forces on a
#      copy rattled by 0.05 A against the defaults, next to the same build's run-to-run spread; the forward and w'
#      kernels' registers / local bytes; the sources dumped;
#   1. the force/stress gate of FACTOR_W (the only rounding-class variant);
#   2. per variant the MD-step A/B a = H200 defaults, b = the variant, both build orders (skin 0.5, 1 fs).
# Variants: packed = the first radial table's coefficients two terms per float4 (tabled_outer_sum.PACKED_COEF);
# factor = w times a path's common factor once (spline_forward.FACTOR_W); cursor32 = int loop state
# (spline_forward.CURSOR32); locator = one spline locator per edge for both spline kernels
# (spline_forward.SHARED_LOCATOR).
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
declare -A on=([packed]="fastmace.kernels.triton.tabled_outer_sum:PACKED_COEF=True" [factor]="$s:FACTOR_W=True"
               [cursor32]="$s:CURSOR32=True" [locator]="$s:SHARED_LOCATOR=True")
declare -A off=([packed]="fastmace.kernels.triton.tabled_outer_sum:PACKED_COEF=False" [factor]="$s:FACTOR_W=False"
                [cursor32]="$s:CURSOR32=False" [locator]="$s:SHARED_LOCATOR=False")
names="packed factor cursor32 locator"
note "ideas345_job on $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"

failed=""
for name in $names; do
  note "start check-$name"
  python tools/spline_w_check.py --checkpoint "$ck" --structure "$st@10x10x10" --output "$out/check-$name.json" \
    --constant "${on[$name]}" --rattle 0.05 --sources "$out/sources-$name" --compare-constant "${off[$name]}" \
    > "$out/check-$name.log" 2>&1 < /dev/null
  code=$?
  note "end check-$name exit=$code $(grep -hoE '"(spline_in_use|forward_registers|forward_local_bytes|max_abs|same_build_max_abs|max_force)": [^,]+' "$out/check-$name.log" | tr -d '\n ' | cut -c1-240)"
  if [ "$code" != 0 ]; then failed="$failed $name"; fi
done

note "start gate-factor"
python tools/force_stress_gate.py --output "$out/gate-factor" --engine fastmace --model omat0-medium \
  --checkpoint "$ck" --structures-root "$FASTMACE_STRUCTURES" --structures "$st@3x3x3" NaCl NbCl5 \
  --production-constant "${on[factor]}" > "$out/gate-factor.log" 2>&1 < /dev/null
code=$?
note "end gate-factor exit=$code"
if [ "$code" != 0 ]; then failed="$failed factor"; fi

for name in $names; do
  case " $failed " in *" $name "*) note "skip ab-$name: check or gate failed"; continue ;; esac
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    note "start ab-$name-$order"
    python tools/ab_rewrites.py --output "$out/ab-$name-$order" --checkpoint "$ck" --model omat0-medium \
      --structures "$st@10x10x10" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
      --warmup 200 --block 256 --blocks "${BLOCKS:-6}" --order "$order" --pattern "$pattern" \
      --b-constant "${on[$name]}" > "$out/ab-$name-$order.log" 2>&1 < /dev/null
    note "end ab-$name-$order exit=$? $(grep -h -E 'b/a' "$out/ab-$name-$order.log" | tail -n 1 | cut -c1-240)"
  done
done
note done
