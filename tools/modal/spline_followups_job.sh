#!/usr/bin/env bash
# SPLINE_W follow-ups on the H200 against the H200 defaults (SPLINE_W and R128 on; 7 November 2026): one container
# (resource framework: packed) with
#   0. tools/spline_w_check.py per variant (Benchmark_loup build, SrTiO3 10x10x10): the spline must be in use, and its
#      record gives each forward kernel's registers and local (spill) bytes; the job stops if the spline is not used;
#   1. per variant the MD-step A/B a = defaults, b = the variant, both build orders (ab_pack's command; skin 0.5,
#      1 fs, as Benchmark_loup).
# Variants: vec4 = the w' kernel reads the spline table as one float4 per weight (wprime_backward.SPLINE_VECTOR=4);
# cap5 / nocap = 128 threads with MIN_BLOCKS 5 / no cap (R128's 6 leaves an 8-byte spill); r128parts = PARTS on
# top of R128.
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
m=fastmace.kernels.oeq_streaming.spline_forward
w=fastmace.kernels.oeq_streaming.wprime_backward
declare -A variant=([vec4]="$w:SPLINE_VECTOR=4" [cap5]="$m:R128=False $m:THREADS=128 $m:MIN_BLOCKS=5"
                    [nocap]="$m:R128=False $m:THREADS=128" [r128parts]="$m:PARTS=True")
names="vec4 cap5 nocap r128parts"
note "spline_followups_job on $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"

for name in vec4 cap5; do
  note "start check-$name"
  # shellcheck disable=SC2086
  python tools/spline_w_check.py --checkpoint "$ck" --structure "$st@10x10x10" --output "$out/check-$name.json" \
    --constant ${variant[$name]} > "$out/check-$name.log" 2>&1 < /dev/null
  code=$?
  note "end check-$name exit=$code $(grep -hoE '"(spline_in_use|forward_registers|forward_local_bytes)": [^]]+' "$out/check-$name.log" | head -n 3 | tr '\n' ' ' | cut -c1-200)"
  if [ "$code" != 0 ]; then note "stop: spline not in use"; exit 4; fi
done

for name in $names; do
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    note "start ab-$name-$order"
    # shellcheck disable=SC2086
    python tools/ab_rewrites.py --output "$out/ab-$name-$order" --checkpoint "$ck" --model omat0-medium \
      --structures "$st@10x10x10" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
      --warmup 200 --block 256 --blocks "${BLOCKS:-6}" --order "$order" --pattern "$pattern" \
      --b-constant ${variant[$name]} > "$out/ab-$name-$order.log" 2>&1 < /dev/null
    note "end ab-$name-$order exit=$? $(grep -h -E 'b/a' "$out/ab-$name-$order.log" | tail -n 1 | cut -c1-240)"
  done
done
note done
