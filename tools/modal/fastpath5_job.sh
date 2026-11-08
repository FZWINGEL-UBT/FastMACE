#!/usr/bin/env bash
# Round 5 (8 November 2026) against the H200 defaults on main 65923679 (rounds 1-3 adopted, round 4 closed): per
# variant tools/spline_w_check.py (forces against the defaults next to the same build's spread) and the MD-step A/B
# a = H200 defaults, b = the variant, both build orders. Variants: t3264 / t6432 = the CUTLASS 3xTF32 GEMMs' output
# tile (cutlass.tf32x3.TILE, two warps, twice the blocks; bitwise: the same K loop). Measured on the A100 only
# (1.4-1.7 % slower there, 108 SMs); the H200 has 132 SMs and 316 blocks of 64x64 per product.
# Environment: OUT (the job sets it), EXPECT_GPU (default H200), BLOCKS (default 6), BREAKDOWN (default 0).
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$real" > "$out/gpu.txt"
if ! echo "$real" | grep -qE "${EXPECT_GPU:-H200}"; then note "WRONG-GPU $real"; exit 3; fi
ck=${CHECKPOINT:-$FASTMACE_MODELS/mace-omat-0-medium.model}
st=results/loup-a100-20261103/SrTiO3.cif
t=fastmace.kernels.cutlass.tf32x3
declare -A on=([t3264]="$t:TILE='32x64'" [t6432]="$t:TILE='64x32'")
declare -A off=([t3264]="$t:TILE='64x64'" [t6432]="$t:TILE='64x64'")
names="t3264 t6432"
note "fastpath5_job on $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"

failed=""
for name in $names; do
  note "start check-$name"
  python tools/spline_w_check.py --checkpoint "$ck" --structure "$st@10x10x10" --output "$out/check-$name.json" \
    --constant ${on[$name]} --rattle 0.05 --sources "$out/sources-$name" --compare-constant ${off[$name]} \
    > "$out/check-$name.log" 2>&1 < /dev/null
  code=$?
  note "end check-$name exit=$code $(grep -hoE '"(compose_skip_in_use|forward_stage_edges|forward_min_blocks|wprime_stage_sh|forward_registers|forward_local_bytes|registers|local_bytes|max_abs|same_build_max_abs|energy_diff|same_build_energy_diff)": [^,]+' "$out/check-$name.log" | tr -d '\n ' | cut -c1-400)"
  if [ "$code" != 0 ]; then failed="$failed $name"; fi
done


ab() {  # name order constants...
  local name=$1 order=$2; shift 2
  local pattern; pattern=$([ "$order" = ab ] && echo abab || echo abba)
  note "start ab-$name-$order"
  python tools/ab_rewrites.py --output "$out/ab-$name-$order" --checkpoint "$ck" --model omat0-medium \
    --structures "$st@10x10x10" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
    --warmup 200 --block 256 --blocks "${BLOCKS:-6}" --order "$order" --pattern "$pattern" \
    --b-constant "$@" > "$out/ab-$name-$order.log" 2>&1 < /dev/null
  note "end ab-$name-$order exit=$? $(grep -h -E 'b/a' "$out/ab-$name-$order.log" | tail -n 1 | cut -c1-240)"
}

for name in $names; do
  case " $failed " in *" $name "*) note "skip ab-$name: check or gate failed"; continue ;; esac
  for order in ab ba; do ab "$name" "$order" ${on[$name]}; done
done

if [ "${BREAKDOWN:-0}" != 0 ]; then
  note "start breakdown"
  python tools/step_breakdown.py --output "$out/breakdown" --checkpoint "$ck" --model omat0-medium \
    --structure "$st@10x10x10" --temperature-K 300 --timestep-fs 1 --skin 0.5 --equilibrate-steps 100 --steps 200 \
    > "$out/breakdown.log" 2>&1 < /dev/null
  note "end breakdown exit=$?"
  rm -f "$out"/breakdown/trace.json
fi
note done
