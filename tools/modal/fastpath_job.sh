#!/usr/bin/env bash
# Codex H200 report items 1, 4, 5 against the H200 defaults (8 November 2026): one container (resource framework:
# packed) with
#   0. the new GPU tests (species_mm_joined against the per-path products, eager and compiled, CUTLASS on the H200);
#   1. per variant tools/spline_w_check.py with the variant on (Benchmark_loup build, SrTiO3 10x10x10): forces on a
#      copy rattled by 0.05 A against the defaults, next to the same build's run-to-run spread;
#   2. the force/stress gate of the rounding-class variants (tail; cueq, whose einsum a compiled forward may lower
#      differently);
#   3. per variant the MD-step A/B a = H200 defaults, b = the variant, both build orders (skin 0.5, 1 fs), and all
#      four together.
# Variants: fast = COMPOSE_SKIP's cached weights and joined product (compose_skip.COMPOSE_FAST); split = linear_div's
# backward blocks in one kernel (linear_div.LINEAR_DIV_SPLIT); cueq = cuEq's projected weight kept
# (cueq_projection.CUEQ_PROJECTED); tail = the generic readout-tail fold (species.GENERIC_FOLD_READOUT_TAIL).
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
declare -A on=([fast]="fastmace.kernels.triton.compose_skip:COMPOSE_FAST=True"
               [split]="fastmace.kernels.cutlass.linear_div:LINEAR_DIV_SPLIT=True"
               [cueq]="fastmace.rewrites.cueq_projection:CUEQ_PROJECTED=True"
               [tail]="fastmace.rewrites.species:GENERIC_FOLD_READOUT_TAIL=True")
declare -A off=([fast]="fastmace.kernels.triton.compose_skip:COMPOSE_FAST=False"
                [split]="fastmace.kernels.cutlass.linear_div:LINEAR_DIV_SPLIT=False"
                [cueq]="fastmace.rewrites.cueq_projection:CUEQ_PROJECTED=False"
                [tail]="fastmace.rewrites.species:GENERIC_FOLD_READOUT_TAIL=False")
names="fast split cueq tail"
note "fastpath_job on $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"

note "start pytest-joined"
python -m pytest -q -p no:cacheprovider tests/gpu/test_species_mm_joined_cuda.py > "$out/pytest-joined.log" 2>&1 < /dev/null
code=$?
note "end pytest-joined exit=$code $(tail -n 1 "$out/pytest-joined.log" | cut -c1-200)"
failed=""
if [ "$code" != 0 ]; then failed="$failed fast"; fi

for name in $names; do
  note "start check-$name"
  python tools/spline_w_check.py --checkpoint "$ck" --structure "$st@10x10x10" --output "$out/check-$name.json" \
    --constant "${on[$name]}" --rattle 0.05 --compare-constant "${off[$name]}" \
    > "$out/check-$name.log" 2>&1 < /dev/null
  code=$?
  note "end check-$name exit=$code $(grep -hoE '"(compose_skip_in_use|max_abs|same_build_max_abs|max_force)": [^,]+' "$out/check-$name.log" | tr -d '\n ' | cut -c1-240)"
  if [ "$code" != 0 ]; then failed="$failed $name"; fi
done

for name in tail cueq; do
  note "start gate-$name"
  python tools/force_stress_gate.py --output "$out/gate-$name" --engine fastmace --model omat0-medium \
    --checkpoint "$ck" --structures-root "$FASTMACE_STRUCTURES" --structures "$st@3x3x3" NaCl NbCl5 \
    --production-constant "${on[$name]}" > "$out/gate-$name.log" 2>&1 < /dev/null
  code=$?
  note "end gate-$name exit=$code"
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

all=()
for name in $names; do
  case " $failed " in *" $name "*) note "skip ab-$name: test, check or gate failed"; continue ;; esac
  all+=("${on[$name]}")
  for order in ab ba; do ab "$name" "$order" "${on[$name]}"; done
done
if [ "${#all[@]}" -gt 1 ]; then
  for order in ab ba; do ab all "$order" "${all[@]}"; done
fi
note done
