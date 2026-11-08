#!/usr/bin/env bash
# OMAT-0 small on the H200, after the ab_rewrites fix (8 November 2026). The packed and diagnostic jobs
# (omat0_small_job.sh, omat0_small_diag.sh) timed 5.4 ms in most builds and 2.2-3.1 ms in some: tools/ab_rewrites.py
# planned each side outside its constants, so the label-keyed pair (DECISION_66_TF32X3_LABELS,
# DECISION_66_GENERIC_WPRIME_LABELS, read in FixedPlanner.plan) took whatever the last timed step had left set.
# Now every side's constants cover its force field, plan and engine. Here, Benchmark_loup (SrTiO3 10x10x10, 300 K,
# 1 fs, skin 0.5):
#   0. tools/spline_w_check.py --model omat0-small with medium's settings: small's in-use record;
#   1. plain-ab, plain-ba: a = medium's settings (the pair forced on), b = without the pair, one process each;
#   2. the switch pack again, both orders: joined, nospline, nocompose, nocueq, notail;
#   3. tools/step_breakdown.py without the pair (where the 5.4 ms go), with plan options.
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$real" > "$out/gpu.txt"
if ! echo "$real" | grep -qE "${EXPECT_GPU:-H200}"; then note "WRONG-GPU $real"; exit 3; fi
ck=${CHECKPOINT:-$FASTMACE_MODELS/mace-omat-0-small.model}
st=results/loup-a100-20261103/SrTiO3.cif
f=fastmace.planner.fixed
s=fastmace.kernels.oeq_streaming.spline_forward
c=fastmace.kernels.triton.compose_skip
labels="$f:DECISION_66_TF32X3_LABELS=('omat0-medium',None) $f:DECISION_66_GENERIC_WPRIME_LABELS=('omat0-medium',None)"
nolabels="$f:DECISION_66_TF32X3_LABELS=('omat0-medium',) $f:DECISION_66_GENERIC_WPRIME_LABELS=('omat0-medium',)"
note "omat0_small_fix on $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader); source $(cat .source_commit 2>/dev/null); checkpoint $(sha256sum "$ck" | cut -c1-16)"

note "start check"
# shellcheck disable=SC2086
python tools/spline_w_check.py --model omat0-small --checkpoint "$ck" --structure "$st@10x10x10" \
  --output "$out/check.json" --constant $labels --rattle 0.05 --compare-constant $labels > "$out/check.log" 2>&1 < /dev/null
note "end check exit=$? $(grep -hoE '"(spline_in_use|compose_skip_in_use|joined_layout_in_use|max_abs|same_build_max_abs|max_force)": [^,]+' "$out/check.log" | tr '\n' ' ' | cut -c1-300)"

ab() {  # name order [--pack P] -- b-constants...
  local name=$1 order=$2; shift 2
  local pattern; pattern=$([ "$order" = ab ] && echo abab || echo abba)
  # shellcheck disable=SC2086
  python tools/ab_rewrites.py --output "$out/ab-$name" --checkpoint "$ck" --model "$ck" \
    --structures "$st@10x10x10" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
    --warmup 200 --block 256 --blocks 6 --order "$order" --pattern "$pattern" \
    --a-constant $labels "$@" > "$out/ab-$name.log" 2>&1 < /dev/null
  note "end ab-$name exit=$?"
  grep -h -E 'b/a' "$out/ab-$name.log" | sed "s/^/  $name /" >> "$log"
}
# shellcheck disable=SC2086
ab plain-ab ab --b-constant $nolabels
# shellcheck disable=SC2086
ab plain-ba ba --b-constant $nolabels
pack="joined|$c:JOINED_LAYOUT=True;nospline|$s:SPLINE_W=False;nocompose|$c:COMPOSE_SKIP=False"
pack="$pack;nocueq|fastmace.rewrites.cueq_projection:CUEQ_PROJECTED=False;notail|fastmace.rewrites.species:GENERIC_FOLD_READOUT_TAIL=False"
for order in ab ba; do
  # shellcheck disable=SC2086
  ab pack-$order $order --b-constant $labels --pack "$pack"
done

note "start breakdown-nolabels"
# shellcheck disable=SC2086
python tools/step_breakdown.py --output "$out/breakdown-nolabels" --checkpoint "$ck" --model "$ck" \
  --structure "$st@10x10x10" --temperature-K 300 --timestep-fs 1 --skin 0.5 --equilibrate-steps 100 --steps 200 \
  --constant $nolabels > "$out/breakdown-nolabels.log" 2>&1 < /dev/null
note "end breakdown-nolabels exit=$?"
rm -f "$out"/breakdown-*/trace.json
note done
