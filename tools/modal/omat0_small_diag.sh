#!/usr/bin/env bash
# OMAT-0 small on the H200, diagnostic (8 November 2026): the packed job (omat0_small_job.sh) gave side-b steps of
# 2.2-3.1 ms against side a's 5.4 ms in order ab only, and 5.4 ms everywhere in order ba. Here one process per A/B:
#   same-ab, same-ba, same2-ab: a = b = medium's H200 settings (identical sides: is a build 5.4 or 2.2 ms at random?);
#   plain-ab, plain-ba: a = medium's settings, b = the defaults without the label-keyed pair;
#   breakdown: tools/step_breakdown.py with medium's settings (where the step goes).
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
labels="$f:DECISION_66_TF32X3_LABELS=('omat0-medium',None) $f:DECISION_66_GENERIC_WPRIME_LABELS=('omat0-medium',None)"
nolabels="$f:DECISION_66_TF32X3_LABELS=('omat0-medium',) $f:DECISION_66_GENERIC_WPRIME_LABELS=('omat0-medium',)"
note "omat0_small_diag on $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"
ab() {  # name order b-constants...
  local name=$1 order=$2; shift 2
  local pattern; pattern=$([ "$order" = ab ] && echo abab || echo abba)
  # shellcheck disable=SC2086
  python tools/ab_rewrites.py --output "$out/ab-$name" --checkpoint "$ck" --model "$ck" \
    --structures "$st@10x10x10" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
    --warmup 200 --block 256 --blocks 6 --order "$order" --pattern "$pattern" \
    --a-constant $labels --b-constant "$@" > "$out/ab-$name.log" 2>&1 < /dev/null
  note "end ab-$name exit=$? $(grep -h -E 'b/a' "$out/ab-$name.log" | tail -n 1)"
}
ab same-ab ab $labels
ab same-ba ba $labels
ab same2-ab ab $labels
ab plain-ab ab $nolabels
ab plain-ba ba $nolabels
note "start breakdown"
# shellcheck disable=SC2086
python tools/step_breakdown.py --output "$out/breakdown" --checkpoint "$ck" --model "$ck" --structure "$st@10x10x10" \
  --temperature-K 300 --timestep-fs 1 --skin 0.5 --equilibrate-steps 100 --steps 200 --constant $labels \
  > "$out/breakdown.log" 2>&1 < /dev/null
note "end breakdown exit=$?"
rm -f "$out"/breakdown/trace.json
note done
