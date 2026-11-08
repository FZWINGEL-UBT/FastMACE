#!/usr/bin/env bash
# OMAT-0 small on the H200 with OMAT-0 medium's H200 settings (8 November 2026; owner: "run with the same settings as
# medium"), Benchmark_loup (SrTiO3 10x10x10, 300 K, 1 fs, skin 0.5). OMAT-0 small has no FastMACE label, so the two
# label-keyed medium defaults (3xTF32 GEMMs, generic w' backward) are forced on with the label tuples: side a everywhere.
# One container (resource framework: packed):
#   0. tools/spline_w_check.py with those settings: which paths are in use, forces against the same build;
#   1. tools/ab_rewrites.py --pack, a = medium's settings, b = a plus one change, both build orders:
#      joined (round 6's JOINED_LAYOUT on), nolabel (the label-keyed pair off again), nospline, nocompose, nocueq,
#      notail (each H200 switch group off).
# Environment: OUT (the job sets it), EXPECT_GPU (default H200), BLOCKS (default 6).
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
note "omat0_small_job on $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); source $(cat .source_commit 2>/dev/null); checkpoint $(sha256sum "$ck" | cut -c1-16)"

note "start check"
python tools/spline_w_check.py --checkpoint "$ck" --structure "$st@10x10x10" --output "$out/check.json" \
  --constant $labels --rattle 0.05 --compare-constant $labels > "$out/check.log" 2>&1 < /dev/null
note "end check exit=$? $(grep -hoE '"(spline_in_use|compose_skip_in_use|joined_layout_in_use|max_abs|same_build_max_abs|max_force)": [^,]+' "$out/check.log" | tr '\n' ' ' | cut -c1-300)"

pack="joined|$c:JOINED_LAYOUT=True;nolabel|$f:DECISION_66_TF32X3_LABELS=('omat0-medium',) $f:DECISION_66_GENERIC_WPRIME_LABELS=('omat0-medium',)"
pack="$pack;nospline|$s:SPLINE_W=False;nocompose|$c:COMPOSE_SKIP=False"
pack="$pack;nocueq|fastmace.rewrites.cueq_projection:CUEQ_PROJECTED=False;notail|fastmace.rewrites.species:GENERIC_FOLD_READOUT_TAIL=False"
for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  note "start pack-$order"
  # shellcheck disable=SC2086
  python tools/ab_rewrites.py --output "$out/ab-small-$order" --checkpoint "$ck" --model "$ck" \
    --structures "$st@10x10x10" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
    --warmup 200 --block 256 --blocks "${BLOCKS:-6}" --order "$order" --pattern "$pattern" \
    --a-constant $labels --b-constant $labels --pack "$pack" > "$out/ab-small-$order.log" 2>&1 < /dev/null
  note "end pack-$order exit=$?"
  grep -h -E 'b/a' "$out/ab-small-$order.log" | sed "s/^/  $order /" >> "$log"
done
note "done"
