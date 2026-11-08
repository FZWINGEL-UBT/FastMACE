#!/usr/bin/env bash
# ab_pack v1: one Modal container, one baseline, N switches (aim for 4-8), both build orders (resource framework
# §3.4; AGENTS.md "Using resources"). The model load, compile and container start are paid once for the whole pack.
# Side a of every A/B is the GPU's default plan (the baseline); side b is the default plan with one switch.
#
# Environment (tools/jobs/launch.py sets it):
#   OUT          results directory (fastmace_modal.py's job sets it)
#   PACK         "name|MODULE:NAME=VALUE ...;name|..."  one entry per switch; a body starting with "--" goes to
#                tools/ab_rewrites.py as it is (e.g. "--b radial_tables=off")
#   EXPECT_GPU   grep -E pattern the GPU name must match (e.g. "H200", "A100"); else the job exits 3 (WRONG-GPU)
#   ALIAS        optional: run under another GPU's plan (tools/bench/gpu_alias); default: the GPU's own plan
#   MODEL        model label (default omat0-medium), CHECKPOINT (default $FASTMACE_MODELS/mace-omat-0-medium.model)
#   STRUCTURE    structure file (default results/loup-a100-20261103/SrTiO3.cif), CELL repeat (default 10x10x10)
#   BLOCKS       timed blocks of 256 steps per side (default 6)
# Results: $OUT/run.log (one "b/a" line per switch and order), $OUT/ab-<name>-<order>/ and .log.
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$real" > "$out/gpu.txt"
if ! echo "$real" | grep -qE "${EXPECT_GPU:?set EXPECT_GPU}"; then echo "WRONG-GPU: $real"; note "WRONG-GPU $real"; exit 3; fi
if [ -n "${ALIAS:-}" ]; then
  export PYTHONPATH=$PWD/tools/bench/gpu_alias:${PYTHONPATH:-} FASTMACE_GPU_ALIAS="$real=$ALIAS"
fi
ck=${CHECKPOINT:-$FASTMACE_MODELS/mace-omat-0-medium.model}
st=${STRUCTURE:-results/loup-a100-20261103/SrTiO3.cif}
note "ab_pack v1 on $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); alias ${FASTMACE_GPU_ALIAS:-none}; source $(cat .source_commit 2>/dev/null)"
IFS=';' read -r -a entries <<< "${PACK:?set PACK}"
failed=0
for entry in "${entries[@]}"; do
  [ -n "$entry" ] || continue
  name=${entry%%|*}
  read -r -a bc <<< "${entry#*|}"
  [ "${bc[0]:0:2}" = "--" ] || bc=(--b-constant "${bc[@]}")
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    note "start ab-$name-$order"
    python tools/ab_rewrites.py --output "$out/ab-$name-$order" --checkpoint "$ck" --model "${MODEL:-omat0-medium}" \
      --structures "$st@${CELL:-10x10x10}" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
      --warmup 200 --block 256 --blocks "${BLOCKS:-6}" --order "$order" --pattern "$pattern" "${bc[@]}" \
      > "$out/ab-$name-$order.log" 2>&1 < /dev/null
    code=$?
    [ "$code" = 0 ] || failed=$((failed + 1))
    note "end ab-$name-$order exit=$code $(grep -h -E 'b/a' "$out/ab-$name-$order.log" | tail -n 1 | cut -c1-240)"
  done
done
note "done failed=$failed"
[ "$failed" = 0 ]
