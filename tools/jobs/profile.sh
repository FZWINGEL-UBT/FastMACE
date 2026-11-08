#!/usr/bin/env bash
# profile v1: one torch.profiler breakdown (tools/step_breakdown.py, main pass) in one Modal container (resource
# framework §3.4). With PACK set, side b's switches (MODULE:NAME=VALUE ...) are applied as production constants;
# the launcher puts one manifest row into one profile job.
#
# Environment (tools/jobs/launch.py sets it):
#   OUT          results directory (fastmace_modal.py's job sets it)
#   PACK         optional "name|MODULE:NAME=VALUE ..." (one entry; empty or "default|" profiles the default plan)
#   EXPECT_GPU   grep -E pattern the GPU name must match; else the job exits 3 (WRONG-GPU)
#   MODEL        model label (default omat0-medium), CHECKPOINT (default $FASTMACE_MODELS/mace-omat-0-medium.model)
#   STRUCTURE    structure (default results/loup-a100-20261103/SrTiO3.cif@10x10x10)
#   BACKEND      hybrid|cueq (default hybrid), STEPS (default 1000)
# Results: $OUT/run.log, $OUT/profile-<name>/ (tables; the trace is dropped) and .log.
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$real" > "$out/gpu.txt"
if ! echo "$real" | grep -qE "${EXPECT_GPU:?set EXPECT_GPU}"; then echo "WRONG-GPU: $real"; note "WRONG-GPU $real"; exit 3; fi
ck=${CHECKPOINT:-$FASTMACE_MODELS/mace-omat-0-medium.model}
entry=${PACK:-default|}
entry=${entry%%;*}
name=${entry%%|*}; name=${name:-default}
read -r -a consts <<< "${entry#*|}"
extra=()
[ "${#consts[@]}" = 0 ] || extra=(--constant "${consts[@]}")
note "profile v1 on $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"
note "start profile-$name"
python tools/step_breakdown.py --output "$out/profile-$name" --checkpoint "$ck" --model "${MODEL:-omat0-medium}" \
  --backend "${BACKEND:-hybrid}" --timestep-fs 1 --steps "${STEPS:-1000}" \
  --structure "${STRUCTURE:-results/loup-a100-20261103/SrTiO3.cif@10x10x10}" "${extra[@]}" \
  > "$out/profile-$name.log" 2>&1 < /dev/null
code=$?
rm -f "$out/profile-$name/trace.json"  # large; the tables keep what the reports use
note "end profile-$name exit=$code"
note done
exit "$code"
