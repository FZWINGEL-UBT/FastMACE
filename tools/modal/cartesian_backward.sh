#!/usr/bin/env bash
# The two new w' backward candidates of plans/codex-ideas-20261106.md ("Second pass"; owner, 6 November 2026) on
# OMAT-0 medium, Benchmark_loup SrTiO3 10x10x10: tools/lab_cartesian.py (N1 Cartesian harmonic adjoints with their
# node converter, N2 FP64 matrix-adjoint skeleton bound) against the plan's default w' kernel, LABS fresh processes.
# Usage (Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu H200 --timeout-min 25 --commit <sha> \
#     --command "ALIAS='NVIDIA H200' bash tools/modal/cartesian_backward.sh" --out ~/fm-runs/cartesian-lab-h200
set -uo pipefail
out=${OUT:?the job sets OUT}
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
[ "$gpu" = "${ALIAS:?set ALIAS}" ] || { echo "WRONG-GPU: $gpu"; exit 3; }
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "$(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); source $(cat .source_commit)"
for t in $(seq 1 "${LABS:-2}"); do
  note "start lab-$t"
  python tools/lab_cartesian.py --checkpoint "$ck" --rounds "${ROUNDS:-21}" ${LAB_ARGS:-} --output "$out/lab-$t" \
    > "$out/lab-$t.log" 2>&1 < /dev/null
  note "end lab-$t exit=$?"
  grep -h -E "^wprime |^N1 " "$out/lab-$t.log" | cut -c1-240 >> "$log"
done
note done
