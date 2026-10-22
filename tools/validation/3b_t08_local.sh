#!/usr/bin/env bash
# 3b-T08 on the local RTX 5060 Ti (WSL): tools/polar_stress_fd.py, POLAR-1-S, one local GPU reservation
# (<= 15 min) per job, in sequence. Usage: tools/validation/3b_t08_local.sh RUN_ID [JOB...]
# Results: results/RUN_ID/<job>.json and .log. docs/validation/3b-T08.md has the findings.
set -uo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
run=${1:?usage: 3b_t08_local.sh RUN_ID [JOB...]}
shift
out=$repo/results/$run
mkdir -p "$out"
export PYTHONPATH=$repo FASTMACE_MODELS=$HOME/models FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18
export FASTMACE_BASELINE=$HOME/fastmace-baseline/1a-T18 PYTHONDONTWRITEBYTECODE=1 FASTMACE_SOURCE_COMMIT=${FASTMACE_SOURCE_COMMIT:-$(git rev-parse HEAD)}
py=$HOME/environments/mace_cueq_oeq/bin/python
# A float64 reference evaluation of NbCl5 3x3x3 takes about 5 s, so each job stays under about 150 evaluations
# (12 per step size and configuration, one per scan point). ref-f64 and ref-f64-frozen (both configurations, six
# steps, 81-point scan) timed out after the ideal configuration; their gate halves are ref-f64-gate*.
F64_STEPS="--steps 1e-3 3e-4 1e-4 3e-5 1e-5"
F32_STEPS="--steps 3e-3 1e-3 3e-4 1e-4"
declare -A JOBS=(
  [ref-f64]="--path reference --dtype float64 --probe --scan"
  [ref-f64-frozen]="--path reference --dtype float64 --frozen-kset --scan"
  [ref-f64-gate]="--path reference --dtype float64 --probe --configs gate $F64_STEPS"
  [ref-f64-gate-frozen]="--path reference --dtype float64 --frozen-kset --configs gate $F64_STEPS"
  [ref-f64-gate-frozen-h1e-4]="--path reference --dtype float64 --frozen-kset --configs gate --steps 1e-4"
  [ref-f64-gate-scan]="--path reference --dtype float64 --probe --configs gate --steps 1e-4 --scan --scan-points 21 --scan-max 1e-4"
  [ref-f32]="--path reference --dtype float32 --steps 1e-3"
  [stock-f64]="--path stock --dtype float64"
  [cueq-f64]="--path cueq --dtype float64 --eager --steps 1e-3"
  [cueq-f32]="--path cueq --dtype float32 --eager --steps 1e-3"
  [hybrid-f32]="--path hybrid --dtype float32 --steps 1e-3"
  [hybrid-f32-frozen]="--path hybrid --dtype float32 --frozen-kset $F32_STEPS"
  [ref-f64-NaCl]="--path reference --dtype float64 --probe --structure NaCl"
  [ref-f64-NbCl5-222]="--path reference --dtype float64 --probe --repeat 2 2 2"
  [research-f64-gate]="--path research --dtype float64 --configs gate --steps 1e-3 3e-4 1e-4 3e-5"
  [research-f64-gate-rebase]="--path research --dtype float64 --research-rebase --configs gate --steps 1e-3 3e-4 1e-4 3e-5"
)
# 1a-T19's references ran in mace_cueq_latest; the research path keeps that.
declare -A JOB_PY=([research-f64-gate]=$HOME/environments/mace_cueq_latest/bin/python
                   [research-f64-gate-rebase]=$HOME/environments/mace_cueq_latest/bin/python)
# Plain cuEq runs eager: the planner refuses compiled plain cuEq for PolarMACE (zero forces).
# Precision (analytic stress of each path against the float64 reference; one step for a coarse check), then
# the energy step on a fine strain grid. research-f64-gate* are not needed: fastmace.reference reproduces
# 1a-T19's 4.364e-4 to four digits (ref-f64-gate).
ORDER=(ref-f64-gate-frozen-h1e-4 ref-f32 cueq-f32 hybrid-f32 cueq-f64 ref-f64-gate-scan)
[ $# -gt 0 ] && ORDER=("$@")
for job in "${ORDER[@]}"; do
  echo "$(date -u +%FT%TZ) start $job"
  # shellcheck disable=SC2086  (the job's options are words)
  python3 tools/ci/local_gpu.py run --wait-minutes 240 --purpose "3b-T08 polar_stress_fd $job (<15 min)" -- \
    timeout 870 "${JOB_PY[$job]:-$py}" tools/polar_stress_fd.py ${JOBS[$job]} --output "$out/$job.json" > "$out/$job.log" 2>&1 < /dev/null
  status=$?
  echo "$(date -u +%FT%TZ) end $job exit=$status $(grep -E 'Error|Traceback' "$out/$job.log" | tail -1 | cut -c1-200)"
done
echo "$(date -u +%FT%TZ) done"
