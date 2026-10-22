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
export PYTHONDONTWRITEBYTECODE=1 FASTMACE_SOURCE_COMMIT=${FASTMACE_SOURCE_COMMIT:-$(git rev-parse HEAD)}
py=$HOME/environments/mace_cueq_oeq/bin/python
declare -A JOBS=(
  [ref-f64]="--path reference --dtype float64 --probe --scan"
  [ref-f64-frozen]="--path reference --dtype float64 --frozen-kset --scan"
  [ref-f32]="--path reference --dtype float32 --probe"
  [ref-f32-frozen]="--path reference --dtype float32 --frozen-kset"
  [stock-f64]="--path stock --dtype float64"
  [cueq-f64]="--path cueq --dtype float64"
  [cueq-f32]="--path cueq --dtype float32"
  [hybrid-f32]="--path hybrid --dtype float32"
  [hybrid-f32-frozen]="--path hybrid --dtype float32 --frozen-kset"
  [cueq-f64-frozen]="--path cueq --dtype float64 --frozen-kset"
  [ref-f64-NaCl]="--path reference --dtype float64 --probe --structure NaCl"
  [ref-f64-NbCl5-222]="--path reference --dtype float64 --probe --repeat 2 2 2"
)
ORDER=(ref-f64 ref-f64-frozen stock-f64 ref-f32 ref-f32-frozen cueq-f64 cueq-f64-frozen cueq-f32 hybrid-f32
       hybrid-f32-frozen ref-f64-NaCl ref-f64-NbCl5-222)
[ $# -gt 0 ] && ORDER=("$@")
for job in "${ORDER[@]}"; do
  echo "$(date -u +%FT%TZ) start $job"
  # shellcheck disable=SC2086  (the job's options are words)
  python3 tools/ci/local_gpu.py run --wait-minutes 240 --purpose "3b-T08 polar_stress_fd $job (<15 min)" -- \
    timeout 870 "$py" tools/polar_stress_fd.py ${JOBS[$job]} --output "$out/$job.json" > "$out/$job.log" 2>&1 < /dev/null
  status=$?
  echo "$(date -u +%FT%TZ) end $job exit=$status $(grep -E 'Error|Traceback' "$out/$job.log" | tail -1 | cut -c1-200)"
done
echo "$(date -u +%FT%TZ) done"
