#!/usr/bin/env bash
# 3b-T10: phase 3b's one Ada spot check (decision 39, conductor's choice 8) on the Modal L4 (sm_89): first the new 3b
# GPU test files (D3 list, kernel, integration, NPT + D3), then the GPU acceptance tier. Each pytest run goes through
# tools/acceptance/stoppable_pytest.sh under `timeout --signal=INT`, so a run cut by its budget keeps its junit and
# per-test lines. Budgets: D3_MIN (default 20) and ACCEPTANCE_MIN (default 80) minutes; keep their sum plus start-up
# under the job's --timeout-min.
# Usage (from WSL, Modal client venv; the guard in fastmace_modal.py):
#   modal run tools/modal/fastmace_modal.py::job --gpu L4 --timeout-min 110 --commit <sha> \
#     --command "bash tools/modal/3b_t10_ada.sh" --out ~/fastmace-acceptance/3b-T10-ada-<date>
set -uo pipefail
out=${OUT:?the job sets OUT}
export FASTMACE_SOURCE_COMMIT=$(cat .source_commit)
export FASTMACE_BASELINE=/data/baseline/1a-T18 FASTMACE_HYBRID_PYTHON=$(command -v python)
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "Modal $(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader); source $FASTMACE_SOURCE_COMMIT"
python -m pip list 2>/dev/null | grep -i -E '^(torch|torch-dftd|dftd3|triton|cuequivariance|openequivariance|mace-torch) ' >> "$log"
stoppable() {  # name minutes pytest-arguments...
  local name=$1 minutes=$2; shift 2
  export FASTMACE_RESULTS=$out/$name/results
  note "start $name (${minutes} min)"
  timeout --signal=INT --kill-after=180 "${minutes}m" bash tools/acceptance/stoppable_pytest.sh "$out/$name" python \
    -p no:cacheprovider --basetemp="/tmp/pytest-$name" -rfEs --durations=40 "$@" > "$out/$name.log" 2>&1 < /dev/null
  local code=$?
  note "end $name exit=$code"
  tail -n 3 "$out/$name.log"
}
stoppable d3 "${D3_MIN:-20}" tests/gpu/test_d3_list_gpu.py tests/gpu/test_d3_kernel_gpu.py \
  tests/gpu/test_d3_integration_gpu.py tests/gpu/test_npt_d3_gpu.py
stoppable acceptance "${ACCEPTANCE_MIN:-80}" tests/gpu -m acceptance
note done
