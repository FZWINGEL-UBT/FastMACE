#!/usr/bin/env bash
# 3b-T10: phase 3b acceptance jobs on the local RTX 5060 Ti (WSL), one job per call, from the pinned copy.
#
#   FASTMACE_3_COMMIT=<sha of tag fastmace-3> bash tools/acceptance/phase3b_local.sh SRC JOB [ARGS...]
#
# SRC is the pinned copy (git archive of the 3b-T10 head, with .source_commit); its fastmace/ equals the freeze's.
# Run every call under the GPU reservation: python3 SRC/tools/ci/local_gpu.py run --purpose "3b-T10 JOB" -- bash ...
# Outputs: ~/fastmace-acceptance/3b-T10-local-<date>/<job>/ and <job>.log beside it. Runbook, pass rules and job
# table: docs/validation/phase-3b-runbook.md.
# Jobs:
#   timed (quiet PC, first, alone): regression (item 5, NVT A/B), regression-npt (item 5, NPT, one side per process),
#     d3-cost (item 3)
#   untimed: preflight TEST... (the stale-expectation reruns), d3-references (item 1), npt-compare (item 2),
#     protocol (item 4), trajectories-cpu, trajectories-reproducible (item 5), acceptance-tier, slow-tier,
#     targeted TEST... (tests corrected at acceptance)
# pytest jobs run through stoppable_pytest.sh: results.jsonl per finished test, junit.xml also when stopped. Stop one
# with kill -INT "$(cat <root>/<job>/pytest.pid)", never by killing this script's shell or the reservation first.
# d3-references, d3-cost and npt-compare call the tools of 3b-T03/T04, T06 and T05 (filled in at the freeze, ff80ca2);
# protocol waits for the owner's D1 pressure answer (exit 3 until then); npt-ase is item 2's pre-declared extension.
set -uo pipefail
src=$(cd "${1:?SRC}" && pwd); job=${2:?JOB}; shift 2
run_id=3b-T10-local-${RUN_DATE:-$(date -u +%Y%m%d)}
root=$HOME/fastmace-acceptance/$run_id
cache=$HOME/fastmace-cache/3b-T10
envs=$HOME/environments
hybrid=$envs/mace_cueq_oeq/bin/python
latest=$envs/mace_cueq_latest/bin/python
checkpoint=$HOME/models/mace-mh-1.model
versions=$HOME/fastmace-3b-T10/versions
mkdir -p "$root" "$cache"/{tmp,fastmace,inductor,triton}
export PYTHONPATH=$src PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=4
export TMPDIR=$cache/tmp FASTMACE_CACHE=$cache/fastmace TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton
export FASTMACE_MODELS=$HOME/models FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18
export FASTMACE_BASELINE=$HOME/fastmace-baseline/1a-T18 FASTMACE_HYBRID_PYTHON=$hybrid
export FASTMACE_SOURCE_COMMIT=$(cat "$src/.source_commit" 2>/dev/null || echo unknown)
log=$root/jobs.log
note() { echo "$(date -u +%FT%TZ) $*" | tee -a "$log"; }
run() {  # name command...
  local name=$1; shift
  note "start $name gpu=$(nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader) source=$FASTMACE_SOURCE_COMMIT"
  (cd "$src" && "$@") > "$root/$name.log" 2>&1 < /dev/null
  local status=$?
  note "end $name exit=$status"
  return $status
}
pytest_job() {  # name pytest-arguments...
  local name=$1; shift
  export FASTMACE_RESULTS=$root/$name/results  # the tests' result records (tests/gpu/test_engine_gpu.py: report)
  run "$name" bash "$src/tools/acceptance/stoppable_pytest.sh" "$root/$name" "$latest" -p no:cacheprovider \
    --basetemp="$cache/pytest-$name" -rfEs --durations=60 "$@"
}
need_baseline() {  # the fastmace-3 commit, for the release side: $base
  [ -n "${FASTMACE_3_COMMIT:-}" ] || { echo "set FASTMACE_3_COMMIT (tag fastmace-3, or the commit the conductor names)" >&2; exit 2; }
  base=$FASTMACE_3_COMMIT
}
# The preflight's files (their slow and acceptance tests run first); the tiers then leave them out, so nothing runs twice.
PREFLIGHT_FILES=(tests/gpu/test_ase_equivalence_gpu.py tests/gpu/test_engine_gpu.py tests/gpu/test_events_gpu.py
  tests/gpu/test_fused_integrator_gpu.py tests/gpu/test_integrators_gpu.py tests/gpu/test_overflow_commit_gpu.py
  tests/gpu/test_registry_gpu.py tests/gpu/test_selfcheck_gpu.py tests/gpu/test_planner_gpu.py
  tests/gpu/test_gate_resources_gpu.py tests/gpu/test_canonical_restart_gpu.py tests/gpu/test_reproducible_gpu.py
  tests/gpu/test_reference_tier_gpu.py)
IGNORE_PREFLIGHT=()
for f in "${PREFLIGHT_FILES[@]}"; do IGNORE_PREFLIGHT+=(--ignore="$f"); done
ab() {  # name blocks pattern sides...
  local name=$1 blocks=$2 pattern=$3; shift 3
  run "$name" "$hybrid" tools/ab_fastmace.py --release 3 --expected-commit "$base" --renamed "$versions" \
    --output "$root/$name" --model mh1-r2scan --checkpoint "$checkpoint" --structures NaCl NbCl5 --sides "$@" \
    --warmup 300 --block 300 --blocks "$blocks" --pattern "$pattern" --dump-forces
}
case "$job" in
  regression)  # item 5, NVT: this checkout (3b, D3 off) against fastmace-3, 3 same-process A/B processes, no controls
    need_baseline
    ab regression-ab-3-first 12 abab 3 3b
    ab regression-ab-3b-first 12 abab 3b 3
    ab regression-ab-abba 12 abba 3 3b
    ;;
  regression-npt)  # item 5, NPT, NbCl5: one version per process (two-sided NPT is refused), alternating 3 3b 3b 3
    need_baseline
    i=0
    for side in 3 3b 3b 3; do
      i=$((i + 1))
      run regression-npt-$i-$side "$hybrid" tools/ab_fastmace.py --release 3 --expected-commit "$base" \
        --renamed "$versions" --output "$root/regression-npt-$i-$side" --model mh1-r2scan --checkpoint "$checkpoint" \
        --structures NbCl5 --sides "$side" --ensemble npt --warmup 300 --block 300 --blocks 12 --pattern abab
    done
    ;;
  d3-cost)  # item 3: 3b-T06's A/B, D3 on against off in one process, ABBA blocks of 1000 steps, builds in the clock.
    # NVT: the dense phase (heat 0 -> 1000 K in 5 ps, then 8 ps at 1000 K), 3b-T06's route; NPT: the anneal route at
    # D3_COST_PRESSURE_GPA (default 0, the runbook's; the owner's D1 answer may add 0.1).
    run d3-cost-nvt "$hybrid" tools/bench/d3_cost.py ab --ensemble nvt --route "heat:0:1000:5000,peak:1000:976:8000" \
      --output "$root/d3-cost-nvt" --checkpoint "$checkpoint"
    p=${D3_COST_PRESSURE_GPA:-0.0}
    run d3-cost-npt-${p}GPa "$hybrid" tools/bench/d3_cost.py ab --ensemble npt --pressure-GPa "$p" \
      --output "$root/d3-cost-npt-${p}GPa" --checkpoint "$checkpoint"
    ;;
  d3-references)  # item 1: the D3 GPU test files (list, kernel, integration, NPT), the CPU reference tests, and the table
    pytest_job d3-references -rA tests/gpu/test_d3_list_gpu.py tests/gpu/test_d3_kernel_gpu.py \
      tests/gpu/test_d3_integration_gpu.py tests/gpu/test_npt_d3_gpu.py tests/cpu/test_d3_references.py \
      tests/cpu/test_d3_kernel.py
    run d3-reference-table env PYTHONPATH="$src:$src/tests" "$latest" tools/validation/d3_reference_table.py \
      --output "$root/d3-reference-table" --devices cpu cuda
    ;;
  d3-table)  # item 1's numbers alone (d3_reference_table.py), e.g. after a fix of the tool
    run d3-reference-table env PYTHONPATH="$src:$src/tests" "$latest" tools/validation/d3_reference_table.py       --output "$root/d3-reference-table" --devices cpu cuda
    ;;
  npt-compare)  # item 2: the FastMACE side at the freeze (3b-T05's tool and settings), seeds 1-3 (EXTRA_SEEDS: the
    # pre-declared extension, e.g. "4 5 6"); the ASE side is 3b-T05's (cited, unchanged), analysed together
    for temperature in 300 600; do
      for seed in ${SEEDS:-1 2 3}; do
        run npt-fastmace-${temperature}K-s$seed "$hybrid" tools/validation/npt_d3_comparison.py run-fastmace \
          --temperature-K $temperature --seed $seed --output "$root/npt-compare/fastmace-${temperature}K-s$seed" \
          --checkpoint "$checkpoint" --machine "RTX 5060 Ti (local)"
      done
    done
    ;;
  npt-ase)  # item 2's extension only: ASE runs on this GPU, SEEDS required ("104 105 106"), TEMPERATURES (default 300 600)
    [ -n "${SEEDS:-}" ] || { echo "npt-ase needs SEEDS" >&2; exit 2; }
    for temperature in ${TEMPERATURES:-300 600}; do
      for seed in $SEEDS; do
        run npt-ase-${temperature}K-s$seed "$latest" tools/validation/npt_d3_comparison.py run-ase \
          --temperature-K $temperature --seed $seed --output "$root/npt-compare/ase-${temperature}K-s$seed" \
          --checkpoint "$checkpoint" --machine "RTX 5060 Ti (local)"
      done
    done
    ;;
  protocol)  # item 4: waits for the owner's D1 pressure answer (0 or 0.1 GPa); filled in then
    echo "protocol: waiting for the owner's D1 pressure answer" >&2; exit 3 ;;
  preflight)  # the GPU tests whose covered code changed after they last ran (runbook, "stale expectations"):
    # the slow and acceptance tests of the files that cover BAOAB, the planner, restart and the engine graph
    if [ $# -gt 0 ]; then pytest_job preflight -rA -m "slow or acceptance" "$@"
    else pytest_job preflight -rA -m "slow or acceptance" "${PREFLIGHT_FILES[@]}"; fi
    ;;
  targeted)  # the tests corrected at acceptance
    [ $# -gt 0 ] || { echo "targeted takes the test node ids" >&2; exit 2; }
    pytest_job targeted -rA "$@"
    ;;
  acceptance-tier) pytest_job acceptance-tier tests/gpu -m acceptance "${IGNORE_PREFLIGHT[@]}" ;;
  slow-tier) pytest_job slow-tier tests/gpu -m "slow and not acceptance" "${IGNORE_PREFLIGHT[@]}" ;;
  trajectories-cpu)  # item 5: CPU float64, tiny MH-1-class model, NaCl 2x2x2, NVT and NPT; compare --bitwise (no GPU)
    need_baseline
    tiny=$root/tiny-mh1-11-17.model
    [ -f "$tiny" ] || (cd "$src" && env CUDA_VISIBLE_DEVICES= PYTHONPATH="$src:$src/tests" "$hybrid" -c "
import sys, torch
from _support.tiny_models import FAMILIES, tiny_mace_model
torch.save(tiny_mace_model([11, 17], **FAMILIES['mh1']), sys.argv[1])" "$tiny")
    for ensemble in nvt npt; do
      for side in 3 3b; do
        run traj-cpu-$ensemble-$side env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 "$hybrid" tools/trajectory_versions.py \
          run --against 3 --side $side --expected-commit "$base" --renamed "$versions" --device cpu \
          --backend auto --ensemble $ensemble --output "$root/traj-cpu-$ensemble-$side" --model "$tiny" \
          --checkpoint "$tiny" --structure NaCl@2x2x2 --steps 100 --every 10 --seeds 20261022 --positions
      done
      run traj-cpu-$ensemble-compare "$hybrid" tools/trajectory_versions.py compare \
        --release "$root/traj-cpu-$ensemble-3" --current "$root/traj-cpu-$ensemble-3b" --bitwise
    done
    ;;
  trajectories-reproducible)  # item 5: the reproducible plan, plain cuEq float64, MH-1 NaCl, NVT and NPT; --bitwise
    need_baseline
    for ensemble in nvt npt; do
      for side in 3 3b; do
        run traj-repro-$ensemble-$side "$latest" tools/trajectory_versions.py run --against 3 --side $side \
          --expected-commit "$base" --renamed "$versions" --reproducible --backend cueq --ensemble $ensemble \
          --output "$root/traj-repro-$ensemble-$side" --checkpoint "$checkpoint" --structure NaCl --steps 500 \
          --every 50 --seeds 20261022 --positions
      done
      run traj-repro-$ensemble-compare "$latest" tools/trajectory_versions.py compare \
        --release "$root/traj-repro-$ensemble-3" --current "$root/traj-repro-$ensemble-3b" --bitwise
    done
    ;;
  *) echo "unknown job $job" >&2; exit 2 ;;
esac
