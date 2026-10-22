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
# d3-references, d3-cost, npt-compare and protocol call the tools of 3b-T03/T04, T06, T05 and T07; their commands are
# filled in at the freeze, once those tasks are merged (until then they stop with exit 3).
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
pending() { echo "$job: $1's tool is merged at the freeze; fill in this job's command then" >&2; exit 3; }
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
  d3-cost) pending 3b-T06 ;;
  d3-references) pending "3b-T03 and 3b-T04" ;;
  npt-compare) pending 3b-T05 ;;
  protocol) pending 3b-T07 ;;
  preflight)  # the GPU tests whose covered code changed after they last ran (runbook, "stale expectations")
    [ $# -gt 0 ] || { echo "preflight takes the test node ids" >&2; exit 2; }
    pytest_job preflight -rA "$@"
    ;;
  targeted)  # the tests corrected at acceptance
    [ $# -gt 0 ] || { echo "targeted takes the test node ids" >&2; exit 2; }
    pytest_job targeted -rA "$@"
    ;;
  acceptance-tier) pytest_job acceptance-tier tests/gpu -m acceptance ;;
  slow-tier) pytest_job slow-tier tests/gpu -m "slow and not acceptance" ;;
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
