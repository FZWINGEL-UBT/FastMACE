#!/usr/bin/env bash
# 5-T11: phase 5 acceptance jobs on the local RTX 5060 Ti (WSL), one job per call, from the pinned copy.
#
#   FASTMACE_4_COMMIT=<sha of tag fastmace-4, else dae3848> bash tools/acceptance/phase5_local.sh SRC JOB [ARGS...]
#
# SRC is the pinned copy (git archive of the 5-T11 head, with .source_commit); its fastmace/ equals the freeze's.
# GPU jobs run under the reservation: python3 SRC/tools/ci/local_gpu.py run --wait-minutes 600 --purpose "5-T11 JOB"
# -- bash ...; CPU jobs (conformance, cpu-acceptance, corpus, trajectories-cpu, restart-across) run without it, never
# during a timed stretch. Outputs: ~/fastmace-acceptance/5-T11-local-<date>/<job>/ and <job>.log beside it. Runbook,
# pass rules and job table: docs/validation/phase-5-runbook.md.
# Jobs:
#   CPU: conformance (item 1), cpu-acceptance (item 2), corpus (item 3), trajectories-cpu, restart-across (item 4)
#   timed (quiet PC, alone): regression (item 5), adapt-storm (item 6), calibration (item 9)
#   untimed GPU: preflight TEST..., trajectories-reproducible (item 4), adapt-production, adapt-tests (items 6, 7),
#     compat, acceptance-tier, slow-tier, targeted TEST...
# The release side (--release 4 of ab_fastmace.py, trajectory_versions.py and restart_versions.py) and the jobs
# that call 5-T07's and 5-T09's tools are filled in at the freeze, after 4-T13's tools merge (exit 3 until then).
set -uo pipefail
src=$(cd "${1:?SRC}" && pwd); job=${2:?JOB}; shift 2
run_id=5-T11-local-${RUN_DATE:-$(date -u +%Y%m%d)}
root=$HOME/fastmace-acceptance/$run_id
cache=$HOME/fastmace-cache/5-T11
envs=$HOME/environments
cpu=$envs/fastmace_cpu/bin/python
hybrid=$envs/mace_cueq_oeq/bin/python
latest=$envs/mace_cueq_latest/bin/python
versions=$HOME/fastmace-5-T11/versions
mkdir -p "$root" "$cache"/{tmp,fastmace,inductor,triton}
export PYTHONPATH=$src PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=4
export TMPDIR=$cache/tmp FASTMACE_CACHE=$cache/fastmace TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton
export FASTMACE_MODELS=$HOME/models FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18
export FASTMACE_BASELINE=$HOME/fastmace-baseline/1a-T18 FASTMACE_HYBRID_PYTHON=$hybrid
export FASTMACE_SOURCE_COMMIT=$(cat "$src/.source_commit" 2>/dev/null || echo unknown)
log=$root/jobs.log
note() { echo "$(date -u +%FT%TZ) $*" | tee -a "$log"; }
gpu_state() { nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo none; }
run() {  # name command...
  local name=$1; shift
  note "start $name gpu=$(gpu_state) load=$(cut -d' ' -f1-3 /proc/loadavg) source=$FASTMACE_SOURCE_COMMIT"
  (cd "$src" && "$@") > "$root/$name.log" 2>&1 < /dev/null
  local status=$?
  note "end $name exit=$status"
  return $status
}
pytest_job() {  # name python pytest-arguments...
  local name=$1 python=$2; shift 2
  export FASTMACE_RESULTS=$root/$name/results  # the tests' result records (tests/gpu/test_engine_gpu.py: report)
  run "$name" bash "$src/tools/acceptance/stoppable_pytest.sh" "$root/$name" "$python" -p no:cacheprovider \
    --basetemp="$cache/pytest-$name" -rfEs --durations=60 "$@"
}
cpu_pytest() {  # name pytest-arguments...: CPU only, at most two workers (AGENTS.md: other sessions share WSL)
  local name=$1; shift
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 pytest_job "$name" "$cpu" -n "${CPU_WORKERS:-2}" "$@"
}
need_baseline() {  # the fastmace-4 commit and its renamed export: $base
  [ -n "${FASTMACE_4_COMMIT:-}" ] || { echo "set FASTMACE_4_COMMIT (tag fastmace-4, else dae3848)" >&2; exit 2; }
  base=$FASTMACE_4_COMMIT
  [ -d "$versions/fastmace_4" ] || { echo "export the baseline first: python -m fastmace.bench.versions export" \
    "--repository /mnt/d/Margraf/FastMACE --ref $base --name fastmace_4 --destination $versions" >&2; exit 2; }
}
not_yet() { echo "$job: filled in at the freeze ($*)" >&2; exit 3; }
case "$job" in
  conformance)  # item 1: every rule-marked test and the meta-test, its listing kept
    cpu_pytest conformance -rA -m rule tests/cpu
    cpu_pytest conformance-meta -rA tests/cpu/test_rules_meta.py
    ;;
  cpu-acceptance)  # item 2: the CPU acceptance tests (the full CPU suite is the conductor's trial, cited)
    cpu_pytest cpu-acceptance -m acceptance tests/cpu
    ;;
  corpus)  # item 3: the acceptance set planned by the baseline and by this copy, compared (rule 12 explains)
    need_baseline
    mkdir -p "$root/corpus"
    run corpus-old env CUDA_VISIBLE_DEVICES='' PYTHONPATH="$src:$versions" "$cpu" tools/plan_corpus.py dump \
      --package fastmace_4 --out "$root/corpus/old.json"
    run corpus-new env CUDA_VISIBLE_DEVICES='' "$cpu" tools/plan_corpus.py dump --package fastmace \
      --out "$root/corpus/new.json"
    run corpus-compare env CUDA_VISIBLE_DEVICES='' "$cpu" tools/plan_corpus.py compare "$root/corpus/old.json" \
      "$root/corpus/new.json" --explain-rules 12 --out "$root/corpus/report.json"
    ;;
  trajectories-cpu|trajectories-reproducible|restart-across|regression)  # items 4 and 5 against fastmace-4
    need_baseline
    not_yet "--release 4 / --against 4 of tools/ab_fastmace.py, trajectory_versions.py, restart_versions.py, after 4-T13"
    ;;
  adapt-storm|adapt-production|calibration)  # items 6 and 9: 5-T07's storm tool, the profiles, 5-T09's smoke
    not_yet "5-T07's and 5-T09's tools"
    ;;
  adapt-tests)  # items 6 and 7: 5-T07's and 5-T08's GPU tests, by file (named at the freeze)
    [ $# -gt 0 ] || not_yet "the GPU test files of 5-T07 and 5-T08"
    pytest_job adapt-tests "$latest" -rA "$@"
    ;;
  preflight)  # the GPU tests whose covered code changed after they last ran (runbook, "stale expectations")
    [ $# -gt 0 ] || { echo "preflight needs the selected tests (runbook: the stale-expectation listing)" >&2; exit 2; }
    pytest_job preflight "$latest" -rA "$@"
    ;;
  targeted)  # tests corrected at acceptance, rerun on their own
    pytest_job "targeted-$(date -u +%H%M)" "$latest" -rA "$@"
    ;;
  compat)  # the compatibility tier on Blackwell (tools/acceptance/compat.py records the GPU and provider)
    run compat "$latest" tools/acceptance/compat.py --provider local --label "RTX 5060 Ti" --output "$root/compat"
    ;;
  acceptance-tier)
    pytest_job acceptance-tier "$latest" -m acceptance tests/gpu
    ;;
  slow-tier)
    pytest_job slow-tier "$latest" -m "slow and not acceptance" tests/gpu
    ;;
  *)
    echo "unknown job $job (see the header)" >&2; exit 2
    ;;
esac
