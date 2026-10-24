#!/usr/bin/env bash
# 5-T11: phase 5 acceptance jobs on the local RTX 5060 Ti (WSL), one job per call, from the pinned copy.
#
#   bash tools/acceptance/phase5_local.sh SRC JOB [ARGS...]   (baseline: tag fastmace-4, dae3848; FASTMACE_4_COMMIT
#   overrides it, FASTMACE_REPOSITORY names the checkout that holds the tag, default /mnt/d/Margraf/FastMACE)
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
# The release side is tools/ab_fastmace.py --release 4 and tools/trajectory_versions.py --against 4 (sides 4 and
# 5), as 4-T13's jobs against 3b; items 6 and 9 use tools/validation/adapt_runs.py and 5-T09's GPU test.
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
checkpoint=$HOME/models/mace-mh-1.model
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
  base=${FASTMACE_4_COMMIT:-$(git -C "${FASTMACE_REPOSITORY:-/mnt/d/Margraf/FastMACE}" rev-parse --short     "fastmace-4^{commit}" 2>/dev/null)}
  [ -n "$base" ] || { echo "the tag fastmace-4 is not in FASTMACE_REPOSITORY; set FASTMACE_4_COMMIT" >&2; exit 2; }
  [ -d "$versions/fastmace_4" ] || { echo "export the baseline first: python -m fastmace.bench.versions export" \
    "--repository /mnt/d/Margraf/FastMACE --ref $base --name fastmace_4 --destination $versions" >&2; exit 2; }
}
ab() {  # name blocks pattern sides...: tools/ab_fastmace.py --release 4, MH-1 hybrid Langevin NVT, 4-T13's protocol
  local name=$1 blocks=$2 pattern=$3; shift 3
  run "$name" "$hybrid" tools/ab_fastmace.py --release 4 --expected-commit "$base" --renamed "$versions"     --output "$root/$name" --model mh1-r2scan --checkpoint "$checkpoint" --structures NaCl NbCl5 --sides "$@"     --warmup 300 --block 300 --blocks "$blocks" --pattern "$pattern"
}
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
  regression)  # item 5, NVT: this checkout (5, D3 off) against fastmace-4, 3 same-process A/B processes (4-T13's)
    need_baseline
    ab regression-ab-4-first 12 abab 4 5
    ab regression-ab-5-first 12 abab 5 4
    ab regression-ab-abba 12 abba 4 5
    ;;
  trajectories-cpu)  # item 4: CPU float64, tiny MH-1-class model, NaCl 2x2x2, NVT and NPT, D3 off and on; --bitwise
    need_baseline
    tiny=$root/tiny-mh1-11-17.model
    [ -f "$tiny" ] || (cd "$src" && env CUDA_VISIBLE_DEVICES= PYTHONPATH="$src:$src/tests" "$hybrid" -c "
import sys, torch
from _support.tiny_models import FAMILIES, tiny_mace_model
torch.save(tiny_mace_model([11, 17], **FAMILIES['mh1']), sys.argv[1])" "$tiny")
    for ensemble in nvt npt; do
      for dispersion in none d3bj; do
        for side in 4 5; do
          run traj-cpu-$ensemble-$dispersion-$side env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 "$hybrid"             tools/trajectory_versions.py run --against 4 --side $side --expected-commit "$base" --renamed "$versions"             --device cpu --backend auto --ensemble $ensemble --dispersion $dispersion             --output "$root/traj-cpu-$ensemble-$dispersion-$side" --model "$tiny" --checkpoint "$tiny"             --structure NaCl@2x2x2 --steps 100 --every 10 --seeds 20261023 --positions
        done
        run traj-cpu-$ensemble-$dispersion-compare "$hybrid" tools/trajectory_versions.py compare           --release "$root/traj-cpu-$ensemble-$dispersion-4" --current "$root/traj-cpu-$ensemble-$dispersion-5" --bitwise
      done
    done
    ;;
  trajectories-reproducible)  # item 4: the reproducible plan, plain cuEq float64, MH-1 NaCl, NVT and NPT; --bitwise
    need_baseline
    for ensemble in nvt npt; do
      for side in 4 5; do
        run traj-repro-$ensemble-$side "$latest" tools/trajectory_versions.py run --against 4 --side $side           --expected-commit "$base" --renamed "$versions" --reproducible --backend cueq --ensemble $ensemble           --output "$root/traj-repro-$ensemble-$side" --checkpoint "$checkpoint" --structure NaCl --steps 500           --every 50 --seeds 20261023 --positions
      done
      run traj-repro-$ensemble-compare "$latest" tools/trajectory_versions.py compare         --release "$root/traj-repro-$ensemble-4" --current "$root/traj-repro-$ensemble-5" --bitwise
    done
    ;;
  restart-across)  # item 4: a baseline (format 4) restart at step 50 resumed at the freeze to 100, CPU float64, tiny
    # model; the release stages run a git archive of the baseline as the plain package (tools/restart_versions.py)
    need_baseline
    tiny=$root/tiny-mh1-11-17.model
    [ -f "$tiny" ] || { echo "run trajectories-cpu first (it writes the tiny model)" >&2; exit 2; }
    repository=${FASTMACE_REPOSITORY:-/mnt/d/Margraf/FastMACE}
    release=$HOME/fastmace-5-T11/release-$base
    [ -d "$release/fastmace" ] || { mkdir -p "$release" && git -C "$repository" archive "$base" | tar -x -C "$release"       && git -C "$repository" rev-parse "$base" > "$release/.source_commit"; }
    common=(--model "$tiny" --checkpoint "$tiny" --structure NaCl@2x2x2 --steps 100 --restart-step 50 --seed 20261023)
    for stage in write uninterrupted; do
      run restart-across-$stage env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONPATH= "$hybrid"         tools/restart_versions.py run --stage $stage --release-root "$release" --expected-commit "$base"         "${common[@]}" --output "$root/restart-across/$stage"
    done
    run restart-across-resume env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 "$hybrid" tools/restart_versions.py run       --stage resume --source "$root/restart-across/write" "${common[@]}" --output "$root/restart-across/resume"
    run restart-across-compare "$hybrid" tools/restart_versions.py compare "$root/restart-across/uninterrupted"       "$root/restart-across/resume"
    ;;
  adapt-storm)  # item 6, timed: 5-T07's forced re-capture storm, unswitched then switched, one process per side
    for side in ${STORM_SIDES:-unswitched switched}; do
      run adapt-storm-$side "$hybrid" tools/validation/adapt_runs.py storm --side $side         --steps "${STORM_STEPS:-2000}" --output "$root/adapt-storm"
    done
    ;;
  adapt-production)  # item 6: the profiles with the default adaptation, each started and resumed in two pieces
    # through 3-T14's run.sh (relaxation, half the steps; resume: the rest, and the quench where the profile has one),
    # then no tier switch in any of them (adapt_runs.py check)
    export FASTMACE_COMMIT=$FASTMACE_SOURCE_COMMIT
    nacl=$("$hybrid" tools/validation/adapt_runs.py nacl-config "$root/adapt-production/nacl-config")
    for entry in "npt_anneal_NbCl5_short:13500" "swap_anneal_NaNbOCl4_short:13500" "$nacl:135000"; do
      profile=${entry%:*}; half=${entry##*:}; name=$(basename "$profile" .toml)
      run adapt-production-$name-start bash results/3-T14-protocol/run.sh start "$root/adapt-production/$name"         "$profile" "$half"
      run adapt-production-$name-resume bash results/3-T14-protocol/run.sh resume "$root/adapt-production/$name" "$half"
    done
    p=$root/adapt-production
    run adapt-production-check "$hybrid" tools/validation/adapt_runs.py check "$p/npt_anneal_NbCl5_short"       "$p/swap_anneal_NaNbOCl4_short" "$p/nvt_anneal_NaCl" --out "$p/check.json"
    ;;
  calibration)  # item 9, timed as a check: 5-T09's NaCl smoke (MH-1, calibrate=True, hybrid against plain cuEq)
    pytest_job calibration "$latest" -rA tests/gpu/test_calibrate_gpu.py
    ;;
  adapt-tests)  # items 6 and 7: 5-T06's, 5-T07's and 5-T08's GPU tests (monitor, switch, fallbacks), untimed
    pytest_job adapt-tests "$latest" -rA tests/gpu/test_monitor_gpu.py tests/gpu/test_adapt_switch_gpu.py       tests/gpu/test_fallback_gpu.py tests/gpu/test_fallback_mid_run_gpu.py "$@"
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
