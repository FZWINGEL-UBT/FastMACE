#!/usr/bin/env bash
# 5-T11: phase 5 acceptance jobs on the local RTX 5060 Ti (WSL), one job per call, from the pinned copy.
#
#   FASTMACE_4_COMMIT=<sha of the baseline> bash tools/acceptance/phase4_local.sh SRC JOB [ARGS...]
#
# SRC is the pinned copy (git archive of the 5-T11 head, with .source_commit); its fastmace/ equals the freeze's.
# Run every call under the GPU reservation:
#   python3 SRC/tools/ci/local_gpu.py run --wait-minutes 600 --purpose "5-T11 JOB" -- bash SRC/tools/acceptance/...
# Outputs: ~/fastmace-acceptance/5-T11-local-<date>/<job>/ and <job>.log beside it. Runbook, pass rules and job
# table: docs/validation/phase-5-runbook.md.
# Jobs:
#   timed (quiet PC, first, alone): regression (item 5, NVT A/B), regression-npt (item 5, NPT, one side per process),
#     ase-speed (recorded: stock ASE against fastmace.ase, and each new integrator's step time)
#   untimed: preflight [TEST...], optimisers [PART...] (item 1), minima (item 2, only if the runbook requires it),
#     protocol-start / protocol-resume (item 2's production path), nve-drift, trajectories-cpu,
#     trajectories-reproducible, restart-across (item 5), compat (the compatibility tier, needs fastmace/5-T11-compat),
#     cpu-acceptance, acceptance-tier, slow-tier, targeted TEST...
# pytest jobs run through 3b-T10's stoppable_pytest.sh: results.jsonl per finished test, junit.xml also when stopped.
# Stop one with kill -INT "$(cat <root>/<job>/pytest.pid)", never by killing this script's shell or the reservation.
# regression, regression-npt, trajectories-* and restart-across need the release-4 tooling (tools/ab_fastmace.py
# --release 4, tools/trajectory_versions.py --against 4) and tools/restart_versions.py (5-T11).
set -uo pipefail
src=$(cd "${1:?SRC}" && pwd); job=${2:?JOB}; shift 2
run_id=5-T11-local-${RUN_DATE:-$(date -u +%Y%m%d)}
root=$HOME/fastmace-acceptance/$run_id
cache=$HOME/fastmace-cache/5-T11
envs=$HOME/environments
hybrid=$envs/mace_cueq_oeq/bin/python
latest=$envs/mace_cueq_latest/bin/python
checkpoint=$HOME/models/mace-mh-1.model
versions=$HOME/fastmace-5-T11/versions
anneal=$HOME/fastmace-data/anneal-minima          # 4-T08's verified frames (anneal_minima.py fetch, verify)
minima_ase=$HOME/fastmace-runs/4-T08-combined      # 4-T08's ASE sides, all ten crystals
benchmark=$HOME/fastmace-baseline/1a-T18           # Structures/Benchmark/R2SCAN, as 4-T07's --structures-root
crystals="NaCl NaNbCl6 NaTaCl6 NaNbO3 NaTaO3 NbCl3O TaCl3O NbCl2O NbCl5 TaCl5"
mkdir -p "$root" "$cache"/{tmp,fastmace,inductor,triton}
export PYTHONPATH=$src PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=4
export TMPDIR=$cache/tmp FASTMACE_CACHE=$cache/fastmace TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton
export FASTMACE_MODELS=$HOME/models FASTMACE_STRUCTURES=$benchmark
export FASTMACE_BASELINE=$benchmark FASTMACE_HYBRID_PYTHON=$hybrid
export FASTMACE_SOURCE_COMMIT=$(cat "$src/.source_commit" 2>/dev/null || echo unknown)
export FASTMACE_COMMIT=$FASTMACE_SOURCE_COMMIT  # the tools' git rev-parse fails in a WSL view of a worktree
log=$root/jobs.log
note() { echo "$(date -u +%FT%TZ) $*" | tee -a "$log"; }
run() {  # name command...
  local name=$1; shift
  note "start $name gpu=$(nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader) load=$(cut -d' ' -f1-3 /proc/loadavg) source=$FASTMACE_SOURCE_COMMIT"
  (cd "$src" && "$@") > "$root/$name.log" 2>&1 < /dev/null
  local status=$?
  note "end $name exit=$status"
  return $status
}
pytest_job() {  # name pytest-arguments...  (PYTEST_PYTHON overrides the environment, PYTEST_ENV adds variables)
  local name=$1; shift
  [ -f "$src/tools/acceptance/stoppable_pytest.sh" ] || {
    echo "tools/acceptance/stoppable_pytest.sh is 3b-T10's: merge it (or copy it from fastmace/3b-T10) first" >&2
    exit 3; }
  export FASTMACE_RESULTS=$root/$name/results  # the tests' result records (tests/gpu/test_engine_gpu.py: report)
  run "$name" env ${PYTEST_ENV:-} bash "$src/tools/acceptance/stoppable_pytest.sh" "$root/$name" \
    "${PYTEST_PYTHON:-$latest}" -p no:cacheprovider --basetemp="$cache/pytest-$name" -rfEs --durations=60 "$@"
}
need_baseline() {  # the baseline commit, for the release side: $base
  [ -n "${FASTMACE_4_COMMIT:-}" ] || {
    echo "set FASTMACE_4_COMMIT (tag fastmace-4, or the commit the conductor names)" >&2; exit 2; }
  base=$FASTMACE_4_COMMIT
}
asserted_ase() {  # the runbook's environment check: ASE 3.27 in both environments
  for python in "$hybrid" "$latest"; do
    v=$("$python" -c 'import ase; print(ase.__version__)')
    note "ase $v in $python"
    case $v in 3.27.*) ;; *) echo "ASE $v in $python: the runbook needs 3.27" >&2; exit 2 ;; esac
  done
}
# The preflight's files (the stale-expectation reruns and the restart coverage-gap file, runbook); the tiers then
# leave them out, so nothing runs twice. At the freeze the engine-vs-ASE file (about 2 h) moved to the acceptance
# tier, after the timed stretch: a defect it finds cannot change the Langevin and BAOAB timings.
PREFLIGHT_FILES=(tests/gpu/test_adapt_switch_gpu.py tests/gpu/test_calibrate_gpu.py
  tests/gpu/test_dynamic_edges_md_gpu.py tests/gpu/test_energy_gpu.py tests/gpu/test_fallback_mid_run_gpu.py
  tests/gpu/test_mc_evaluator_gpu.py tests/gpu/test_memory_gpu.py tests/gpu/test_monitor_gpu.py
  tests/gpu/test_planner_gpu.py tests/gpu/test_polar_gpu.py tests/gpu/test_reproducible_schemes_gpu.py
  tests/gpu/test_subprocess_worker_gpu.py tests/gpu/test_template_cache_gpu.py)  # changed since fastmace-4
IGNORE_PREFLIGHT=()
for f in "${PREFLIGHT_FILES[@]}"; do IGNORE_PREFLIGHT+=(--ignore="$f"); done
ab() {  # name blocks pattern sides...
  local name=$1 blocks=$2 pattern=$3; shift 3
  run "$name" "$hybrid" tools/ab_fastmace.py --release 4 --expected-commit "$base" --renamed "$versions" \
    --output "$root/$name" --model mh1-r2scan --checkpoint "$checkpoint" --structures NaCl NbCl5 --sides "$@" \
    --warmup 300 --block 300 --blocks "$blocks" --pattern "$pattern"
}
frames() {  # NAME=PATH arguments of the given crystals' captured.traj
  for c in "$@"; do printf '%s=%s ' "$c" "$anneal/$c/captured.traj"; done
}
frames_pair() {  # name optimizer [optimizer_frames arguments...]: the ASE side, then the GPU side against it
  local name=$1 optimizer=$2; shift 2
  local out=$root/optimisers/$name
  mkdir -p "$out"
  run optimisers-$name-ase "$hybrid" tools/bench/optimizer_frames.py --side ase --optimizer "$optimizer" \
    --checkpoint "$checkpoint" --modes graph --skins auto "$@" --output "$out/ase"
  run optimisers-$name-gpu "$hybrid" tools/bench/optimizer_frames.py --side gpu --optimizer "$optimizer" \
    --checkpoint "$checkpoint" --modes graph --skins auto "$@" --compare "$out/ase.json" --output "$out/gpu"
}
probe_pair() {  # name optimizer crystal frame [probe arguments...]: float64 step for step, both sides, then compare
  local name=$1 optimizer=$2 crystal=$3 frame=$4; shift 4
  local out=$root/optimisers/$name
  mkdir -p "$out"
  for side in ase gpu; do
    run optimisers-$name-$side "$hybrid" tools/validation/anneal_frame_probe.py run --side $side \
      --optimizer "$optimizer" --checkpoint "$checkpoint" --traj "$crystal=$anneal/$crystal/captured.traj" \
      --frame "$frame" --precision float64 "$@" --output "$out/$side"
  done
  run optimisers-$name-compare "$hybrid" tools/validation/anneal_frame_probe.py compare "$out/ase" "$out/gpu"
}
case "$job" in
  check-environments) asserted_ase ;;
  preflight)  # the GPU tests whose covered code changed after they last ran, and the coverage-gap files
    asserted_ase
    if [ $# -gt 0 ]; then pytest_job preflight -rA -m "slow or acceptance" "$@"
    else pytest_job preflight -rA -m "slow or acceptance" "${PREFLIGHT_FILES[@]}"; fi
    ;;
  regression)  # item 5, NVT: this checkout (4, D3 off) against the baseline, 3 same-process A/B processes
    need_baseline
    ab regression-ab-4-first-base 12 abab 4 5
    ab regression-ab-4-first 12 abab 5 4
    ab regression-ab-abba 12 abba 4 5
    ;;
  regression-npt)  # item 5, NPT, NbCl5: one version per process, alternating 4 5 5 4
    need_baseline
    for process in ${NPT_PROCESSES:-1:4 2:5 3:5 4:4}; do
      i=${process%%:*}; side=${process##*:}
      run regression-npt-$i-$side "$hybrid" tools/ab_fastmace.py --release 4 --expected-commit "$base" \
        --renamed "$versions" --output "$root/regression-npt-$i-$side" --model mh1-r2scan --checkpoint "$checkpoint" \
        --structures NbCl5 --sides "$side" --ensemble npt --warmup 300 --block 300 --blocks 12 --pattern abab
    done
    ;;
  ase-speed)  # recorded: each class, stock ASE + MACE's calculator against fastmace.ase, one side per process;
    # the fastmace sides (Langevin included) are also the step costs against Langevin and BAOAB (runbook)
    for structure in ${SPEED_STRUCTURES:-NaCl NbCl5}; do
      for class in ${SPEED_CLASSES:-Langevin VelocityVerlet Bussi NoseHooverChainNVT IsotropicMTKNPT MTKNPT \
                                     NVTBerendsen NPTBerendsen Inhomogeneous_NPTBerendsen}; do
        for side in stock fastmace; do
          run ase-speed-$side-$class-$structure "$hybrid" tools/bench/ase_integrators_speed.py --side $side \
            --integrator "$class" --structure "$structure" --output "$root/ase-speed"
        done
      done
    done
    ;;
  optimisers)  # item 1's optimiser rows; PARTS: fixed-f32 fixed-f64 fix-atoms cell-lbfgs cell-d3 cell-fire relax-cli,
    # and cell-soft NAME... (the pre-declared soft-cell extension, fmax 0.002, both sides from the same start)
    parts=${*:-fixed-f32 fixed-f64 fix-atoms cell-lbfgs cell-d3 cell-fire relax-cli}
    [ "${1:-}" = cell-soft ] && parts=cell-soft
    for part in $parts; do
      case $part in
        fixed-f32)
          for optimizer in lbfgs fire; do
            frames_pair fixed-f32-$optimizer $optimizer --traj $(frames NaCl NbCl5) --frames 0 10 20
          done ;;
        fixed-f64)
          for optimizer in lbfgs fire; do
            for crystal in NaCl NbCl5; do probe_pair fixed-f64-$optimizer-$crystal $optimizer $crystal 10; done
          done ;;
        fix-atoms)  # the ASE side on MACE's own calculator: fm.calc refuses constraints (4-T04)
          for optimizer in lbfgs fire; do
            probe_pair fix-atoms-$optimizer $optimizer NbCl5 10 --fix-every 3 --ase-calculator mace
          done ;;
        cell-lbfgs) frames_pair cell-lbfgs lbfgs --cell-filter frechet --structures $crystals \
                      --structures-root "$benchmark" ;;
        cell-d3) frames_pair cell-d3 lbfgs --cell-filter frechet --structures NbCl5 --structures-root "$benchmark" \
                   --dispersion d3bj ;;
        cell-fire) frames_pair cell-fire fire --cell-filter frechet --structures NaCl NbCl5 \
                     --structures-root "$benchmark" ;;
        cell-soft)
          shift
          [ $# -gt 0 ] || { echo "optimisers cell-soft takes the crystal names (d3: NbCl5-d3)" >&2; exit 2; }
          for name in "$@"; do
            extra=(); crystal=$name
            [ "${name%-d3}" != "$name" ] && { crystal=${name%-d3}; extra=(--dispersion d3bj); }
            frames_pair cell-soft-$name lbfgs --cell-filter frechet --structures "$crystal" \
              --structures-root "$benchmark" --fmax 0.002 "${extra[@]}"
          done ;;
        relax-cli)
          mkdir -p "$root/optimisers/relax-cli"
          run optimisers-relax-cli "$hybrid" -m fastmace.cli relax "$benchmark/Structures/Benchmark/R2SCAN/NaCl.cif" \
            --model mh1-r2scan --repeat 3 3 3 --fmax 0.01 --optimizer gpu-LBFGS \
            --output "$root/optimisers/relax-cli/NaCl-relaxed.extxyz" ;;
        *) echo "unknown optimisers part $part" >&2; exit 2 ;;
      esac
    done
    ;;
  minima)  # item 2, only if the runbook requires it: the GPU side at the freeze against 4-T08's ASE sides (copied),
    # MINIMA_ASE=1 reruns the ASE sides too; MINIMA_CRYSTALS (default all ten); then the decision-46 evaluation
    for crystal in ${MINIMA_CRYSTALS:-$crystals}; do
      out=$root/minima/$crystal
      mkdir -p "$out"
      if [ "${MINIMA_ASE:-0}" = 1 ]; then
        run minima-$crystal-ase env OMP_NUM_THREADS=2 "$hybrid" tools/bench/optimizer_frames.py --side ase \
          --checkpoint "$checkpoint" --traj "$crystal=$anneal/$crystal/captured.traj" --frames all --modes graph \
          --skins auto --resume --output "$out/ase"
      else
        cp -n "$minima_ase/$crystal/ase.json" "$minima_ase/$crystal/ase.npz" "$out/"
      fi
      run minima-$crystal-gpu "$hybrid" tools/bench/optimizer_frames.py --side gpu --checkpoint "$checkpoint" \
        --traj "$crystal=$anneal/$crystal/captured.traj" --frames all --modes graph --skins auto --resume \
        --compare "$out/ase.json" --output "$out/gpu"
    done
    run minima-evaluate python3 tools/validation/anneal_minima.py evaluate --data "$anneal" --runs "$root/minima"
    ;;
  protocol-start|protocol-resume)  # item 2's production path: the short NbCl5 NPT anneal with gpu-LBFGS (decision 47),
    # in two pieces through 3-T14's run.sh (start: relaxation and the first 13,500 steps; resume: the rest, the quench)
    dir=$root/protocol/short
    if [ "$job" = protocol-start ]; then
      run protocol-start bash results/3-T14-protocol/run.sh start "$dir" npt_anneal_NbCl5_short "${PIECE_STEPS:-13500}"
    else
      run protocol-resume-$(date -u +%H%M) bash results/3-T14-protocol/run.sh resume "$dir" "${PIECE_STEPS:-13500}"
    fi
    ;;
  nve-drift)  # recorded: NVE in the default plan, 20 ps at 1 fs
    for structure in NaCl NbCl5; do
      run nve-drift-$structure "$hybrid" tools/bench/nve_drift.py --structure $structure --output "$root/nve-drift"
    done
    ;;
  trajectories-cpu)  # item 5: CPU float64, tiny MH-1-class model, NaCl 2x2x2, NVT and NPT, D3 off and on; --bitwise
    need_baseline
    tiny=$root/tiny-mh1-11-17.model
    [ -f "$tiny" ] || (cd "$src" && env CUDA_VISIBLE_DEVICES= PYTHONPATH="$src:$src/tests" "$hybrid" -c "
import sys, torch
from _support.tiny_models import FAMILIES, tiny_mace_model
torch.save(tiny_mace_model([11, 17], **FAMILIES['mh1']), sys.argv[1])" "$tiny")
    for ensemble in nvt npt; do
      for dispersion in none d3bj; do
        for side in 4 5; do
          run traj-cpu-$ensemble-$dispersion-$side env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 "$hybrid" \
            tools/trajectory_versions.py run --against 4 --side $side --expected-commit "$base" --renamed "$versions" \
            --device cpu --backend auto --ensemble $ensemble --dispersion $dispersion \
            --output "$root/traj-cpu-$ensemble-$dispersion-$side" --model "$tiny" --checkpoint "$tiny" \
            --structure NaCl@2x2x2 --steps 100 --every 10 --seeds 20261023 --positions
        done
        run traj-cpu-$ensemble-$dispersion-compare "$hybrid" tools/trajectory_versions.py compare \
          --release "$root/traj-cpu-$ensemble-$dispersion-4" --current "$root/traj-cpu-$ensemble-$dispersion-5" --bitwise
      done
    done
    ;;
  trajectories-reproducible)  # item 5: the reproducible plan, plain cuEq float64, MH-1 NaCl, NVT and NPT; --bitwise
    need_baseline
    for ensemble in nvt npt; do
      for side in 4 5; do
        run traj-repro-$ensemble-$side "$latest" tools/trajectory_versions.py run --against 4 --side $side \
          --expected-commit "$base" --renamed "$versions" --reproducible --backend cueq --ensemble $ensemble \
          --output "$root/traj-repro-$ensemble-$side" --checkpoint "$checkpoint" --structure NaCl --steps 500 \
          --every 50 --seeds 20261023 --positions
      done
      run traj-repro-$ensemble-compare "$latest" tools/trajectory_versions.py compare \
        --release "$root/traj-repro-$ensemble-4" --current "$root/traj-repro-$ensemble-5" --bitwise
    done
    ;;
  restart-across)  # item 5: a baseline (format 3) restart at step 50 resumed at the freeze to 100, CPU float64, tiny
    # model; the release stages run a git archive of the baseline as the plain package (tools/restart_versions.py)
    need_baseline
    tiny=$root/tiny-mh1-11-17.model
    [ -f "$tiny" ] || { echo "run trajectories-cpu first (it writes the tiny model)" >&2; exit 2; }
    release=$HOME/fastmace-5-T11/release-$base
    [ -d "$release/fastmace" ] || { mkdir -p "$release" && git -C "${FASTMACE_REPOSITORY:-/mnt/d/Margraf/FastMACE}"       archive "$base" | tar -x -C "$release" && git -C "${FASTMACE_REPOSITORY:-/mnt/d/Margraf/FastMACE}"       rev-parse "$base" > "$release/.source_commit"; }
    common=(--model "$tiny" --checkpoint "$tiny" --structure NaCl@2x2x2 --steps 100 --restart-step 50 --seed 20261023)
    for stage in write uninterrupted; do
      run restart-across-$stage env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONPATH= "$hybrid"         tools/restart_versions.py run --stage $stage --release-root "$release" --expected-commit "$base"         "${common[@]}" --output "$root/restart-across/$stage"
    done
    run restart-across-resume env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 "$hybrid" tools/restart_versions.py run       --stage resume --source "$root/restart-across/write" "${common[@]}" --output "$root/restart-across/resume"
    run restart-across-compare "$hybrid" tools/restart_versions.py compare "$root/restart-across/uninterrupted"       "$root/restart-across/resume"
    ;;
  compat)  # the GPU compatibility check on this Blackwell GPU (SPEC decision 39, amended; tests/_support/tiers.py)
    PYTEST_PYTHON=${PYTEST_PYTHON:-$latest}  # pytest lives here; hybrid children use FASTMACE_HYBRID_PYTHON
    run compat "$PYTEST_PYTHON" tools/acceptance/compat.py --provider local --label "RTX 5060 Ti (local)"       --output "$root/compat"
    ;;
  cpu-acceptance)  # recorded: the sampling checks (choice 8), CPU, in the GPU environment's ASE 3.27
    PYTEST_ENV="CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2" pytest_job cpu-acceptance -rA tests/cpu -m acceptance
    ;;
  targeted)  # the tests corrected at acceptance
    [ $# -gt 0 ] || { echo "targeted takes the test node ids" >&2; exit 2; }
    pytest_job targeted -rA "$@"
    ;;
  acceptance-tier) pytest_job acceptance-tier tests/gpu -m acceptance "${IGNORE_PREFLIGHT[@]}" ;;
  slow-tier) pytest_job slow-tier tests/gpu -m "slow and not acceptance" "${IGNORE_PREFLIGHT[@]}" ;;
  *) echo "unknown job $job" >&2; exit 2 ;;
esac
