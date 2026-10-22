#!/usr/bin/env bash
# 3-T21: phase 3 acceptance jobs on the local RTX 5060 Ti (WSL), one job per call, from the pinned copy.
#
#   bash tools/acceptance/phase3_local.sh SRC JOB
#
# SRC is the pinned copy (git archive of the 3-T21 head, with .source_commit); its fastmace/ equals the freeze's.
# Run every call under the GPU reservation: python3 SRC/tools/ci/local_gpu.py run --purpose "3-T21 JOB" -- bash ...
# Outputs: ~/fastmace-acceptance/3-T21-local-<date>/<job>/ and <job>.log beside it.
# Jobs (docs/validation/phase-3-runbook.md):
#   timed (quiet PC): regression3 (item 10), events (item 1, swap-back), mc-cost (item 2), rule9
#   untimed: acceptance-tier, slow-tier, protocol (item 9), delta-e (item 3)
set -uo pipefail
src=$(cd "${1:?SRC}" && pwd); job=${2:?JOB}
run_id=3-T21-local-${RUN_DATE:-$(date -u +%Y%m%d)}
root=$HOME/fastmace-acceptance/$run_id
cache=$HOME/fastmace-cache/3-T21
envs=$HOME/environments
hybrid=$envs/mace_cueq_oeq/bin/python
latest=$envs/mace_cueq_latest/bin/python
checkpoint=$HOME/models/mace-mh-1.model
versions=$HOME/fastmace-3-T21/versions
mkdir -p "$root" "$cache"/{tmp,fastmace,inductor,triton}
export PYTHONPATH=$src PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=4
export TMPDIR=$cache/tmp FASTMACE_CACHE=$cache/fastmace TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton
export FASTMACE_MODELS=$HOME/models FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18
export FASTMACE_BASELINE=$HOME/fastmace-baseline/1a-T18 FASTMACE_HYBRID_PYTHON=$hybrid
export FASTMACE_SOURCE_COMMIT=$(cat "$src/.source_commit")
log=$root/jobs.log
note() { echo "$(date -u +%FT%TZ) $*" | tee -a "$log"; }
run() {  # name command...
  local name=$1; shift
  note "start $name gpu=$(nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader) source=$FASTMACE_SOURCE_COMMIT"
  (cd "$src" && "$@") > "$root/$name.log" 2>&1 < /dev/null
  note "end $name exit=$?"
}
tier() {  # name marker
  run "$1" "$latest" -m pytest tests/gpu -m "$2" -p no:cacheprovider --basetemp="$cache/pytest-$1" -rfEs \
    --durations=60 --junitxml="$root/$1/junit.xml"
}
ab() {  # name blocks pattern sides...
  local name=$1 blocks=$2 pattern=$3; shift 3
  run "$name" "$hybrid" tools/ab_fastmace.py --release p3b --renamed "$versions" --output "$root/$name" \
    --model mh1-r2scan --checkpoint "$checkpoint" --structures NaCl NbCl5 --sides "$@" --warmup 300 --block 300 \
    --blocks "$blocks" --pattern "$pattern" --dump-forces
}
case "$job" in
  regression3)  # item 10: this checkout (3, fixcm_projection=ase) against phase-3-baseline, 3 A/B processes + controls
    ab regression3-ab-abab 12 abab p3b 3
    ab regression3-ab-3-first 12 abab 3 p3b
    ab regression3-ab-abba 12 abba p3b 3
    ab regression3-control-p3b 4 abab p3b
    ab regression3-control-3 4 abab 3
    ;;
  events)  # item 1: 3-T20's swap-back series (8 runs)
    for structure in NaCl NbCl5; do
      for superset in "" "Na Nb O Cl"; do
        tag=$structure${superset:+-superset}
        run events-$tag-ab "$hybrid" tools/bench/event_speed.py --output "$root/events-$tag-ab" --checkpoint "$checkpoint" \
          --structures "$structure" ${superset:+--superset $superset} --order ab --pattern abab --settle 5 --swap-back
        run events-$tag-ba "$hybrid" tools/bench/event_speed.py --output "$root/events-$tag-ba" --checkpoint "$checkpoint" \
          --structures "$structure" ${superset:+--superset $superset} --order ba --pattern abba --settle 5 --swap-back
      done
    done
    ;;
  mc-cost)  # item 2: 3-T20's mc_cost runs
    run mc-NaCl-swap "$hybrid" tools/bench/mc_cost.py --output "$root/mc-NaCl-swap" --checkpoint "$checkpoint" \
      --structure NaCl --temperature-K 1500 --block 100 --rounds 8
    run mc-NbCl5-swap "$hybrid" tools/bench/mc_cost.py --output "$root/mc-NbCl5-swap" --checkpoint "$checkpoint" \
      --structure NbCl5 --temperature-K 1500 --block 50 --rounds 8
    for structure in NaCl NbCl5; do
      run mc-$structure-displace "$hybrid" tools/bench/mc_cost.py --output "$root/mc-$structure-displace" \
        --checkpoint "$checkpoint" --structure "$structure" --temperature-K 1500 --displace-delta-A 0.1 --block 20 --rounds 8
    done
    ;;
  rule9)  # decision 38's measurement: graph against compiled with host callbacks
    for structure in NaCl NbCl5; do
      run rule9-$structure "$hybrid" tools/ab_host_callbacks.py --label mh1-r2scan --structure "$structure" \
        --intervals 1 5 10 --output "$root/rule9/$structure.json"
    done
    ;;
  acceptance-tier) tier acceptance-tier acceptance ;;
  slow-tier) tier slow-tier "slow and not acceptance" ;;
  protocol)  # item 9: the shipped swap-MC protocol end to end (3-T14's run.sh, in two pieces)
    export FASTMACE_COMMIT=$FASTMACE_SOURCE_COMMIT
    run protocol-start bash results/3-T14-protocol/run.sh start "$root/protocol/full" swap_anneal_NaNbOCl4 135000
    run protocol-resume bash results/3-T14-protocol/run.sh resume "$root/protocol/full" 135000
    ;;
  delta-e)  # item 3: MH-1 matpes_r2scan and omol heads (OMOL-0 XL: 3-T11's finding, not rerun)
    for head in matpes_r2scan omol; do
      run de-$head "$hybrid" tools/validation/move_energy_precision.py --model mh1 --head "$head" \
        --checkpoint "$checkpoint" --structures NaCl NbCl5 NaNbO3 --output "$root/de-$head"
    done
    ;;
  *) echo "unknown job $job" >&2; exit 2 ;;
esac
