#!/usr/bin/env bash
# 2-T23: acceptance reruns of the two tests that failed locally, with test-only fixes (2-T15d 8ec5063, opt-T07b 9f2aca7).
set -uo pipefail
SRC=~/fastmace-2-T23/src-rerun-8ec5063-9f2aca7
ROOT=~/fastmace-acceptance/2-T23-local-20261021-combined/rerun-two-tests
CACHE=~/fastmace-cache/acceptance
mkdir -p $ROOT/tmp
echo "rerun start $(date -u +%FT%TZ)"
env PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$CACHE FASTMACE_MODELS=~/models \
    FASTMACE_STRUCTURES=~/fastmace-baseline/1a-T18 TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton \
    MACE_COMPILE_CACHE=$CACHE/compile XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 \
    FASTMACE_HYBRID_PYTHON=~/environments/mace_cueq_oeq/bin/python FASTMACE_BASELINE=~/fastmace-baseline/1a-T18 \
    ~/environments/mace_cueq_latest/bin/python -m pytest -p no:cacheprovider -rfEs --durations=10 \
    --basetemp=$ROOT/pytest --junitxml=$ROOT/junit.xml -m acceptance \
    "$SRC/tests/gpu/test_fallback_gpu.py::test_a_failed_candidates_memory_is_freed_before_the_next_build" \
    "$SRC/tests/gpu/test_subprocess_worker_gpu.py::test_a_killed_parent_releases_the_workers_gpu_memory" \
    > $ROOT/pytest.log 2>&1
echo "rerun exit $? $(date -u +%FT%TZ)"
