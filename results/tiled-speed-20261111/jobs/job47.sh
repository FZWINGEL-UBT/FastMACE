#!/bin/bash
# tiled job 47 (11 November 2026): the GPU test of compiled_graphs beside uncaptured calls (fixes 35bcd4ab and
# d7d9e7ce), tests/gpu/test_tiled_speed_cuda.py::test_compiled_graphs_alternate_with_uncaptured_calls.
set -u
RUN=/home/fzwin/fm-runs/tiled-speed-20261108
SRC=$RUN/src-${COMMIT}
OUT=$RUN/${JOB}
mkdir -p $OUT
export PATH=/home/fzwin/environments/mace_cueq_oeq/bin:/usr/lib/wsl/lib:$PATH PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1
export FASTMACE_MODELS=/home/fzwin/models FASTMACE_SOURCE_COMMIT=$COMMIT
unset PYTORCH_CUDA_ALLOC_CONF
cd $SRC
echo "== pytest $(date +%T) $(cut -d' ' -f1-3 /proc/loadavg)"
python -m pytest -q -s -p no:cacheprovider --basetemp=$OUT/tmp \
  "tests/gpu/test_tiled_speed_cuda.py::test_compiled_graphs_alternate_with_uncaptured_calls" > $OUT/pytest.log 2>&1
echo "exit=$? $(date +%T)"; tail -15 $OUT/pytest.log | cut -c1-400
echo done
