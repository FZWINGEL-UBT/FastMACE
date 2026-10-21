#!/usr/bin/env bash
# 2-T23 combined acceptance, local RTX 5060 Ti (runbook L1-L7), freeze be92979, tools a3eaf89.
set -uo pipefail
SRC=~/fastmace-2-T23/src-a3eaf89
RUN=2-T23-local-20261021-combined
ROOT=~/fastmace-acceptance/$RUN
CACHE=~/fastmace-cache/acceptance
mkdir -p $ROOT/item1-0315 $ROOT/tmp
echo "L1 item1 start $(date -u +%FT%TZ)"
env PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1 FASTMACE_MODELS=~/models FASTMACE_CACHE=$CACHE \
    TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor-mace0315 TRITON_CACHE_DIR=$CACHE/triton TMPDIR=$ROOT/tmp \
    TORCHINDUCTOR_COMPILE_THREADS=4 \
    ~/environments/fastmace_mace_other/bin/python $SRC/results/2-T04-acc1-0315-local-20261020b/acc1.py $ROOT/item1-0315 \
    > $ROOT/item1-0315/run.log 2>&1
echo "L1 item1 exit $? $(date -u +%FT%TZ)"
for suite in regression2 acceptance-tier gate-local trajectory2 round-tests full-tier; do
  echo "$suite start $(date -u +%FT%TZ)"
  python3 $SRC/tools/acceptance/run.py $suite --machine local --run-id $RUN --fastmace $SRC --versions ~/fastmace-versions
  echo "$suite exit $? $(date -u +%FT%TZ)"
done
