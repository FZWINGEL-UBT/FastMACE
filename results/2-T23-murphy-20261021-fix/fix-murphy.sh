#!/usr/bin/env bash
# 2-T23 rerun on the fix (main f993713; tools dda0323) on murphy: item 4's repeat, the Simulation tests, OMOL-0 XL.
set -uo pipefail
SRC=/mnt/homes/zwingel/fastmace-2-T23/src-dda0323
RUN=2-T23-murphy-20261021-fix
for suite in runner-repeat simulation-tests; do
  echo "$suite start $(date -u +%FT%TZ)"
  python3 $SRC/tools/acceptance/run.py $suite --machine murphy --run-id $RUN --fastmace $SRC
  echo "$suite exit $? $(date -u +%FT%TZ)"
done
echo "gate-omol0 start $(date -u +%FT%TZ)"
python3 $SRC/tools/acceptance/run.py gate-omol0 --machine murphy --run-id $RUN --fastmace $SRC --only gate-omol0-xl
echo "gate-omol0 exit $? $(date -u +%FT%TZ)"
