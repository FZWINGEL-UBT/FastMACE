#!/usr/bin/env bash
# 2-T23 on murphy, untimed suites.
set -uo pipefail
SRC=/mnt/homes/zwingel/fastmace-2-T23/src-37c4014
for suite in trajectory2 runner-statistics; do
  echo "$suite start $(date -u +%FT%TZ)"
  python3 $SRC/tools/acceptance/run.py $suite --machine murphy --run-id 2-T23-murphy-20261020 --fastmace $SRC \
      --versions /mnt/homes/zwingel/fastmace-2-T23/versions
  echo "$suite exit $? $(date -u +%FT%TZ)"
done
