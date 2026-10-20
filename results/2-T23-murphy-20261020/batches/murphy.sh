#!/usr/bin/env bash
# 2-T23 on murphy: timed suites first (the GPU must have no process of anyone), then untimed ones.
set -uo pipefail
SRC=/mnt/homes/zwingel/fastmace-2-T23/src-37c4014
RUN=2-T23-murphy-20261020
for suite in regression2 acceptance-tier trajectory2 runner-statistics full-tier; do
  echo "$suite start $(date -u +%FT%TZ)"
  python3 $SRC/tools/acceptance/run.py $suite --machine murphy --run-id $RUN --fastmace $SRC \
      --versions /mnt/homes/zwingel/fastmace-2-T23/versions
  echo "$suite exit $? $(date -u +%FT%TZ)"
done
