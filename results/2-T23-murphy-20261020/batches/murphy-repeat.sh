#!/usr/bin/env bash
# 2-T23 on murphy: item 4's repeat (owner, 20 October 2026), after the timed batch.
set -uo pipefail
B=/mnt/homes/zwingel/fastmace-2-T23/batches
while ! grep -q 'acceptance-tier exit' $B/murphy-timed.log; do sleep 60; done
SRC=/mnt/homes/zwingel/fastmace-2-T23/src-36766f6
echo "runner-repeat start $(date -u +%FT%TZ)"
python3 $SRC/tools/acceptance/run.py runner-repeat --machine murphy --run-id 2-T23-murphy-20261020 --fastmace $SRC
echo "runner-repeat exit $? $(date -u +%FT%TZ)"
