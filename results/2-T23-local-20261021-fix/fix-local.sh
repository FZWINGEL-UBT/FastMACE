#!/usr/bin/env bash
# 2-T23 rerun on the fix (main f993713; tools dda0323), local RTX 5060 Ti: the Simulation tests.
set -uo pipefail
SRC=~/fastmace-2-T23/src-dda0323
echo "simulation-tests start $(date -u +%FT%TZ)"
python3 $SRC/tools/acceptance/run.py simulation-tests --machine local --run-id 2-T23-local-20261021-fix --fastmace $SRC
echo "simulation-tests exit $? $(date -u +%FT%TZ)"
