#!/usr/bin/env bash
# 2-T23 on murphy after 2-T21's jobs: the gate's 4 MACE-OFF labels and OMOL-0 XL as frozen; then the 2-T19b fix check.
set -uo pipefail
FROZEN=/mnt/homes/zwingel/fastmace-2-T23/src-8e35fc4
FIX=/mnt/homes/zwingel/fastmace-2-T23/src-fixcheck-911ad7b
echo "gate-rest start $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)"
python3 $FROZEN/tools/acceptance/run.py gate-murphy --machine murphy --run-id 2-T23-murphy-20261021-gate-rest --fastmace $FROZEN \
    --only gate-off23-medium gate-off23-large gate-off23b-medium gate-off24-medium
echo "gate-rest exit $? $(date -u +%FT%TZ)"
echo "omol0-frozen start $(date -u +%FT%TZ)"
python3 $FROZEN/tools/acceptance/run.py gate-omol0 --machine murphy --run-id 2-T23-murphy-20261021-omol0 --fastmace $FROZEN --only gate-omol0-xl
echo "omol0-frozen exit $? $(date -u +%FT%TZ)"
echo "omol0-fixcheck start $(date -u +%FT%TZ)"
python3 $FIX/tools/acceptance/run.py gate-local --machine murphy --run-id 2-T23-murphy-20261021-omol0-fixcheck --fastmace $FIX \
    --only gate-omol0-xl gate-omol0-xl-NaNbO3-model-limited
echo "omol0-fixcheck exit $? $(date -u +%FT%TZ)"
