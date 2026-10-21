#!/usr/bin/env bash
# 2-T23: OMOL-0 XL's gate on murphy after gate-murphy: as frozen, and with OPT-T06's template cache off.
set -uo pipefail
B=/mnt/homes/zwingel/fastmace-2-T23/batches
while ! grep -q 'gate-murphy exit' $B/combined-murphy.log; do sleep 60; done
SRC=/mnt/homes/zwingel/fastmace-2-T23/src-8e35fc4
echo "gate-omol0 start $(date -u +%FT%TZ)"
python3 $SRC/tools/acceptance/run.py gate-omol0 --machine murphy --run-id 2-T23-murphy-20261021-omol0 --fastmace $SRC
echo "gate-omol0 exit $? $(date -u +%FT%TZ)"
