#!/usr/bin/env bash
# 2-T23 on murphy after the acceptance tier: three NaCl A/B processes (timed; load recorded), then item 4's repeat.
set -uo pipefail
B=/mnt/homes/zwingel/fastmace-2-T23/batches
while ! grep -q 'acceptance-tier exit' $B/murphy-timed.log; do sleep 60; done
SRC=/mnt/homes/zwingel/fastmace-2-T23/src-a8eee21
( while true; do echo "$(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg) others_pcpu $(ps -eo user=,pcpu= | awk '$1!="zwingel" {s+=$2} END {print s+0}') gpu_procs $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"; sleep 60; done ) > $B/murphy-load-nacl.log 2>&1 &
LOGGER=$!
echo "regression2-nacl start $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)"
python3 $SRC/tools/acceptance/run.py regression2-nacl --machine murphy --run-id 2-T23-murphy-20261020 --fastmace $SRC --versions /mnt/homes/zwingel/fastmace-2-T23/versions
echo "regression2-nacl exit $? $(date -u +%FT%TZ)"
kill $LOGGER
echo "runner-repeat start $(date -u +%FT%TZ)"
python3 $SRC/tools/acceptance/run.py runner-repeat --machine murphy --run-id 2-T23-murphy-20261020 --fastmace $SRC
echo "runner-repeat exit $? $(date -u +%FT%TZ)"
