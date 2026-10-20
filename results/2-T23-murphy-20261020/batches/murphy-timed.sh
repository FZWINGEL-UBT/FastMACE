#!/usr/bin/env bash
# 2-T23 on murphy, timed: item 8 A/B, then the acceptance tier. Records other users load every 60 s.
set -uo pipefail
SRC=/mnt/homes/zwingel/fastmace-2-T23/src-4b35b60
B=/mnt/homes/zwingel/fastmace-2-T23/batches
( while true; do echo "$(date -u +%FT%TZ) load $(cut -d" " -f1-3 /proc/loadavg) others_pcpu $(ps -eo user=,pcpu= | awk "\$1!=\"zwingel\" {s+=\$2} END {print s+0}") gpu_procs $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"; sleep 60; done ) > $B/murphy-load.log 2>&1 &
LOGGER=$!
trap "kill $LOGGER" EXIT
for suite in regression2 acceptance-tier; do
  echo "$suite start $(date -u +%FT%TZ) load $(cut -d" " -f1-3 /proc/loadavg)"
  python3 $SRC/tools/acceptance/run.py $suite --machine murphy --run-id 2-T23-murphy-20261020 --fastmace $SRC \
      --versions /mnt/homes/zwingel/fastmace-2-T23/versions
  echo "$suite exit $? $(date -u +%FT%TZ)"
done
