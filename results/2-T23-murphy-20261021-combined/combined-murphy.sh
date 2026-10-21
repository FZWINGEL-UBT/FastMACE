#!/usr/bin/env bash
# 2-T23 combined acceptance on murphy (runbook M1-M4), freeze be92979, tools a3eaf89.
set -uo pipefail
SRC=/mnt/homes/zwingel/fastmace-2-T23/src-a3eaf89
B=/mnt/homes/zwingel/fastmace-2-T23/batches
RUN=2-T23-murphy-20261021-combined
others() { ps -eo user=,pcpu= | awk '$1!="zwingel" {s+=$2} END {print s+0}'; }
( while true; do echo "$(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg) others_pcpu $(others) gpu_procs $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"; sleep 60; done ) > $B/combined-murphy-load.log 2>&1 &
LOGGER=$!
trap "kill $LOGGER" EXIT
quiet() {  # wait up to 30 min for other users' load to be about 4 or below (400 %CPU); record either way
  for i in $(seq 1 30); do
    if [ "$(others | cut -d. -f1)" -le 400 ]; then echo "quiet: others_pcpu $(others) load $(cut -d' ' -f1-3 /proc/loadavg)"; return; fi
    sleep 60
  done
  echo "NOT QUIET after 30 min: others_pcpu $(others) load $(cut -d' ' -f1-3 /proc/loadavg)"
}
for suite in regression2 regression2-nacl acceptance-tier runner-repeat gate-murphy; do
  case $suite in regression2|regression2-nacl|acceptance-tier) quiet;; esac
  echo "$suite start $(date -u +%FT%TZ)"
  python3 $SRC/tools/acceptance/run.py $suite --machine murphy --run-id $RUN --fastmace $SRC \
      --versions /mnt/homes/zwingel/fastmace-2-T23/versions
  echo "$suite exit $? $(date -u +%FT%TZ)"
done
