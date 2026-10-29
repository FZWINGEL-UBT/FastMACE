#!/usr/bin/env bash
# Repeats 13-minute reservations of one scan until it prints "scan: done" (at most 12 rounds):
# (TAG and REWRITES pass through to local.sh)
#   bash results/capacity-16gb-20261029/scan_loop.sh SRC ROOT STRUCTURE PRECISION [START_ATOMS]
set -u
SRC=$1 ROOT=$2 S=$3 P=$4 START=${5:-}
for round in $(seq 1 12); do
  python3 "$SRC/tools/ci/local_gpu.py" run --purpose "fm capacity-16gb scan $S $P round $round (<15 min)" -- \
    env FM="$SRC" bash "$SRC/results/capacity-16gb-20261029/local.sh" "$ROOT" "scan:$S:$P${START:+:$START}"
  if grep -q "scan: done $S $P${TAG:-}: " "$ROOT/jobs.log"; then grep "scan: done $S $P${TAG:-}: " "$ROOT/jobs.log"; exit 0; fi
done
echo "scan $S $P not done after 12 rounds"; exit 1
