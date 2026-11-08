#!/usr/bin/env bash
# The default plan's capacity limit for OMAT-0 medium (float32 NVT, capacity_scan.py scan: 20 warm-up and 200 timed
# steps per size, the largest passing cubic repeat) on the RTX 4000 Ada, idle check before each structure.
#   ROOT=<dir> bash scan.sh SRC_DIR STRUCTURE[:START_ATOMS]...
set -u -o pipefail
FM=$1; shift
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$FM" && export PYTHONPATH=$FM
note "scan start fastmace=$(cat .source_commit) $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
for job in "$@"; do
  structure=${job%%:*} start=${job#*:}
  [ "$start" = "$job" ] && start=4000
  if why=$(others); then note "STOP before scan $structure: $why"; exit 3; fi
  python tools/bench/capacity_scan.py scan --structure "$structure" --model omat0-medium --precision float32 \
    --output "$ROOT/points" --start-atoms "$start" --growth ${GROWTH:-1.25} --budget-minutes ${BUDGET:-150} > "$ROOT/scan-$structure.log" 2>&1 < /dev/null
  note "end scan $structure exit=$? $(tail -n 2 "$ROOT/scan-$structure.log" | tr '\n' ' ' | cut -c1-300)"
done
note DONE
