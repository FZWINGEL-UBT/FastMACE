#!/usr/bin/env bash
# The Ada scan's first failures were 900 s timeouts (SrTiO3 32^3; NaCl 34^3 and 36^3, the NaCl ones while another
# user's GPU job had started). Rerun each boundary point alone with a 40-minute limit, idle check before each, and a
# GPU process log every 30 s (gpu-STRUCTURE-REPEAT.log) to show whether anything else shared the card.
#   ROOT=<dir> bash edge.sh SRC STRUCTURE:REPEAT...
set -u -o pipefail
FM=$1; shift
source /mnt/homes/zwingel/fm-ada-capacity/jobs2/common.sh
cd "$FM" && export PYTHONPATH=$FM
note "edge start fastmace=$(cat .source_commit)"
for job in "$@"; do
  structure=${job%%:*} repeat=${job#*:}
  # Workstation rule: before every point, wait (do not stop) until no other user's process or GPU job is left,
  # then two quiet minutes.
  quiet=0; last=""
  while [ $quiet -lt 2 ]; do
    if why=$(others); then quiet=0; [ "${why//[0-9]/}" != "$last" ] && note "waiting before $job: $why"; last=${why//[0-9]/}; else quiet=$((quiet + 1)); fi
    sleep 60
  done
  ( while true; do echo "$(date -u +%FT%TZ) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | paste -sd' ' -)"; sleep 30; done ) > "$ROOT/gpu-$structure-$repeat.log" 2>&1 &
  mon=$!
  timeout 2400 python tools/bench/capacity_scan.py point --diagnose --structure "$structure" --model omat0-medium --precision float32 \
    --output "$ROOT/points" --repeat "$repeat" > "$ROOT/point-$structure-$repeat.log" 2>&1 < /dev/null
  rc=$?; kill $mon
  note "end $job exit=$rc $(tail -n 2 "$ROOT/point-$structure-$repeat.log" | tr '\n' ' ' | cut -c1-300)"
done
note DONE
