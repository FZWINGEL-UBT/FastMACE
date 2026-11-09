#!/usr/bin/env bash
# leto: capacity points (capacity_scan.py point, 40-minute limit) per structure in the given order, stopping a
# structure at its first failure. Before each point: wait until no other user and no GPU job (we wait, never stop),
# then two quiet minutes; the GPU's processes every 30 s in gpu-<structure>-<repeat>.log.
#   ROOT=<dir> bash points.sh SRC STRUCTURE:REPEAT,REPEAT,...  ...
set -u -o pipefail
FM=$1; shift
source /mnt/homes/zwingel/fm-ada-capacity/jobs2/common.sh
cd "$FM" && export PYTHONPATH=$FM
busy() {  # as leto.sh: our own CPU suites may run (capacity, not timing)
  local gpu users
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
  users=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $2 > 20 {print $1":"$3":"$2}' | paste -sd, -)
  [ -n "$gpu$users" ] && { echo "gpu=[$gpu] users=[$users]"; return 0; }
  return 1
}
note "points start fastmace=$(cat .source_commit)"
for spec in "$@"; do
  structure=${spec%%:*}
  for repeat in $(echo "${spec#*:}" | tr ',' ' '); do
    quiet=0; last=""
    while [ $quiet -lt 4 ]; do
      if why=$(busy); then quiet=0; [ "${why//[0-9]/}" != "$last" ] && note "waiting before $structure $repeat: $why"; last=${why//[0-9]/}; else quiet=$((quiet + 1)); fi
      sleep 30
    done
    ( while true; do echo "$(date -u +%FT%TZ) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | paste -sd' ' -)"; sleep 30; done ) > "$ROOT/gpu-$structure-$repeat.log" 2>&1 &
    mon=$!
    timeout 2400 python tools/bench/capacity_scan.py point --structure "$structure" --model omat0-medium --precision float32 \
      --output "$ROOT/points" --repeat "$repeat" > "$ROOT/point-$structure-$repeat.log" 2>&1 < /dev/null
    rc=$?; kill $mon
    note "end $structure $repeat exit=$rc $(tail -n 2 "$ROOT/point-$structure-$repeat.log" | tr '\n' ' ' | cut -c1-300)"
    # a failing point exits 0 and reports "): oom" (or another status) on its "point ..." line
    grep -q "^point .*): ok," "$ROOT/point-$structure-$repeat.log" || break
  done
done
note "POINTS DONE"
