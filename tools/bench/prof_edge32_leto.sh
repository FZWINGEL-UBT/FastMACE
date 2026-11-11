#!/usr/bin/env bash
# leto (RTX 4000 Ada): memory profiles (tools/bench/memory_profile.py, md case) of SrTiO3 at REPEAT with
# fastmace.neighbors.skin.EDGE_INDEX_INT32 off and on, to find why the int32 side reserves more. Idle check (four
# quiet checks 30 s apart) before each side. The omat0-ada-capacity common.sh (environment, others()).
#   ROOT=<dir> bash prof_edge32_leto.sh SRC_DIR [REPEAT]
set -u -o pipefail
FM=$1; REPEAT=${2:-34x34x34}
source /mnt/homes/zwingel/fm-ada-capacity/jobs2/common.sh
cd "$FM" && export PYTHONPATH=$FM
note "prof_edge32 start fastmace=$(cat .source_commit) repeat=$REPEAT"
for side in off on; do
  value=$([ "$side" = on ] && echo True || echo False)
  quiet=0; last=""
  while [ $quiet -lt 4 ]; do
    if why=$(others); then quiet=0; [ "${why//[0-9]/}" != "$last" ] && note "waiting before $side: $why"; last=${why//[0-9]/}; else quiet=$((quiet + 1)); fi
    sleep 30
  done
  python tools/bench/memory_profile.py --case md --structure SrTiO3 --repeat "$REPEAT" --output "$ROOT/profiles-$side" \
    --constant fastmace.neighbors.skin:EDGE_INDEX_INT32=$value > "$ROOT/md-$REPEAT-$side.log" 2>&1 < /dev/null
  note "end $side exit=$? $(grep -m1 'atoms:' "$ROOT/md-$REPEAT-$side.log" | cut -c1-300)"
done
note "PROF_EDGE32 DONE"
