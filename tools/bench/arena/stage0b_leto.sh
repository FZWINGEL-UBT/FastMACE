#!/usr/bin/env bash
# two MD records of SrTiO3 (default pool) and two with no_split, each pair compared. Waits for the idle check (no
# other user above 20 % CPU, no GPU process, no pytest) with four quiet checks 30 s apart before starting.
#   ROOT=<dir> bash stage0_leto.sh SRC REPEAT
set -u -o pipefail
FM=$1; REPEAT=$2
source /mnt/homes/zwingel/fm-ada-capacity/jobs2/common.sh
cd "$FM" && export PYTHONPATH=$FM
note "stage0 start fastmace=$(cat .source_commit) repeat=$REPEAT"
quiet=0; last=""
while [ $quiet -lt 4 ]; do
  if why=$(others); then quiet=0; [ "${why//[0-9]/}" != "$last" ] && note "waiting: $why"; last=${why//[0-9]/}; else quiet=$((quiet + 1)); fi
  sleep 30
done
S=tools/bench/arena/arena_stage0.py
{
  cp /mnt/homes/zwingel/fm-arena/ada-36/arena_shim.so "$ROOT/" && echo "shim copied"
  python "$S" md "$ROOT" --repeat "$REPEAT" --tag a; echo "md a exit=$?"
  python "$S" md "$ROOT" --repeat "$REPEAT" --tag b; echo "md b exit=$?"
  python "$S" compare "$ROOT/md-a.json" "$ROOT/md-b.json"; echo "compare exit=$?"
} > "$ROOT/stage0.log" 2>&1 < /dev/null
note "STAGE0 DONE"
