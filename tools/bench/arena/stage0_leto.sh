#!/usr/bin/env bash
# leto (RTX 4000 Ada, NAS mace_cueq_oeq): capture-arena stage 0. Build the shim with the system gcc, the toy capture,
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
  python "$S" build "$ROOT" && echo "build exit=0"
  python "$S" toy "$ROOT"; echo "toy exit=$?"
  python "$S" md "$ROOT" --repeat "$REPEAT" --tag a; echo "md a exit=$?"
  python "$S" md "$ROOT" --repeat "$REPEAT" --tag b; echo "md b exit=$?"
  python "$S" compare "$ROOT/md-a.json" "$ROOT/md-b.json"; echo "compare exit=$?"
  python "$S" md "$ROOT" --repeat "$REPEAT" --tag c --no-split; echo "md c exit=$?"
  python "$S" md "$ROOT" --repeat "$REPEAT" --tag d --no-split; echo "md d exit=$?"
  python "$S" compare "$ROOT/md-c.json" "$ROOT/md-d.json"; echo "compare no-split exit=$?"
} > "$ROOT/stage0.log" 2>&1 < /dev/null
note "STAGE0 DONE"
