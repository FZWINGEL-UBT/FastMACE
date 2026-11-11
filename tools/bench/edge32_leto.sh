#!/usr/bin/env bash
# leto (RTX 4000 Ada, NAS mace_cueq_oeq): the int32 edge_index falsifier, off then on, then the comparison, per
# REPEAT. Waits for the idle check (no other user above 20 % CPU, no GPU process, no pytest; four quiet checks 30 s
# apart) before each repeat.
#   ROOT=<dir> [ONLY_REPRODUCIBLE=1] bash edge32_leto.sh SRC REPEAT [REPEAT ...]
set -u -o pipefail
FM=$1; shift
source /mnt/homes/zwingel/fm-ada-capacity/jobs2/common.sh
cd "$FM" && export PYTHONPATH=$FM
note "edge32 start fastmace=$(cat .source_commit) repeats=$*"
S=tools/bench/edge32_probe.py
for REPEAT in "$@"; do
  quiet=0; last=""
  while [ $quiet -lt 4 ]; do
    if why=$(others); then quiet=0; [ "${why//[0-9]/}" != "$last" ] && note "waiting: $why"; last=${why//[0-9]/}; else quiet=$((quiet + 1)); fi
    sleep 30
  done
  O=$ROOT/$REPEAT; mkdir -p "$O"
  {
    if [ "${ONLY_REPRODUCIBLE:-0}" != 1 ]; then
    python "$S" run "$O" --side off --repeat "$REPEAT"; echo "off exit=$?"
    python "$S" run "$O" --side on --repeat "$REPEAT"; echo "on exit=$?"
    python "$S" run "$O" --side off --repeat "$REPEAT" --tag 2; echo "off2 exit=$?"
    python "$S" compare "$O/edge32-off.pt" "$O/edge32-on.pt"; echo "compare exit=$?"
    python "$S" compare "$O/edge32-off.pt" "$O/edge32-off2.pt"; echo "control exit=$?"
    fi
    python "$S" run "$O" --side off --repeat "$REPEAT" --reproducible --model mh1 --tag r; echo "off r exit=$?"
    python "$S" run "$O" --side on --repeat "$REPEAT" --reproducible --model mh1 --tag r; echo "on r exit=$?"
    python "$S" run "$O" --side off --repeat "$REPEAT" --reproducible --model mh1 --tag r2; echo "off r2 exit=$?"
    python "$S" compare "$O/edge32-offr.pt" "$O/edge32-onr.pt"; echo "reproducible compare exit=$?"
    python "$S" compare "$O/edge32-offr.pt" "$O/edge32-offr2.pt"; echo "reproducible control exit=$?"
  } > "$O/edge32.log" 2>&1 < /dev/null
  note "end $REPEAT $(grep -E 'exit=|all_bitwise' "$O/edge32.log" | tr '\n' ' ' | cut -c1-400)"
done
note "EDGE32 DONE"
