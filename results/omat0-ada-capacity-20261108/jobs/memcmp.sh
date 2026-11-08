#!/usr/bin/env bash
# Peak memory of the new Ada plan against the old one (spline, compose, readout tail, HORNER and OMAT-0 tf32x3 off) at
# the same sizes: capacity_scan.py point, idle check before each.  ROOT=<dir> bash memcmp.sh SRC STRUCTURE:REPEAT...
set -u -o pipefail
FM=$1; shift
source /mnt/homes/zwingel/fm-ada-capacity/jobs2/common.sh
cd "$FM" && export PYTHONPATH=$FM
OLD="$s:SPLINE_W=False $s:R128=False $s:STAGE_EDGES=False $w:HORNER=False $c:COMPOSE_SKIP=False $c:COMPOSE_FAST=False $t:GENERIC_FOLD_READOUT_TAIL=False"
note "memcmp start fastmace=$(cat .source_commit)"
for job in "$@"; do
  structure=${job%%:*} repeat=${job#*:}
  for side in new old; do
    if why=$(others); then note "STOP before $side $job: $why"; exit 3; fi
    if [ $side = old ]; then extra=(--tag old --rewrites tf32x3_gemms=off --constant $OLD); else extra=(--tag new); fi
    python tools/bench/capacity_scan.py point --structure "$structure" --model omat0-medium --precision float32 \
      --output "$ROOT/points" --repeat "$repeat" "${extra[@]}" > "$ROOT/point-$side-$structure-$repeat.log" 2>&1 < /dev/null
    note "end $side $job exit=$? $(tail -n 2 "$ROOT/point-$side-$structure-$repeat.log" | tr '\n' ' ' | cut -c1-300)"
  done
done
note DONE
