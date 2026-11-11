#!/usr/bin/env bash
# murphy, after murphy-X34: the POLAR-1 megakernel gate (S, M at 3x3x3; L at 2x2x2, its float64 reference does not fit
# 20 GB at 3x3x3, as polar-reuse), NaCl and NbCl5, MEGAKERNEL forced on.
B=/mnt/homes/zwingel/fm-ada5060
until grep -qE "DONE|STOP" $B/runs/murphy-X34/run.log 2>/dev/null; do sleep 120; done
MK=fastmace.longrange.polar:MEGAKERNEL=True
for spec in "S 3" "M 3" "L 2"; do set -- $spec
  bash $B/jobs/queue.sh env HOST=ada ROOT=$B/runs/murphy-polar-gate bash $B/jobs/gate.sh $B/src-7d18028d mk-$1 polar1-$(echo $1 | tr SML sml) MACE-POLAR-1-$1.model "NaCl NbCl5" "$2 $2 $2" $MK
done
