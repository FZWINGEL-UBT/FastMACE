#!/usr/bin/env bash
# main 4e9388c8's switches on the Ada (conductor, 11 November 2026), on murphy after its memcheck2.sh, one job at a time,
# on src-screen (the screen branch merged with main and the registry fix):
# 1. ab3.sh A5 (GEOM; PACKED8; both) and A6 (SC_FACTORED on top of SC_GENERATED), both orders;
# 2. tools/virial_ab.py (VIRIAL_ELEMENTWISE, force call with stress, SrTiO3 10x10x10), ab / abab and ba / abba;
# 3. the force/stress gate of VIRIAL_ELEMENTWISE (NaCl, NbCl5 3x3x3).
B=/mnt/homes/zwingel/fm-ada5060
FM=$B/src-screen
while pgrep -f "[m]emcheck2.sh|[r]epro.sh" > /dev/null; do sleep 60; done
QUIET=3 bash $B/jobs/queue.sh true
export HOST=ada ROOT=$B/runs/ada-new
mkdir -p $ROOT
bash $B/jobs/ab3.sh $FM A5 A6
source $B/jobs/common.sh
cd $FM && export PYTHONPATH=$FM
for order in ab ba; do
  pattern=$([ $order = ab ] && echo abab || echo abba)
  if why=$(others); then note "STOP before virial $order: $why"; exit 3; fi
  python tools/virial_ab.py --checkpoint "$CK" --structure "$ST@10x10x10" --output $ROOT/virial-$order.json \
    --order $order --pattern $pattern > $ROOT/virial-$order.log 2>&1 < /dev/null
  note "end virial $order exit=$? $(tail -1 $ROOT/virial-$order.log | cut -c1-400)"
done
bash $B/jobs/gate.sh $FM virial omat0-medium mace-omat-0-medium.model "NaCl NbCl5" "3 3 3" \
  fastmace.rewrites.engine:VIRIAL_ELEMENTWISE=True
note DONE-new
