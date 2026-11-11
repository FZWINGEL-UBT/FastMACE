#!/usr/bin/env bash
# VIRIAL_ELEMENTWISE on the Ada again, longer (12 blocks of 40 calls per side; the first pair was mixed: 1.0041 / 0.9805),
# on murphy after ada_srows.sh, on src-screen.
B=/mnt/homes/zwingel/fm-ada5060
until grep -q DONE $B/runs/ada-srows/run.log 2>/dev/null; do sleep 60; done
QUIET=3 bash $B/jobs/queue.sh true
export HOST=ada ROOT=$B/runs/ada-virial2
source $B/jobs/common.sh
cd $B/src-screen && export PYTHONPATH=$B/src-screen
for order in ab ba; do
  pattern=$([ $order = ab ] && echo abab || echo abba)
  if why=$(others); then note "STOP before virial $order: $why"; exit 3; fi
  python tools/virial_ab.py --checkpoint "$CK" --structure "$ST@10x10x10" --output $ROOT/virial-$order.json \
    --order $order --pattern $pattern --blocks 12 --calls 40 > $ROOT/virial-$order.log 2>&1 < /dev/null
  note "end virial $order exit=$? $(python3 -c "import json; d = json.load(open('$ROOT/virial-$order.json')); print(d['a_ms'], d['b_ms'], d['b_over_a'])" 2>&1)"
done
note DONE
