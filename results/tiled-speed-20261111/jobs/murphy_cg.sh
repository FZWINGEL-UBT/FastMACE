#!/bin/bash
# murphy_cg.sh SHA: compiled_graphs on the RTX 4000 Ada (11 November 2026): probe_cg.py (the guard's verdict, the real
# capture pools under a 0.9 memory fraction, ABBA on one field) at SrTiO3 40^3 and 58^3, R 8,192. Each step waits for
# run.sh's idle check (and no claims / none of our jobs), retrying every 2 min for up to 12 h.
set -u
SHA=$1
export SRC=/mnt/homes/zwingel/fm-tiled-speed/src-$SHA
LOG=/mnt/homes/zwingel/fm-tiled-speed/murphy-cg-$SHA.log
run() { name=$1; shift
  for attempt in $(seq 1 360); do
    ours=$(pgrep -u zwingel -fc '[f]m-ada-capacity|[c]apacity_scan|[p]ytest|[t]iled_capacity|[t]iled_ab|[p]robe_cg')
    claims=$(ls -d /mnt/homes/zwingel/fm-ada-capacity/runs/ada-claim /mnt/homes/zwingel/fm-ada-capacity/runs/ada-claim2 2>/dev/null | wc -l)
    if [ "$ours" = 0 ] && [ "$claims" = 0 ]; then
      echo "$(date -Is) start $name" >> $LOG
      bash /mnt/homes/zwingel/fm-tiled/run.sh murphy-cg-$SHA-$name probe_cg.py "$@" > /dev/null 2>&1
      code=$?
      if [ $code != 3 ]; then
        echo "$(date -Is) end $name exit=$code $(grep -v '^exit' /mnt/homes/zwingel/fm-tiled/runs/murphy-cg-$SHA-$name/run.log | tail -1 | cut -c1-600)" >> $LOG
        return
      fi
      echo "$(date -Is) $name: not idle, waiting" >> $LOG
    fi
    sleep 120
  done
  echo "$(date -Is) $name: gave up waiting" >> $LOG
}
run r40 --repeat 40x40x40 --receivers 8192 --blocks 2
run r58 --repeat 58x58x58 --receivers 8192 --blocks 3
echo "$(date -Is) cg over" >> $LOG
