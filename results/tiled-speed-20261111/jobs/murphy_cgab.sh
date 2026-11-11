#!/bin/bash
# murphy_cgab.sh SHA: the conductor's Ada A/B (11 November 2026): TileConfig.compiled_graphs off (a) against on with
# the memory guard (b), same process, both orders, SrTiO3 58^3 (975,560 atoms), R 8,192 (budget), compiled, device
# state. Waits for murphy_cg.sh's probes ("cg over"), then run.sh's idle check before each order.
set -u
SHA=$1
export SRC=/mnt/homes/zwingel/fm-tiled-speed/src-$SHA
LOG=/mnt/homes/zwingel/fm-tiled-speed/murphy-cgab-$SHA.log
until grep -q "cg over" /mnt/homes/zwingel/fm-tiled-speed/murphy-cg-$SHA.log 2>/dev/null; do sleep 120; done
run() { name=$1; shift
  for attempt in $(seq 1 360); do
    ours=$(pgrep -u zwingel -fc '[f]m-ada-capacity|[c]apacity_scan|[p]ytest|[t]iled_capacity|[t]iled_ab\.py|[p]robe_cg\.py')
    claims=$(ls -d /mnt/homes/zwingel/fm-ada-capacity/runs/ada-claim /mnt/homes/zwingel/fm-ada-capacity/runs/ada-claim2 2>/dev/null | wc -l)
    if [ "$ours" = 0 ] && [ "$claims" = 0 ]; then
      echo "$(date -Is) start $name" >> $LOG
      bash /mnt/homes/zwingel/fm-tiled/run.sh murphy-cgab-$SHA-$name tools/bench/tiled_ab.py "$@" > /dev/null 2>&1
      code=$?
      if [ $code != 3 ]; then
        echo "$(date -Is) end $name exit=$code $(grep -v '^exit' /mnt/homes/zwingel/fm-tiled/runs/murphy-cgab-$SHA-$name/run.log | tail -1 | cut -c1-600)" >> $LOG
        return
      fi
      echo "$(date -Is) $name: not idle, waiting" >> $LOG
    fi
    sleep 120
  done
  echo "$(date -Is) $name: gave up waiting" >> $LOG
}
for order in ab ba; do
  run $order --repeat 58x58x58 --receivers 8192 --mode compiled --a "config:compiled_graphs=False" \
    --b "config:compiled_graphs=True" --order $order --blocks 4 --calls 2 --warmup 3 --tag=-cg
done
echo "$(date -Is) cgab over" >> $LOG
