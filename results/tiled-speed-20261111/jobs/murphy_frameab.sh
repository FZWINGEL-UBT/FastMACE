#!/bin/bash
# murphy_frameab.sh SHA: idea 11 on the Ada (11 November 2026): TILE_LOCAL_FRAME off (a) against on (b), same
# process, both orders, SrTiO3 40^3 (320,000 atoms), R 8,192 (budget), compiled, device state. run.sh's idle check
# before each order (waits while our pytest suite runs). (Its first run, with the template's run line left in,
# measured compiled_graphs at 58^3 instead: runs murphy-frameab-SHA-ab/ba, kept as an accuracy point.)
set -u
SHA=$1
export SRC=/mnt/homes/zwingel/fm-tiled-speed/src-$SHA
LOG=/mnt/homes/zwingel/fm-tiled-speed/murphy-frameab2-$SHA.log
run() { name=$1; shift
  for attempt in $(seq 1 360); do
    ours=$(pgrep -u zwingel -fc '[f]m-ada-capacity|[c]apacity_scan|[p]ytest|[t]iled_capacity|[t]iled_ab\.py|[p]robe_cg\.py')
    claims=$(ls -d /mnt/homes/zwingel/fm-ada-capacity/runs/ada-claim /mnt/homes/zwingel/fm-ada-capacity/runs/ada-claim2 2>/dev/null | wc -l)
    if [ "$ours" = 0 ] && [ "$claims" = 0 ]; then
      echo "$(date -Is) start $name" >> $LOG
      bash /mnt/homes/zwingel/fm-tiled/run.sh murphy-frameab2-$SHA-$name tools/bench/tiled_ab.py "$@" > /dev/null 2>&1
      code=$?
      if [ $code != 3 ]; then
        echo "$(date -Is) end $name exit=$code $(grep -v '^exit' /mnt/homes/zwingel/fm-tiled/runs/murphy-frameab2-$SHA-$name/run.log | tail -1 | cut -c1-600)" >> $LOG
        return
      fi
      echo "$(date -Is) $name: not idle, waiting" >> $LOG
    fi
    sleep 120
  done
  echo "$(date -Is) $name: gave up waiting" >> $LOG
}
for order in ab ba; do
  run $order --repeat 40x40x40 --receivers 8192 --mode compiled --a "fastmace.model.tiled:TILE_LOCAL_FRAME=False" \
    --b "fastmace.model.tiled:TILE_LOCAL_FRAME=True" --order $order --blocks 4 --calls 2 --warmup 3 --tag=-frame40-$order
done
echo "$(date -Is) frameab over" >> $LOG
