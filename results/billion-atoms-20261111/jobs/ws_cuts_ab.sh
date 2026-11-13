#!/bin/bash
# ws_cuts_ab.sh SHA_A SHA_B: the resident cuts against main, on this workstation's RTX 4000 Ada. Tree A: main's
# forces.py / skin.py / tile_list.py under the branch's tools (--digest); tree B: the branch. The go/no-go's call
# (two calls, --ledger --digest) at SrTiO3 160^3 (perfect crystal) and 80^3 (rattle 0.05, strain 0.01), A then B
# then A then B: the digests must be equal, the ledgers show the bytes. run.sh's idle check before each run.
set -u
A=$1; B=$2
LOG=/mnt/homes/zwingel/fm-tiled-speed/ws-cuts-ab-$(hostname).log
C="tools/bench/tiled_capacity.py --structure SrTiO3 --model omat0-medium --mode auto --tile-list --host-state
   --alias-state --species-index --wavefront on --calls 2 --state-dtype auto --host-positions --first-call-probe off
   --ledger --digest"
run() { name=$1; tree=$2; shift 2
  for attempt in $(seq 1 180); do
    SRC=/mnt/homes/zwingel/fm-tiled-speed/src-$tree bash /mnt/homes/zwingel/fm-tiled/run.sh \
      $(hostname)-cuts-$name "$@" > /dev/null 2>&1
    code=$?
    [ $code != 3 ] && break
    echo "$(date -Is) $name: not idle, waiting" >> $LOG
    sleep 120
  done
  echo "$(date -Is) end $name exit=$code" >> $LOG
}
for rep in 1 2; do
  run 80-a$rep $A $C --repeat 80x80x80 --rattle 0.05 --strain 0.01
  run 80-b$rep $B $C --repeat 80x80x80 --rattle 0.05 --strain 0.01
done
run 160-a $A $C --repeat 160x160x160
run 160-b $B $C --repeat 160x160x160
echo "$(date -Is) cuts-ab over" >> $LOG
