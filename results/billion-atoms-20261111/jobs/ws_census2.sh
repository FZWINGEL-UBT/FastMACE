#!/bin/bash
# ws_census.sh SHA [TAG] (census2: --list-reuse off as at 397.5M, a 30M-entry trace): the go/no-go's call (Alex 4985889) at SrTiO3 160^3 (20,480,000 atoms) on this
# workstation's RTX 4000 Ada, with the CUDA allocator's trace in call 0 (the eager first call) and then in call 1
# (the compiled call): the allocations live at each call's peak by source line, and the ledgers.
# run.sh's idle check before each run (exit 3: wait 2 min and retry).
set -u
SHA=$1; TAG=${2:-}
export SRC=/mnt/homes/zwingel/fm-tiled-speed/src-$SHA
LOG=/mnt/homes/zwingel/fm-tiled-speed/ws-census-$(hostname).log
C="tools/bench/tiled_capacity.py --structure SrTiO3 --model omat0-medium --repeat 160x160x160 --mode auto
   --tile-list --host-state --alias-state --species-index --wavefront on --calls 2 --state-dtype auto
   --host-positions --first-call-probe off --ledger --list-reuse off --history-entries 30000000"
for call in 0 1; do
  name=census$TAG-call$call
  for attempt in $(seq 1 180); do
    bash /mnt/homes/zwingel/fm-tiled/run.sh $(hostname)-$SHA-$name $C --call-history $call > /dev/null 2>&1
    code=$?
    [ $code != 3 ] && break
    echo "$(date -Is) $name: not idle, waiting" >> $LOG
    sleep 120
  done
  echo "$(date -Is) end $name exit=$code" >> $LOG
done
echo "$(date -Is) census over" >> $LOG
