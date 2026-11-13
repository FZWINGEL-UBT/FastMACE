#!/bin/bash
# ws_hostrows_ab.sh SHA: same-process A/B of TILE_HOST_ROWS_KEPT (a: False, the device float64 copy as on main; b:
# True, host rows) on this workstation's RTX 4000 Ada: SrTiO3 160^3, --host-positions (both sides), list reuse on
# (the kept lists reused from the second call), host state, aliased, wavefront on, mode auto (compiled calls after
# the warm-up). Both orders, one process each. run.sh's idle check (GPU) before each.
set -u
SHA=$1
export SRC=/mnt/homes/zwingel/fm-tiled-speed/src-$SHA
LOG=/mnt/homes/zwingel/fm-tiled-speed/ws-hostrows-ab-$(hostname).log
M=fastmace.model.tiled
C="tools/bench/tiled_ab.py --structure SrTiO3 --model omat0-medium --repeat 160x160x160 --mode auto --host-state
   --alias-state --wavefront on --host-positions --a $M:TILE_HOST_ROWS_KEPT=False --b $M:TILE_HOST_ROWS_KEPT=True
   --warmup 3 --blocks 3 --calls 2"
for order in ab ba; do
  for attempt in $(seq 1 180); do
    bash /mnt/homes/zwingel/fm-tiled/run.sh $(hostname)-$SHA-hostrows-$order $C --order $order > /dev/null 2>&1
    code=$?
    [ $code != 3 ] && break
    echo "$(date -Is) $order: not idle, waiting" >> $LOG
    sleep 120
  done
  echo "$(date -Is) end $order exit=$code $(grep -v '^exit' /mnt/homes/zwingel/fm-tiled/runs/$(hostname)-$SHA-hostrows-$order/run.log | tail -1 | cut -c1-400)" >> $LOG
done
echo "$(date -Is) hostrows-ab over" >> $LOG
