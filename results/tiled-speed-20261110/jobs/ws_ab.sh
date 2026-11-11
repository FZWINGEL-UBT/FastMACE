#!/bin/bash
# ws_ab.sh SHA: the tiled speed switches on this workstation's RTX 4000 Ada (TILED_SPEED_GPUS: the RTX 5060 Ti only):
# as they are here (off) against all on, same process, both orders; device state and the wavefront; SrTiO3 58^3.
# Waits for this host's capacity scan to end ("scan over" in ws-scan-HOST.log), then before each A/B none of our
# other jobs here and run.sh's idle check (no compute process on the GPU, no other user's busy process).
set -u
SHA=$1
export SRC=/mnt/homes/zwingel/fm-tiled-speed/src-$SHA
LOG=/mnt/homes/zwingel/fm-tiled-speed/ws-ab-$(hostname).log
until grep -q "scan over" /mnt/homes/zwingel/fm-tiled-speed/ws-scan-$(hostname).log 2>/dev/null; do sleep 120; done
M=fastmace.model.tiled
ON="$M:TILE_PREFETCH_DEVICE=True $M:TILE_VIRIAL=True $M:TILE_GRAPH_CACHE=0.25 $M:TILE_FILTER32=True $M:TILE_EPILOGUE=True $M:TILE_PINNED_SUPERSET=True $M:TILE_GRAPH_CACHE_WAVE=True $M:TILE_HOST_WORKER=True $M:TILE_COPY_STREAM=True"
run() { name=$1; shift
  for attempt in $(seq 1 360); do
    ours=$(pgrep -u zwingel -fc '[f]m-ada-capacity|[f]m-memprofile|[c]apacity_scan|[l]aunch_|[p]ytest|[t]iled_capacity|[t]iled_ab')
    claims=$(ls -d /mnt/homes/zwingel/fm-ada-capacity/runs/ada-claim /mnt/homes/zwingel/fm-ada-capacity/runs/ada-claim2 2>/dev/null | wc -l)
    if [ "$ours" = 0 ] && [ "$claims" = 0 ]; then
      echo "$(date -Is) start $name" >> $LOG
      bash /mnt/homes/zwingel/fm-tiled/run.sh $(hostname)-ab-$SHA-$name tools/bench/tiled_ab.py "$@" > /dev/null 2>&1
      code=$?
      if [ $code != 3 ]; then
        echo "$(date -Is) end $name exit=$code $(grep -v '^exit' /mnt/homes/zwingel/fm-tiled/runs/$(hostname)-ab-$SHA-$name/run.log | tail -1 | cut -c1-500)" >> $LOG
        return
      fi
      echo "$(date -Is) $name: not idle, waiting" >> $LOG
    fi
    sleep 120
  done
  echo "$(date -Is) $name: gave up waiting" >> $LOG
}
for order in ab ba; do
  run dev-$order --repeat 58x58x58 --order $order --blocks 4 --calls 2 --warmup 3 --tag=-dev-all --b $ON
  run wave-$order --repeat 58x58x58 --order $order --blocks 4 --calls 2 --warmup 3 --tag=-wave-all --host-state \
    --alias-state --wavefront on --b $ON
done
echo "$(date -Is) ab over" >> $LOG
