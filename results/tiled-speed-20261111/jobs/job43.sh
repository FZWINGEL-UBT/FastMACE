#!/bin/bash
# tiled job 43 (11 November 2026, idea 3 follow-up): the aliased state on the device without the wavefront (x and sc
# only, dE/dx in a planned window: half the device state) against the plain device state. 975,560 SrTiO3 atoms,
# compiled tiles, R 16,384 budget; both orders.
set -u
RUN=/home/fzwin/fm-runs/tiled-speed-20261108
SRC=$RUN/src-${COMMIT}
OUT=$RUN/${JOB}
mkdir -p $OUT
export PATH=/home/fzwin/environments/mace_cueq_oeq/bin:/usr/lib/wsl/lib:$PATH PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1
export FASTMACE_MODELS=/home/fzwin/models FASTMACE_SOURCE_COMMIT=$COMMIT
unset PYTORCH_CUDA_ALLOC_CONF
cd $SRC
for order in ab ba; do
  echo "== plain-vs-alias $order $(date +%T) $(cut -d' ' -f1-3 /proc/loadavg)"
  python tools/bench/tiled_ab.py --repeat 58x58x58 --receivers 8192 --wavefront off \
    --b "config:alias_state=True" --order $order --blocks 3 --calls 2 --warmup 3 \
    --tag=-devalias --output $OUT/ab > $OUT/devalias-$order.log 2>&1
  echo "exit=$? $(date +%T)"; tail -1 $OUT/devalias-$order.log | cut -c1-600
done
echo done
