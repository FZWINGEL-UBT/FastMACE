#!/bin/bash
# tiled job 41 (11 November 2026, idea 3): the wavefront's row pool on the device. 975,560 SrTiO3 atoms, R 16,384
# budget, compiled tiles. A/B 1: host wavefront (aliased host state, a) against the device wavefront (b). A/B 2: the
# plain device state (a) against the device wavefront (b). Both orders each.
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
  echo "== host-vs-device $order $(date +%T) $(cut -d' ' -f1-3 /proc/loadavg)"
  python tools/bench/tiled_ab.py --repeat 58x58x58 --receivers 8192 --host-state --alias-state --wavefront on \
    --b "config:state='device'" --order $order --blocks 3 --calls 2 --warmup 3 --tag=-hostdev \
    --output $OUT/ab > $OUT/hostdev-$order.log 2>&1
  echo "exit=$? $(date +%T)"; tail -1 $OUT/hostdev-$order.log | cut -c1-600
done
for order in ab ba; do
  echo "== plain-vs-pool $order $(date +%T) $(cut -d' ' -f1-3 /proc/loadavg)"
  python tools/bench/tiled_ab.py --repeat 58x58x58 --receivers 8192 \
    --b "config:alias_state=True" "config:wavefront='on'" --order $order --blocks 3 --calls 2 --warmup 3 \
    --tag=-devpool --output $OUT/ab > $OUT/devpool-$order.log 2>&1
  echo "exit=$? $(date +%T)"; tail -1 $OUT/devpool-$order.log | cut -c1-600
done
echo done
