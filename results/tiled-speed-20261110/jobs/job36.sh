#!/bin/bash
# tiled job 36: job 35 stopped at 09:48:59 (systemd stop, not ours); its rest: wave-copy ba, host-copy both orders, wprof
set -u
RUN=/home/fzwin/fm-runs/tiled-speed-20261108
SRC=$RUN/src-${COMMIT}
OUT=$RUN/${JOB}
mkdir -p $OUT
export PATH=/home/fzwin/environments/mace_cueq_oeq/bin:/usr/lib/wsl/lib:$PATH PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1
export FASTMACE_MODELS=/home/fzwin/models FASTMACE_SOURCE_COMMIT=$COMMIT
cd $SRC

: # GPU test passed in job 35

M=fastmace.model.tiled
ab() { name=$1; shift; for order in ${ORDERS:-ab ba}; do
  echo "== $name $order $(date +%T) $(cut -d' ' -f1-3 /proc/loadavg)"
  python tools/bench/tiled_ab.py --repeat 58x58x58 --order $order --blocks 4 --calls 2 --warmup 3 --tag=-$name "$@" --output $OUT/ab > $OUT/ab-$name-$order.log 2>&1
  echo "exit=$? $(date +%T)"; tail -1 $OUT/ab-$name-$order.log | cut -c1-400; done; }
ORDERS=ba ab wave-copy --host-state --alias-state --wavefront on --a $M:TILE_COPY_STREAM=False --b $M:TILE_COPY_STREAM=True
ab host-copy --host-state --a $M:TILE_COPY_STREAM=False --b $M:TILE_COPY_STREAM=True
cp $SRC/results/tiled-speed-20261108/jobs/wprof.py $OUT/
echo "== wprof $(date +%T)"
python $OUT/wprof.py 58x58x58 $OUT/wprof-58.json > $OUT/wprof.log 2>&1
echo "exit=$? $(date +%T)"; tail -1 $OUT/wprof.log | cut -c1-1500
echo done
