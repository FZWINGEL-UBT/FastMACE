#!/bin/bash
# tiled gap job 12: device time by labelled region (kprof3.py), and the host-state A/B (all switches off against the
# defaults, both orders; job 8 and 11's single host-state calls were slowed by other processes' CPU load)
set -u
RUN=/home/fzwin/fm-runs/tiled-speed-20261108
SRC=$RUN/src-${COMMIT}
OUT=$RUN/${JOB}
mkdir -p $OUT
export PATH=/home/fzwin/environments/mace_cueq_oeq/bin:/usr/lib/wsl/lib:$PATH PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1
export FASTMACE_MODELS=/home/fzwin/models FASTMACE_SOURCE_COMMIT=$COMMIT
cd $SRC
cp $RUN/kprof3.py $OUT/
echo "== kprof3 $(date +%T) $(cut -d' ' -f1-3 /proc/loadavg)"
python $OUT/kprof3.py 58x58x58 $OUT/kprof3-58.json > $OUT/kprof3.log 2>&1
echo "exit=$? $(date +%T)"; tail -1 $OUT/kprof3.log | cut -c1-300
M=fastmace.model.tiled; G=fastmace.rewrites.generic.modules
OFF="$M:TILE_VIRIAL=False $M:TILE_PREFETCH_DEVICE=False $M:TILE_GRAPH_CACHE=0.0 $G:TILE_BLOCK_ROWS=False"
for order in ab ba; do
  echo "== host-all $order $(date +%T) $(cut -d' ' -f1-3 /proc/loadavg)"
  python tools/bench/tiled_ab.py --repeat 58x58x58 --order $order --blocks 4 --calls 2 --warmup 3 --tag=-host-all --host-state --alias-state --wavefront off --a $OFF --b --output $OUT/ab > $OUT/ab-host-all-$order.log 2>&1
  echo "exit=$? $(date +%T) $(cut -d' ' -f1-3 /proc/loadavg)"; tail -1 $OUT/ab-host-all-$order.log | cut -c1-400
done
echo done
