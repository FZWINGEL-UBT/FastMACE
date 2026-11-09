#!/bin/bash
# tiled-speed job 6 (887b6b1d, the graph cache's budget fixed): graph-cache A/B and all-switches A/B at 975,560 atoms
# (both orders), the after-profile, and whole system against tiled at 29,160 atoms
set -u
RUN=/home/fzwin/fm-runs/tiled-speed-20261108
SRC=$RUN/src-${COMMIT}
OUT=$RUN/${JOB}
mkdir -p $OUT
export PATH=/home/fzwin/environments/mace_cueq_oeq/bin:$PATH PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1
export FASTMACE_MODELS=/home/fzwin/models FASTMACE_SOURCE_COMMIT=$COMMIT
cd $SRC
M=fastmace.model.tiled; G=fastmace.rewrites.generic.modules
ab() { name=$1; shift; for order in ab ba; do
  echo "== $name $order $(date +%T)"
  python tools/bench/tiled_ab.py --repeat 58x58x58 --order $order --blocks 3 --calls 2 --warmup 3 --tag=-$name "$@" --output $OUT/ab > $OUT/ab-$name-$order.log 2>&1
  echo "exit=$? $(date +%T)"; tail -1 $OUT/ab-$name-$order.log | cut -c1-500; done; }
ab graphcache --a $M:TILE_GRAPH_CACHE=0.0 --b $M:TILE_GRAPH_CACHE=0.25
ab all --a $M:TILE_VIRIAL=False $M:TILE_PREFETCH_DEVICE=False $M:TILE_GRAPH_CACHE=0.0 $G:TILE_BLOCK_ROWS=False --b
B="python tools/bench/tiled_capacity.py --model omat0-medium --structure SrTiO3 --tile-list --species-index --mode auto"
run() { name=$1; shift; echo "== $name $(date +%T)"; $B "$@" --output $OUT/$name > $OUT/$name.log 2>&1; echo "exit=$? $(date +%T)"; }
run dev1m --repeat 58x58x58 --calls 3 --sweep-times
run ws29k --repeat 18x18x18 --calls 3 --sweep-times --compare compiled
echo done
