#!/bin/bash
# tiled-speed job 3: smoke + exactness at 29k, then per-switch A/Bs at 975,560 atoms (device state), both orders
set -u
RUN=/home/fzwin/fm-runs/tiled-speed-20261108
SRC=$RUN/src-${COMMIT}
OUT=$RUN/${JOB}
mkdir -p $OUT
export PATH=/home/fzwin/environments/mace_cueq_oeq/bin:$PATH PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1
export FASTMACE_MODELS=/home/fzwin/models FASTMACE_SOURCE_COMMIT=$COMMIT
cd $SRC
B="python tools/bench/tiled_capacity.py --model omat0-medium --structure SrTiO3 --tile-list --species-index --mode auto"
echo "== ws29k $(date +%T)"
$B --repeat 18x18x18 --calls 3 --sweep-times --compare compiled --output $OUT/ws29k > $OUT/ws29k.log 2>&1
echo "exit=$? $(date +%T)"; tail -c 1500 $OUT/ws29k.log | grep -o '"reference".*' | cut -c1-600
grep -q '"phase": "done"' $OUT/ws29k/*.json || { echo "smoke failed"; exit 1; }
M=fastmace.model.tiled; G=fastmace.rewrites.generic.modules
ab() { name=$1; shift; for order in ab ba; do
  echo "== $name $order $(date +%T)"
  python tools/bench/tiled_ab.py --repeat ${REP:-58x58x58} --order $order --blocks 3 --calls 2 --warmup 3 --tag=-$name "$@" --output $OUT/ab > $OUT/ab-$name-$order.log 2>&1
  echo "exit=$? $(date +%T)"; tail -1 $OUT/ab-$name-$order.log | cut -c1-700; done; }
ab prefetch --a $M:TILE_PREFETCH_DEVICE=False --b $M:TILE_PREFETCH_DEVICE=True
ab blockrows --a $G:TILE_BLOCK_ROWS=False --b $G:TILE_BLOCK_ROWS=True
ab virial --a $M:TILE_VIRIAL=False --b $M:TILE_VIRIAL=True
ab all --a $M:TILE_PREFETCH_DEVICE=False $G:TILE_BLOCK_ROWS=False $M:TILE_VIRIAL=False --b $M:TILE_PREFETCH_DEVICE=True $G:TILE_BLOCK_ROWS=True $M:TILE_VIRIAL=True
echo done
