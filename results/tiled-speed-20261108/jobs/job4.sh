#!/bin/bash
# tiled-speed job 4: TILE_GRAPH_CACHE A/B (both orders) at 975,560 atoms, then the after-profiles
set -u
RUN=/home/fzwin/fm-runs/tiled-speed-20261108
SRC=$RUN/src-${COMMIT}
OUT=$RUN/${JOB}
mkdir -p $OUT
export PATH=/home/fzwin/environments/mace_cueq_oeq/bin:$PATH PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1
export FASTMACE_MODELS=/home/fzwin/models FASTMACE_SOURCE_COMMIT=$COMMIT
cd $SRC
M=fastmace.model.tiled
ab() { name=$1; shift; for order in ab ba; do
  echo "== $name $order $(date +%T)"
  python tools/bench/tiled_ab.py --repeat ${REP:-58x58x58} --order $order --blocks 3 --calls 2 --warmup 3 --tag=-$name "$@" --output $OUT/ab > $OUT/ab-$name-$order.log 2>&1
  echo "exit=$? $(date +%T)"; tail -1 $OUT/ab-$name-$order.log | cut -c1-500; done; }
ab graphcache --a $M:TILE_GRAPH_CACHE=0.0 --b $M:TILE_GRAPH_CACHE=0.25
B="python tools/bench/tiled_capacity.py --model omat0-medium --structure SrTiO3 --tile-list --species-index --mode auto"
run() { name=$1; shift; echo "== $name $(date +%T)"; $B "$@" --output $OUT/$name > $OUT/$name.log 2>&1; echo "exit=$? $(date +%T)"; }
run dev1m --repeat 58x58x58 --calls 3 --sweep-times
run host1m --repeat 58x58x58 --calls 3 --sweep-times --host-state --alias-state --wavefront off
run wave1m --repeat 58x58x58 --calls 3 --sweep-times --host-state --alias-state --wavefront on
run prof1m --repeat 58x58x58 --calls 3 --sweep-profile
cp $RUN/kprof.py $OUT/
python $OUT/kprof.py tiled 58x58x58 $OUT/kprof-tiled-58x58x58.json > $OUT/kprof-tiled.log 2>&1; tail -1 $OUT/kprof-tiled.log
echo done
