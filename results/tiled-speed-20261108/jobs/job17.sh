#!/bin/bash
# tiled gap job 17: GPU test, gap switches all off against the defaults (device, wavefront), single-call profiles, whole-system 29k
# the wavefront and with the state on the device (both orders), the after-profiles
set -u
RUN=/home/fzwin/fm-runs/tiled-speed-20261108
SRC=$RUN/src-${COMMIT}
OUT=$RUN/${JOB}
mkdir -p $OUT
export PATH=/home/fzwin/environments/mace_cueq_oeq/bin:/usr/lib/wsl/lib:$PATH PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1
export FASTMACE_MODELS=/home/fzwin/models FASTMACE_SOURCE_COMMIT=$COMMIT
cd $SRC
echo "== gputest $(date +%T)"
python -m pytest -p no:cacheprovider -s -q tests/gpu/test_tiled_speed_cuda.py > $OUT/gputest.log 2>&1
echo "exit=$? $(date +%T)"; tail -3 $OUT/gputest.log
M=fastmace.model.tiled; G=fastmace.rewrites.generic.modules
OFF="$M:TILE_FILTER32=False $M:TILE_EPILOGUE=False $M:TILE_FILL_SKIP=False $M:TILE_PINNED_SUPERSET=False"
ab() { name=$1; shift; for order in ab ba; do
  echo "== $name $order $(date +%T)"
  python tools/bench/tiled_ab.py --repeat 58x58x58 --order $order --blocks 4 --calls 2 --warmup 3 --tag=-$name "$@" --output $OUT/ab > $OUT/ab-$name-$order.log 2>&1
  echo "exit=$? $(date +%T)"; tail -1 $OUT/ab-$name-$order.log | cut -c1-400; done; }
ab wave-gap --host-state --alias-state --wavefront on --a $OFF --b
ab gap --a $OFF --b
ab branch --a $OFF $M:TILE_VIRIAL=False $M:TILE_PREFETCH_DEVICE=False $M:TILE_GRAPH_CACHE=0.0 $G:TILE_BLOCK_ROWS=False --b
B="python tools/bench/tiled_capacity.py --model omat0-medium --structure SrTiO3 --tile-list --species-index --mode auto"
run() { name=$1; shift; echo "== $name $(date +%T)"; $B "$@" --output $OUT/$name > $OUT/$name.log 2>&1; echo "exit=$? $(date +%T)"; }
run dev1m --repeat 58x58x58 --calls 3 --sweep-times
run wave1m --repeat 58x58x58 --calls 3 --sweep-times --host-state --alias-state --wavefront on
run ws29k --repeat 18x18x18 --calls 3 --sweep-times --compare compiled
cp $RUN/kprof4.py $OUT/
echo "== kprof4 $(date +%T)"
python $OUT/kprof4.py 58x58x58 $OUT/kprof4-58.json > $OUT/kprof4.log 2>&1
echo "exit=$? $(date +%T)"; tail -1 $OUT/kprof4.log | cut -c1-300
echo done
