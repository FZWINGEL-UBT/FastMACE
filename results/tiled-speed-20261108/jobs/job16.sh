#!/bin/bash
# tiled gap job 16: GPU tests, TILE_PIECE_START A/B (device, both orders), region profile at the defaults
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
M=fastmace.model.tiled
ab() { name=$1; shift; for order in ab ba; do
  echo "== $name $order $(date +%T) $(cut -d' ' -f1-3 /proc/loadavg)"
  python tools/bench/tiled_ab.py --repeat 58x58x58 --order $order --blocks 4 --calls 2 --warmup 3 --tag=-$name "$@" --output $OUT/ab > $OUT/ab-$name-$order.log 2>&1
  echo "exit=$? $(date +%T)"; tail -1 $OUT/ab-$name-$order.log | cut -c1-400; done; }
ab piecestart --a $M:TILE_PIECE_START=False --b $M:TILE_PIECE_START=True
cp $RUN/kprof4.py $OUT/
echo "== kprof4 $(date +%T)"
python $OUT/kprof4.py 58x58x58 $OUT/kprof4-58.json > $OUT/kprof4.log 2>&1
echo "exit=$? $(date +%T)"; tail -1 $OUT/kprof4.log | cut -c1-300
echo done
