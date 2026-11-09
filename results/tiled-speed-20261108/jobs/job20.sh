#!/bin/bash
# tiled gap job 20: GPU tests, TILE_GRAPH_CACHE_WAVE A/B (wavefront, both orders), wavefront profile
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
ab wave-gcache --host-state --alias-state --wavefront on --a $M:TILE_GRAPH_CACHE_WAVE=False --b $M:TILE_GRAPH_CACHE_WAVE=True
cp /mnt/c/Users/fzwin/AppData/Local/Temp/claude/06a4a8e8fc832e51/1a150e36-92ea-4347-8da9-78c469ccacf9/scratchpad/jobs/wprof.py $OUT/
echo "== wprof $(date +%T)"
python $OUT/wprof.py 58x58x58 $OUT/wprof-58.json > $OUT/wprof.log 2>&1
echo "exit=$? $(date +%T)"; tail -1 $OUT/wprof.log | cut -c1-1500
echo done
