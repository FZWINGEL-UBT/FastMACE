#!/bin/bash
# tiled job 45 (11 November 2026): TileConfig.compiled_graphs off (a) against on (b), device state, compiled tiles,
# 975,560 SrTiO3 atoms, R 16,384 budget, both orders (the twin of the A100's A/B, Alex alex_cgab.sh).
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
  echo "== cg $order $(date +%T) $(cut -d' ' -f1-3 /proc/loadavg)"
  python tools/bench/tiled_ab.py --repeat 58x58x58 --receivers 8192 --mode compiled \
    --b "config:compiled_graphs=True" --order $order --blocks 3 --calls 2 --warmup 3 \
    --tag=-cg --output $OUT/ab > $OUT/cg-$order.log 2>&1
  echo "exit=$? $(date +%T)"; tail -1 $OUT/cg-$order.log | cut -c1-600
done
echo done
