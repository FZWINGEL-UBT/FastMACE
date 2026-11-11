#!/bin/bash
# tiled job 37: receiver-cap screen (default allocator): device state, 975,560 SrTiO3 atoms, the cap fixed (no
# budget) at 16,384 / 32,768 / 65,536, then 16,384 again; a = b (no constants), each side's median is the time.
set -u
RUN=/home/fzwin/fm-runs/tiled-speed-20261108
SRC=$RUN/src-${COMMIT}
OUT=$RUN/${JOB}
mkdir -p $OUT
export PATH=/home/fzwin/environments/mace_cueq_oeq/bin:/usr/lib/wsl/lib:$PATH PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1
export FASTMACE_MODELS=/home/fzwin/models FASTMACE_SOURCE_COMMIT=$COMMIT
unset PYTORCH_CUDA_ALLOC_CONF
cd $SRC
for r in 16384 32768 65536 16384; do
  echo "== R$r $(date +%T) $(cut -d' ' -f1-3 /proc/loadavg)"
  python tools/bench/tiled_ab.py --repeat 58x58x58 --order ab --blocks 2 --calls 2 --warmup 3 --no-budget \
    --receivers $r --tag=-cap$r --output $OUT/ab > $OUT/cap$r-$(date +%H%M).log 2>&1
  echo "exit=$? $(date +%T)"; tail -1 $(ls -t $OUT/cap$r-*.log | head -1) | cut -c1-500
done
echo done
