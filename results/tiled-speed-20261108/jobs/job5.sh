#!/bin/bash
# tiled-speed job 5: GPU test, force/stress gate (tiled compiled, the switches' defaults), wavefront A/B of all switches
set -u
RUN=/home/fzwin/fm-runs/tiled-speed-20261108
SRC=$RUN/src-${COMMIT}
OUT=$RUN/${JOB}
mkdir -p $OUT
export PATH=/home/fzwin/environments/mace_cueq_oeq/bin:$PATH PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1
export FASTMACE_MODELS=/home/fzwin/models FASTMACE_SOURCE_COMMIT=$COMMIT
cd $SRC
echo "== gputest $(date +%T)"
python -m pytest -p no:cacheprovider -s -q tests/gpu/test_tiled_speed_cuda.py > $OUT/gputest.log 2>&1
echo "exit=$? $(date +%T)"; tail -3 $OUT/gputest.log
echo "== gate $(date +%T)"
python tools/force_stress_gate.py --output $OUT/gate --engine fastmace --model omat0-medium \
  --checkpoint $FASTMACE_MODELS/mace-omat-0-medium.model --structures-root /home/fzwin/fastmace-baseline/1a-T18 \
  --structures results/loup-a100-20261103/SrTiO3.cif@3x3x3 NaCl NbCl5 --tiled-receivers 48 --tiled-mode compiled \
  > $OUT/gate.log 2>&1
echo "exit=$? $(date +%T)"; tail -25 $OUT/gate.log
M=fastmace.model.tiled; G=fastmace.rewrites.generic.modules
OFF="$M:TILE_VIRIAL=False $M:TILE_PREFETCH_DEVICE=False $M:TILE_GRAPH_CACHE=0.0 $G:TILE_BLOCK_ROWS=False"
ON="$M:TILE_VIRIAL=True $M:TILE_PREFETCH_DEVICE=True $M:TILE_GRAPH_CACHE=0.25 $G:TILE_BLOCK_ROWS=True"
for order in ab ba; do
  echo "== wave-all $order $(date +%T)"
  python tools/bench/tiled_ab.py --repeat 58x58x58 --order $order --blocks 3 --calls 2 --warmup 3 --tag=-wave-all \
    --host-state --alias-state --wavefront on --a $OFF --b $ON --output $OUT/ab > $OUT/ab-wave-all-$order.log 2>&1
  echo "exit=$? $(date +%T)"; tail -1 $OUT/ab-wave-all-$order.log | cut -c1-500
done
echo done
