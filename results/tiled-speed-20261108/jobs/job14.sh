#!/bin/bash
# tiled gap job 14: GPU tests, TILE_EPILOGUE A/B (device state and wavefront, both orders), region profile
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
echo "== gate $(date +%T)"
python tools/force_stress_gate.py --output $OUT/gate --engine fastmace --model omat0-medium \
  --checkpoint $FASTMACE_MODELS/mace-omat-0-medium.model --structures-root /home/fzwin/fastmace-baseline/1a-T18 \
  --structures results/loup-a100-20261103/SrTiO3.cif@3x3x3 NaCl NbCl5 --tiled-receivers 48 --tiled-mode compiled \
  > $OUT/gate.log 2>&1
echo "exit=$? $(date +%T)"; tail -4 $OUT/gate.log
M=fastmace.model.tiled
ab() { name=$1; shift; for order in ab ba; do
  echo "== $name $order $(date +%T) $(cut -d' ' -f1-3 /proc/loadavg)"
  python tools/bench/tiled_ab.py --repeat 58x58x58 --order $order --blocks 4 --calls 2 --warmup 3 --tag=-$name "$@" --output $OUT/ab > $OUT/ab-$name-$order.log 2>&1
  echo "exit=$? $(date +%T)"; tail -1 $OUT/ab-$name-$order.log | cut -c1-400; done; }
ab epilogue --a $M:TILE_EPILOGUE=False --b $M:TILE_EPILOGUE=True
ab wave-epilogue --host-state --alias-state --wavefront on --a $M:TILE_EPILOGUE=False --b $M:TILE_EPILOGUE=True
cp $RUN/kprof4.py $OUT/
echo "== kprof4 $(date +%T)"
python $OUT/kprof4.py 58x58x58 $OUT/kprof4-58.json > $OUT/kprof4.log 2>&1
echo "exit=$? $(date +%T)"; tail -1 $OUT/kprof4.log | cut -c1-300
echo done
