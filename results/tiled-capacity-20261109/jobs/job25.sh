#!/bin/bash
# tiled capacity job 25: TileConfig.state_dtype gates (host state, wavefront, compiled R 48: float32 store, float16,
# bfloat16; the wavefront global-list fix) and NVE drift at SrTiO3 20x20x20 (float32, float16, bfloat16 store); wavefront A/B without the kept lists
set -u
RUN=/home/fzwin/fm-runs/tiled-speed-20261108
SRC=$RUN/src-${COMMIT}
OUT=$RUN/${JOB}
mkdir -p $OUT
export PATH=/home/fzwin/environments/mace_cueq_oeq/bin:/usr/lib/wsl/lib:$PATH PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1
export FASTMACE_MODELS=/home/fzwin/models FASTMACE_SOURCE_COMMIT=$COMMIT
cd $SRC
gate() { name=$1; shift
  echo "== gate-$name $(date +%T)"
  python tools/force_stress_gate.py --output $OUT/gate-$name --engine fastmace --model omat0-medium \
    --checkpoint $FASTMACE_MODELS/mace-omat-0-medium.model --structures-root /home/fzwin/fastmace-baseline/1a-T18 \
    --structures results/loup-a100-20261103/SrTiO3.cif@3x3x3 NaCl NbCl5 --tiled-receivers 48 --tiled-mode compiled \
    --tiled-config state=host alias_state=true wavefront=on "$@" > $OUT/gate-$name.log 2>&1
  echo "exit=$? $(date +%T)"; tail -4 $OUT/gate-$name.log; }
gate f32
gate f16 state_dtype=float16
gate bf16 state_dtype=bfloat16
nve() { name=$1; shift
  echo "== nve-$name $(date +%T)"
  python tools/bench/tiled_nve.py --repeat 20x20x20 --receivers 2048 --steps 1000 --tag=-$name "$@" --output $OUT/nve > $OUT/nve-$name.log 2>&1
  echo "exit=$? $(date +%T)"; tail -1 $OUT/nve-$name.log | cut -c1-400; }
nve f32
nve f16 --state-dtype float16
nve bf16 --state-dtype bfloat16
ab() { name=$1; shift; for order in ab ba; do
  echo "== $name $order $(date +%T) $(cut -d' ' -f1-3 /proc/loadavg)"
  python tools/bench/tiled_ab.py --repeat 58x58x58 --order $order --blocks 4 --calls 2 --warmup 3 --tag=-$name "$@" --output $OUT/ab > $OUT/ab-$name-$order.log 2>&1
  echo "exit=$? $(date +%T)"; tail -1 $OUT/ab-$name-$order.log | cut -c1-900; done; }
ab wave-nokept --host-state --alias-state --wavefront on --a --b "config:list_reuse=False"
echo done
