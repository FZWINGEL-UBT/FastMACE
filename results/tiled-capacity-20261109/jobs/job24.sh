#!/bin/bash
# tiled capacity job 24 (201233ef): TileConfig.state_dtype. Gates (host state, wavefront, compiled R 48: float32 store,
# float16, bfloat16), wavefront A/Bs at 975,560 SrTiO3 (float32 store against float16 and bfloat16, both orders:
# speed and force error), host ledgers (float32 and float16 store)
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
ab() { name=$1; shift; for order in ab ba; do
  echo "== $name $order $(date +%T) $(cut -d' ' -f1-3 /proc/loadavg)"
  python tools/bench/tiled_ab.py --repeat 58x58x58 --order $order --blocks 4 --calls 2 --warmup 3 --tag=-$name "$@" --output $OUT/ab > $OUT/ab-$name-$order.log 2>&1
  echo "exit=$? $(date +%T)"; tail -1 $OUT/ab-$name-$order.log | cut -c1-900; done; }
ab wave-f16 --host-state --alias-state --wavefront on --a --b "config:state_dtype='float16'"
ab wave-bf16 --host-state --alias-state --wavefront on --a --b "config:state_dtype='bfloat16'"
led() { name=$1; shift
  echo "== ledger-$name $(date +%T)"
  python tools/bench/tiled_capacity.py --structure SrTiO3 --model omat0-medium --repeat 58x58x58 --mode auto \
    --tile-list --host-state --alias-state --species-index --wavefront on --calls 2 --ledger --tag=-$name "$@" \
    --output $OUT/ledger-$name > $OUT/ledger-$name.log 2>&1
  echo "exit=$? $(date +%T)"; tail -2 $OUT/ledger-$name.log | cut -c1-600; }
led f32
led f16 --state-dtype float16
echo done
