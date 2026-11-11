#!/bin/bash -l
# Billion-atom step 1, the conductor's not-slower A/B (decision 66) of the two default cuts (11 November 2026): side a
# the species maps in int32 and the whole-system BLOCK_ORDER order bound (TILE_WHOLE_BLOCK_ORDER=True), side b the
# defaults (uint8 maps, no order); each side its own TiledForceField, same process, both orders. SrTiO3 58^3
# (975,560 atoms), R 16,384 fixed: the default tiled call (device state, wavefront auto) and the capacity
# record's (aliased host state, wavefront on). One A100-SXM4-80GB.
# Usage: sbatch alex_defaults.sh SHA
#SBATCH --job-name=fm-defab
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=01:30:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-tiledgate/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-tiledgate
SHA=${1:?SHA}
OUT=$B/runs/defab-$SLURM_JOB_ID
mkdir -p $OUT
log=$OUT/run.log
note() { echo "$(date +%FT%T%z) $*" | tee -a "$log"; }
SRC=$TMPDIR/src
mkdir -p $SRC && tar -xf $B/src-$SHA.tar -C $SRC
env_dir=/home/atuin/v130cb/v130cb18/fm-a100omat/env
site=$(echo "$env_dir"/lib/python3.*/site-packages)
export PATH=$env_dir/bin:/usr/bin:/bin VIRTUAL_ENV=$env_dir PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib
export FASTMACE_MODELS=/home/atuin/v130cb/v130cb18/fm-a100omat/models FASTMACE_SOURCE_COMMIT=$SHA
export FASTMACE_CACHE=$TMPDIR/cache TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
export XDG_CACHE_HOME=$TMPDIR/xdg
n=$(nproc); export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n PYTHONUNBUFFERED=1
cd $SRC || exit 4
note "job $SLURM_JOB_ID on $(hostname); $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1); source $SHA; nproc $n"
for cfg in default capacity; do
  [ $cfg = default ] && X="" || X="--host-state --alias-state --wavefront on"
  for order in ab ba; do
    note "start $cfg $order"
    python tools/bench/tiled_ab.py --repeat 58x58x58 --receivers 16384 --no-budget --a-int32-maps --a fastmace.model.tiled:TILE_WHOLE_BLOCK_ORDER=True --mode auto $X --order $order --warmup 3 --blocks 4 --calls 2 --tag=-defaults-$cfg --output $OUT/ab > $OUT/ab-$cfg-$order.log 2>&1 < /dev/null
    note "end $cfg $order exit=$? $(tail -1 $OUT/ab-$cfg-$order.log | cut -c1-700)"
  done
done
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg
note done
