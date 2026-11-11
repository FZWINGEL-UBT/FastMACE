#!/bin/bash -l
# Billion-atom plan, step 1 (11 November 2026) on one A100-SXM4-80GB: the 3x float32 force/stress gate (production
# defaults: uint8 species maps, no whole-system BLOCK_ORDER order in the tiled call), the same-process A/B of
# TILE_HOST_ROWS (a: float64 positions on the device, b: on the host; tiled_ab.py --b-host-positions, each side its own
# TiledForceField, R 16,384 fixed) at SrTiO3 58^3 (975,560 atoms), host state and wavefront as the capacity record,
# both orders; then step 2's probe again (integer-valued forces) at 1M and 750M atoms.
# Usage: sbatch alex_step1.sh SHA
#SBATCH --job-name=fm-step1
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=03:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-tiledgate/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-tiledgate
SHA=${1:?SHA}
OUT=$B/runs/step1-$SLURM_JOB_ID
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
note "start gate"
python tools/force_stress_gate.py --output $OUT/gate --engine fastmace --model omat0-medium \
  --checkpoint $FASTMACE_MODELS/mace-omat-0-medium.model --structures-root $B/structures \
  --structures results/loup-a100-20261103/SrTiO3.cif@3x3x3 NaCl NbCl5 --tiled-receivers 48 --tiled-mode compiled \
  > $OUT/gate.log 2>&1 < /dev/null
note "end gate exit=$? $(tail -2 $OUT/gate.log | tr '\n' ' ' | cut -c1-400)"
for order in ab ba; do
  note "start hostrows $order"
  python tools/bench/tiled_ab.py --repeat 58x58x58 --receivers 16384 --no-budget --b-host-positions --mode auto \
    --host-state --alias-state --wavefront on --order $order --warmup 3 --blocks 4 --calls 2 --tag=-hostrows \
    --output $OUT/ab > $OUT/ab-hostrows-$order.log 2>&1 < /dev/null
  note "end hostrows $order exit=$? $(tail -1 $OUT/ab-hostrows-$order.log | cut -c1-700)"
done
for atoms in 1000000 750000000; do
  note "start probe $atoms"
  python tools/bench/overflow_probe.py --atoms $atoms --output $OUT/probe-$atoms > $OUT/probe-$atoms.log 2>&1 < /dev/null
  note "end probe $atoms exit=$? $(tail -1 $OUT/probe-$atoms.log | cut -c1-400)"
done
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg
note done
