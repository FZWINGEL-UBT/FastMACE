#!/bin/bash -l
# TileConfig.compiled_graphs on one A100-SXM4-80GB (11 November 2026): nsys showed the steady device-state call at
# 975,560 SrTiO3 atoms idle 52 % on the GPU, in the host's compiled tile bodies (_first) and autograd.grad. Off (a)
# against on (b), same process, both orders; OMAT-0 medium, SrTiO3 58x58x58, device state, compiled tiles.
# Usage: sbatch alex_cgab.sh SHA
#SBATCH --job-name=fm-tiledcg
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=01:30:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-tiledgate/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-tiledgate
SHA=${1:?SHA}
OUT=$B/runs/cgab-$SLURM_JOB_ID
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
for order in ab ba; do
  note "start cg $order"
  python tools/bench/tiled_ab.py --repeat 58x58x58 --receivers 8192 --mode compiled --order $order --blocks 4 \
    --calls 2 --warmup 3 --tag=-cg --b "config:compiled_graphs=True" --output $OUT/ab > $OUT/ab-cg-$order.log 2>&1 < /dev/null
  note "end cg $order exit=$? $(tail -1 $OUT/ab-cg-$order.log | cut -c1-700)"
done
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg
note done
