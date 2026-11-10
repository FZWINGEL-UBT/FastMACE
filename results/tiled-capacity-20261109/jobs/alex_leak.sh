#!/bin/bash -l
# Device-memory growth over a tiled call on the A100 (9 November 2026): SrTiO3 100^3 = 5M atoms, one eager call,
# allocated memory every 200 tiles and a census of live CUDA tensors after the call. Usage: sbatch alex_leak.sh SHA
#SBATCH --job-name=fm-tiledleak
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-tiledcap/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-tiledcap
SHA=${1:?SHA}
OUT=$B/runs/leak-$SLURM_JOB_ID; mkdir -p $OUT
SRC=$TMPDIR/src; mkdir -p $SRC && tar -xf $B/src-$SHA.tar -C $SRC
env_dir=/home/atuin/v130cb/v130cb18/fm-a100omat/env
site=$(echo "$env_dir"/lib/python3.*/site-packages)
export PATH=$env_dir/bin:/usr/bin:/bin VIRTUAL_ENV=$env_dir PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib
export FASTMACE_MODELS=/home/atuin/v130cb/v130cb18/fm-a100omat/models FASTMACE_SOURCE_COMMIT=$SHA
export FASTMACE_CACHE=$TMPDIR/cache TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor XDG_CACHE_HOME=$TMPDIR/xdg
n=$(nproc); export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n
cd $SRC
python $B/leakprobe2.py --structure SrTiO3 --model omat0-medium --repeat 100x100x100 --mode auto --tile-list \
  --host-state --alias-state --species-index --wavefront on --calls 1 --output $OUT/scan > $OUT/leak.log 2>&1 < /dev/null
echo "exit=$?" >> $OUT/leak.log
