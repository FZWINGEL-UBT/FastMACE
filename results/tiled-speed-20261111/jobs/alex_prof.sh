#!/bin/bash -l
# Region profile of one steady tiled call on one A100-SXM4-80GB (11 November 2026, idea 2 of
# plans/ideas-20261111-tiled.md): kprof6.py (the 5060 Ti's kprof4.py, 25 kernels per region), SrTiO3 58^3
# (975,560 atoms), device state, R = 16,384, the TILED_SPEED_GPUS defaults. Then the whole-system call's kernels at
# 18^3 for the composed path's names. Usage: sbatch alex_prof.sh SHA
#SBATCH --job-name=fm-tiledprof
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-tiledgate/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-tiledgate
SHA=${1:?SHA}
OUT=$B/runs/prof-$SLURM_JOB_ID
mkdir -p $OUT
SRC=$TMPDIR/src
mkdir -p $SRC && tar -xf $B/src-$SHA.tar -C $SRC
env_dir=/home/atuin/v130cb/v130cb18/fm-a100omat/env
site=$(echo "$env_dir"/lib/python3.*/site-packages)
export PATH=$env_dir/bin:/usr/bin:/bin VIRTUAL_ENV=$env_dir PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib
export FASTMACE_MODELS=/home/atuin/v130cb/v130cb18/fm-a100omat/models FASTMACE_SOURCE_COMMIT=$SHA
export FASTMACE_CACHE=$TMPDIR/cache TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
export XDG_CACHE_HOME=$TMPDIR/xdg PYTHONUNBUFFERED=1
n=$(nproc); export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n
cd $SRC || exit 4
echo "job $SLURM_JOB_ID on $(hostname); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1); source $SHA" > $OUT/run.log
cp $B/kprof6.py $OUT/
python $OUT/kprof6.py 58x58x58 $OUT/kprof6-58.json > $OUT/kprof6.log 2>&1
echo "kprof6 exit=$? $(date +%T) $(tail -1 $OUT/kprof6.log | cut -c1-300)" >> $OUT/run.log
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg
echo done >> $OUT/run.log
