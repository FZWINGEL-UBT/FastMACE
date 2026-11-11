#!/bin/bash -l
# GPU idle-gap profile of one steady tiled call on one A100-SXM4-80GB (11 November 2026, idea 2 of
# plans/ideas-20261111-tiled.md): gapprof.py (host self time per tiled method; idle gaps by host label;
# CUDA runtime calls by label), SrTiO3 58^3 (975,560 atoms), device state, R = 16,384, TILED_SPEED_GPUS defaults.
#SBATCH --job-name=fm-tiledgap
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-tiledgate/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-tiledgate
SHA=${1:?SHA}
OUT=$B/runs/gap-$SLURM_JOB_ID
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
cp $B/gapprof.py $OUT/
python $OUT/gapprof.py 58x58x58 $OUT/gapprof-58.json > $OUT/gapprof.log 2>&1
echo "gapprof exit=$? $(date +%T) $(tail -1 $OUT/gapprof.log | cut -c1-600)" >> $OUT/run.log
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg
echo done >> $OUT/run.log
