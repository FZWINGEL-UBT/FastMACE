#!/bin/bash -l
# Nsight Systems capture of one steady tiled call on one A100-SXM4-80GB (11 November 2026; the conductor's check of
# the GPU idle share without the torch profiler): nsysprof.py under nsys (cuda, nvtx), then nsysan.py on its sqlite;
# SrTiO3 58^3 (975,560 atoms), device state, R = 16,384, TILED_SPEED_GPUS defaults.
#SBATCH --job-name=fm-tilednsys
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-tiledgate/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-tiledgate
SHA=${1:?SHA}
OUT=$B/runs/nsys-$SLURM_JOB_ID
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
cp $B/nsysprof.py $B/nsysan.py $OUT/
NSYS=/apps/SPACK/1.1.1-alex/opt/spack/linux-almalinux9-zen3/none-none/cuda-13.0.2-3oqtznz33exzhtoz4yrna6yoarbiodmz/bin/nsys
$NSYS profile -t cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop   -o $TMPDIR/capture -f true python $OUT/nsysprof.py 58x58x58 $OUT/nsysprof-58.json > $OUT/nsysprof.log 2>&1
echo "nsys exit=$? $(date +%T) $(grep '^{' $OUT/nsysprof.log | tail -1 | cut -c1-300)" >> $OUT/run.log
$NSYS export --type sqlite -o $TMPDIR/capture.sqlite -f true $TMPDIR/capture.nsys-rep >> $OUT/nsysprof.log 2>&1
python $OUT/nsysan.py $TMPDIR/capture.sqlite $OUT/nsysan-58.json > $OUT/nsysan.log 2>&1
echo "nsysan exit=$? $(date +%T) $(tail -1 $OUT/nsysan.log | cut -c1-800)" >> $OUT/run.log
$NSYS stats --report cuda_gpu_kern_sum --format csv -o $OUT/kern $TMPDIR/capture.nsys-rep >> $OUT/nsysprof.log 2>&1
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg
echo done >> $OUT/run.log
