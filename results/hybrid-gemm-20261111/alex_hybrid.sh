#!/bin/bash -l
# Hybrid GEMM lab rerun on one Alex A100-SXM4-80GB (11 Nov 2026; job 4983700 failed on the column-major ldb):
# hybrid_lab.py with blocking launches (errors only), then timed: CUTLASS hybrid against production 3xTF32 and
# cuBLAS FP32. Submit from $M/src-<commit>.
#SBATCH --job-name=fm-hybrid-gemm
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=00:40:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -uo pipefail
B=/home/atuin/v130cb/v130cb18/fm-a100omat
M=/home/atuin/v130cb/v130cb18/fm-a100colab
src=$SLURM_SUBMIT_DIR
module purge >/dev/null 2>&1
site=$(echo $B/env/lib/python3.*/site-packages)
export PATH=$B/env/bin:/usr/bin:/bin VIRTUAL_ENV=$B/env PYTHONPATH=$src PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib
export FASTMACE_CACHE=$M/cache FASTMACE_MODELS=$B/models FASTMACE_STRUCTURES=$B/baseline
n=$(nproc); export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n
OUT=$M/runs/review-$SLURM_JOB_ID
mkdir -p $OUT
cd $src
{ echo "host $(hostname) job $SLURM_JOB_ID source $(cat .source_commit 2>/dev/null)"
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader; } > $OUT/host.txt
CUDA_LAUNCH_BLOCKING=1 python results/hybrid-gemm-20261111/hybrid_lab.py --no-time --output $OUT/hybrid_check.json   > $OUT/hybrid_check.log 2>&1
echo "hybrid_check exit $?" >> $OUT/host.txt
python results/hybrid-gemm-20261111/hybrid_lab.py --output $OUT/hybrid_lab.json --reps 50 --blocks 7   > $OUT/hybrid_lab.log 2>&1
echo "hybrid_lab exit $?" >> $OUT/host.txt
