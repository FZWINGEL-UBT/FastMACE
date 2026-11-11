#!/bin/bash -l
# Force/stress gate of GHOIST on one Alex A100 (11 Nov 2026): tools/jobs/gate_pack.sh (default plan, then GHOIST).
#SBATCH --job-name=fm-ghoist-gate
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=01:30:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -uo pipefail
B=/home/atuin/v130cb/v130cb18/fm-a100omat
M=/home/atuin/v130cb/v130cb18/fm-a100colab
src=$M/src-cd7575bb
module purge >/dev/null 2>&1
site=$(echo $B/env/lib/python3.*/site-packages)
export PATH=$B/env/bin:/usr/bin:/bin VIRTUAL_ENV=$B/env PYTHONPATH=$src PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib
export FASTMACE_CACHE=$M/cache FASTMACE_MODELS=$B/models FASTMACE_STRUCTURES=$B/baseline
n=$(nproc); export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n
export OUT=$M/runs/gate-$SLURM_JOB_ID EXPECT_GPU=A100
export PACK="ghoist|fastmace.kernels.oeq_streaming.spline_forward:GHOIST=True"
mkdir -p $OUT
cd $src
echo "host $(hostname) job $SLURM_JOB_ID source $(cat .source_commit)" > $OUT/host.txt
bash tools/jobs/gate_pack.sh
