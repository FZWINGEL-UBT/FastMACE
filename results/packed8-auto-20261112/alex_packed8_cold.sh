#!/bin/bash -l
# packed8-auto (12 November 2026): PACKED8 at the MD step on cold structures (results/packed8-auto-20261112/cold_ab.sh,
# copied beside this file as packed8_cold_ab.sh: results/ is not in the slim tarball). Side a today's A100 defaults
# (PACKED8 on), side b PACKED8=False, both orders, SrTiO3 10x10x10 and NbCl5, Langevin at 1, 50 and 300 K.
# Usage: sbatch alex_packed8_cold.sh SHA [TEMPS]
#SBATCH --job-name=fm-p8cold
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=02:30:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-arena/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-arena
E=/home/atuin/v130cb/v130cb18/fm-omat0small
SHA=${1:?SHA}
out=$B/runs/p8cold-$SLURM_JOB_ID; mkdir -p "$out"
src=$TMPDIR/src; mkdir -p $src && tar -xf $B/src-$SHA.tar -C $src
site=$(echo $E/env/lib/python3.*/site-packages)
export PATH=$E/env/bin:/usr/bin:/bin VIRTUAL_ENV=$E/env PYTHONPATH=$src:$src/tests PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib
export FASTMACE_CACHE=$TMPDIR/cache FASTMACE_MODELS=$B/models FASTMACE_STRUCTURES=$E/baseline FASTMACE_SOURCE_COMMIT=$SHA
export TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor XDG_CACHE_HOME=$TMPDIR/xdg
n=$(nproc); export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n PYTHONUNBUFFERED=1
cd $src || exit 4
echo "p8cold job $SLURM_JOB_ID on $(hostname); source $SHA" > $out/job.txt
TEMPS="${2:-1 50 300}" OUT=$out bash $B/packed8_cold_ab.sh > $out/job.log 2>&1 < /dev/null
echo "exit $?" >> $out/job.log
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg
