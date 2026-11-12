#!/bin/bash -l
# WEIGHTED_VECTOR on the A100 (12 November 2026; Colab at 0 compute units, so Alex): colab_weighted_a100.sh's plan
# against today's A100-SXM4-80GB defaults: MD-step A/B in both build orders (SrTiO3 10x10x10, NbCl5 3x3x3) for
# skip+hoist+weighted and weighted alone, then the 3-structure force/stress gate for an entry under 1 in all four.
# Usage: sbatch alex_weighted_ab.sh SHA [ENTRIES]   (slim source tarball $B/src-SHA.tar; ENTRIES default "skw w")
#SBATCH --job-name=fm-weighted-ab
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-arena/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-arena
E=/home/atuin/v130cb/v130cb18/fm-omat0small
SHA=${1:?SHA}
out=$B/runs/weighted-ab-$SLURM_JOB_ID; mkdir -p "$out"
src=$TMPDIR/src; mkdir -p $src && tar -xf $B/src-$SHA.tar -C $src
site=$(echo $E/env/lib/python3.*/site-packages)
export PATH=$E/env/bin:/usr/bin:/bin VIRTUAL_ENV=$E/env PYTHONPATH=$src:$src/tests PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib
export FASTMACE_CACHE=$TMPDIR/cache FASTMACE_MODELS=$B/models FASTMACE_STRUCTURES=$E/baseline FASTMACE_SOURCE_COMMIT=$SHA
export FASTMACE_RESULTS=$out
export TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor XDG_CACHE_HOME=$TMPDIR/xdg
n=$(nproc); export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n PYTHONUNBUFFERED=1
cd $src || exit 4
echo "weighted-ab job $SLURM_JOB_ID on $(hostname); source $SHA" > $out/job.txt
# results/ is not in the slim tarball: the plan script is copied beside it
ENTRIES="${2:-skw w}" OUT=$out bash $B/colab_weighted_a100.sh > $out/job.log 2>&1 < /dev/null
echo "exit $?" >> $out/job.log
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg
