#!/bin/bash -l
# Idea 13 step 2 on the A100: the same-process MD-step A/B of EDGE_FORCE (b = on) in both build orders on one more
# structure (the adoption rule: SrTiO3 10x10x10 in alex_edge_force.sh, NbCl5 here). OMAT-0 medium float32, 1 fs,
# skin 0.5, 6 x 256 steps.
# Usage: sbatch alex_edge_force_ab.sh SHA [STRUCTURE]   (SRC tarball $B/src-SHA.tar; STRUCTURE default NbCl5)
#SBATCH --job-name=fm-edgeforce-ab
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=01:30:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-arena/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-arena
E=/home/atuin/v130cb/v130cb18/fm-omat0small
SHA=${1:?SHA}; STRUCTURE=${2:-NbCl5}
out=$B/runs/edgeforce-ab-$SLURM_JOB_ID; mkdir -p "$out"; log=$out/run.log
note() { echo "$(date +%FT%T%z) $*" | tee -a "$log"; }
src=$TMPDIR/src; mkdir -p $src && tar -xf $B/src-$SHA.tar -C $src
site=$(echo $E/env/lib/python3.*/site-packages)
export PATH=$E/env/bin:/usr/bin:/bin VIRTUAL_ENV=$E/env PYTHONPATH=$src:$src/tests PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib
export FASTMACE_CACHE=$TMPDIR/cache FASTMACE_MODELS=$B/models FASTMACE_STRUCTURES=$E/baseline FASTMACE_SOURCE_COMMIT=$SHA
export FASTMACE_RESULTS=$out
export TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor XDG_CACHE_HOME=$TMPDIR/xdg
n=$(nproc); export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n PYTHONUNBUFFERED=1
cd $src || exit 4
note "edgeforce-ab job $SLURM_JOB_ID on $(hostname); $(nvidia-smi --query-gpu=name --format=csv,noheader); $SHA $STRUCTURE"
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
K=fastmace.kernels.oeq_streaming.wprime_backward:EDGE_FORCE
for order in ab ba; do
  pattern=$([ $order = ab ] && echo abab || echo abba)
  python tools/ab_rewrites.py --output $out/ab-$order --checkpoint $ck --model omat0-medium \
    --structures "$STRUCTURE" --warmup 200 --block 256 --blocks 6 --order $order --pattern $pattern \
    --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 --a-constant $K=False --b-constant $K=True \
    > $out/ab-$order.log 2>&1 < /dev/null
  note "ab $order exit=$? $(grep -h -E 'b/a' $out/ab-$order.log | tr '\n' ' ' | cut -c1-300)"
done
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg
note done
