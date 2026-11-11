#!/bin/bash -l
# The 3x float32 force/stress gate on one A100-SXM4-80GB (11 November 2026): TileConfig.compiled_graphs off and on
# (the A100's A/B: 0.569 / 0.571); the TILED_SPEED_GPUS defaults; tiled R = 48, compiled; OMAT-0 medium; SrTiO3
# 3x3x3, NaCl, NbCl5. Source and caches on the node's $TMPDIR; records in $B/runs.
# Usage: sbatch alex_gate.sh SHA
#SBATCH --job-name=fm-tgatecg
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=01:30:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-tiledgate/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-tiledgate
SHA=${1:?SHA}
OUT=$B/runs/gate-$SLURM_JOB_ID
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
export XDG_CACHE_HOME=$TMPDIR/xdg TMPDIR=$TMPDIR
n=$(nproc); export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n PYTHONUNBUFFERED=1
cd $SRC || exit 4
note "job $SLURM_JOB_ID on $(hostname); $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1); source $SHA; nproc $n"
for side in off cg; do
  [ $side = off ] && C="" || C="--tiled-config compiled_graphs=True"
  note "start gate-$side"
  python tools/force_stress_gate.py --output $OUT/gate-$side --engine fastmace --model omat0-medium \
    --checkpoint $FASTMACE_MODELS/mace-omat-0-medium.model --structures-root $B/structures \
    --structures results/loup-a100-20261103/SrTiO3.cif@3x3x3 NaCl NbCl5 --tiled-receivers 48 --tiled-mode compiled \
    $C > $OUT/gate-$side.log 2>&1 < /dev/null
  note "end gate-$side exit=$? $(tail -2 $OUT/gate-$side.log | tr '\n' ' ' | cut -c1-400)"
done
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg
note done
