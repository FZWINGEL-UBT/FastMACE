#!/bin/bash -l
# Capture-arena stage 0 on one Alex A100 (10 November 2026): build the shim with the node's gcc, the toy capture, two
# MD records of SrTiO3 at REPEAT (default pool) and two with no_split, each pair compared. Records in $B/runs.
# Usage: sbatch alex_arena0.sh SHA REPEAT
#SBATCH --job-name=fm-arena0
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=01:30:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-arena/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-arena
SHA=${1:?SHA}; REPEAT=${2:-36x36x36}
OUT=$B/runs/stage0-$SLURM_JOB_ID
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
note "job $SLURM_JOB_ID on $(hostname); $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1); source $SHA; gcc $(command -v gcc || echo missing)"
S=tools/bench/arena/arena_stage0.py
{
  python "$S" build "$OUT" && echo "build exit=0"
  python "$S" toy "$OUT"; echo "toy exit=$?"
  python "$S" md "$OUT" --repeat "$REPEAT" --tag a; echo "md a exit=$?"
  python "$S" md "$OUT" --repeat "$REPEAT" --tag b; echo "md b exit=$?"
  python "$S" compare "$OUT/md-a.json" "$OUT/md-b.json"; echo "compare exit=$?"
  python "$S" md "$OUT" --repeat "$REPEAT" --tag c --no-split; echo "md c exit=$?"
  python "$S" md "$OUT" --repeat "$REPEAT" --tag d --no-split; echo "md d exit=$?"
  python "$S" compare "$OUT/md-c.json" "$OUT/md-d.json"; echo "compare no-split exit=$?"
} > "$OUT/stage0.log" 2>&1 < /dev/null
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg
note done
