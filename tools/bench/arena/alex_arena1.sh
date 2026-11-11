#!/bin/bash -l
# Capture-arena stage 1 (option A, EngineConfig.capture_arena) on one Alex A100 80 GB (11 November 2026), OMAT-0
# medium float32 SrTiO3, the default plan:
#   1. the GPU test tests/gpu/test_capture_arena_gpu.py (reproducible MH-1: build, plan check, bitwise against the
#      default capture, retirement);
#   2. capacity points at REPEAT, arena off and on (fresh processes; peaks, the build's record, retirement);
#   3. the same-process A/B at 10x10x10, both build orders (--prepare: the arena is built in each side's build);
#   4. capacity scans, arena off and on (the new limit).
# Records in $B/runs/stage1-<job>. Usage: sbatch alex_arena1.sh SHA [REPEAT]
#SBATCH --job-name=fm-arena1
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=03:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-arena/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-arena
SHA=${1:?SHA}; REPEAT=${2:-36x36x36}
OUT=$B/runs/stage1-$SLURM_JOB_ID
mkdir -p $OUT
log=$OUT/run.log
note() { echo "$(date +%FT%T%z) $*" | tee -a "$log"; }
SRC=$TMPDIR/src
mkdir -p $SRC && tar -xf $B/src-$SHA.tar -C $SRC
env_dir=/home/atuin/v130cb/v130cb18/fm-a100omat/env
site=$(echo "$env_dir"/lib/python3.*/site-packages)
export PATH=$env_dir/bin:/usr/bin:/bin VIRTUAL_ENV=$env_dir PYTHONPATH=$SRC:$SRC/tests PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib
export FASTMACE_MODELS=$B/models FASTMACE_SOURCE_COMMIT=$SHA FASTMACE_RESULTS=$OUT
export FASTMACE_CACHE=$TMPDIR/cache TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
export XDG_CACHE_HOME=$TMPDIR/xdg TMPDIR=$TMPDIR
n=$(nproc); export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n PYTHONUNBUFFERED=1
cd $SRC || exit 4
CK=$B/models/mace-omat-0-medium.model
note "job $SLURM_JOB_ID on $(hostname); $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1); source $SHA; gcc $(command -v gcc || echo missing)"
C=tools/bench/capacity_scan.py
{
  python -m pytest -q -p no:cacheprovider -m slow -s tests/gpu/test_capture_arena_gpu.py > $OUT/gputest.log 2>&1
  echo "gpu test exit=$? $(tail -n 1 $OUT/gputest.log)"
  for side in off on; do
    value=$([ $side = on ] && echo true || echo false)
    python $C point --structure SrTiO3 --model omat0-medium --repeat $REPEAT --steps 200 --warmup 20 \
      --output $OUT/point-$side --engine capture_arena=$value > $OUT/point-$side.log 2>&1
    echo "point $side exit=$?"
  done
  for order in ab ba; do
    pattern=$([ $order = ab ] && echo abab || echo abba)
    python tools/ab_rewrites.py --output $OUT/ab-$order --checkpoint $CK --model omat0-medium \
      --structures "$B/SrTiO3.cif@10x10x10" --warmup 200 --block 256 --blocks 6 --order $order --pattern $pattern \
      --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 --prepare \
      --a-engine capture_arena=false --b-engine capture_arena=true > $OUT/ab-$order.log 2>&1
    echo "ab $order exit=$? $(grep -h -E 'b/a' $OUT/ab-$order.log | tr '\n' ' ' | cut -c1-300)"
  done
  for side in off on; do
    value=$([ $side = on ] && echo true || echo false)
    python $C scan --structure SrTiO3 --model omat0-medium --steps 50 --warmup 20 --start-atoms 200000 \
      --budget-minutes 25 --output $OUT/scan-$side --engine capture_arena=$value > $OUT/scan-$side.log 2>&1
    echo "scan $side exit=$? $(tail -n 3 $OUT/scan-$side.log | tr '\n' ' ' | cut -c1-300)"
  done
} > "$OUT/stage1.log" 2>&1 < /dev/null
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg
note done
