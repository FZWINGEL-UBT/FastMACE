#!/bin/bash -l
# B3 adoption (11 November 2026): the device peak of each receiver cap, one process each: SrTiO3 160^3 (20,480,000
# atoms), the capacity record's settings (wavefront on, aliased host state, state dtype auto, 2 calls), the cap fixed
# at 16,384, 32,768 and 65,536 (--no-budget), then the default budget with TILE_LARGE_CAPS (which cap it takes).
# One A100-SXM4-80GB. Usage: sbatch alex_capspeak.sh SHA
#SBATCH --job-name=fm-capspeak
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-tiledgate/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-tiledgate
SHA=${1:?SHA}
OUT=$B/runs/capspeak-$SLURM_JOB_ID
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
export XDG_CACHE_HOME=$TMPDIR/xdg MALLOC_MMAP_THRESHOLD_=1048576
n=$(nproc); export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n PYTHONUNBUFFERED=1
cd $SRC || exit 4
cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | sed 's#/step_batch.*##')
note "job $SLURM_JOB_ID on $(hostname); cgroup limit $(cat $cg/memory.max 2>/dev/null); $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1); source $SHA; nproc $n"
( while true; do echo "$(date +%T) cg $(( $(cat $cg/memory.current 2>/dev/null || echo 0) / 1048576 )) MiB peak $(( $(cat $cg/memory.peak 2>/dev/null || echo 0) / 1048576 )) MiB gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) MiB"; sleep 60; done >> $OUT/mem.log ) &
sampler=$!
cat > $TMPDIR/run.py <<'PY'
import faulthandler, runpy, sys
faulthandler.dump_traceback_later(600, repeat=True)
sys.argv = sys.argv[1:]
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(sys.argv[0])))
runpy.run_path(sys.argv[0], run_name="__main__")
PY
for r in 16384 32768 65536 budget; do
  [ $r = budget ] && X="" || X="--no-budget --receivers $r"
  note "start $r"
  python $TMPDIR/run.py tools/bench/tiled_capacity.py --structure SrTiO3 --model omat0-medium --repeat 160x160x160     --mode auto --tile-list --host-state --alias-state --species-index --wavefront on --calls 2 --state-dtype auto     $X --output $OUT/scan-$r > $OUT/cap-$r.log 2>&1 < /dev/null
  note "end $r exit=$? $(tail -1 $OUT/cap-$r.log | cut -c1-1500)"
done
kill $sampler 2>/dev/null
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg
note done
