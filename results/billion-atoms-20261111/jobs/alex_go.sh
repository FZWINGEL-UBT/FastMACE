#!/bin/bash -l
# Billion-atom plan, step 3 (go/no-go, 11 November 2026): the 397.5M-atom capacity record's run again (SrTiO3 430^3 =
# 397,535,000 atoms, OMAT-0 medium, wavefront, aliased host state, state dtype auto, 2 calls) with step 1 in and the
# call's float64 positions on the host (--host-positions). Pass: GPU peak at most 28 GiB, valid, compiled call within
# 3 % of 9.84 us/atom. One A100-SXM4-80GB (not a whole node). The cgroup's memory and the GPU's every 60 s;
# Python stacks every 10 min (faulthandler); glibc mmap threshold as the record.
# Usage: sbatch alex_go.sh SHA
#SBATCH --job-name=fm-go
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-tiledgate/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-tiledgate
SHA=${1:?SHA}
OUT=$B/runs/go-$SLURM_JOB_ID
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
note "start 430"
python $TMPDIR/run.py tools/bench/tiled_capacity.py --structure SrTiO3 --model omat0-medium --repeat 430x430x430 \
  --mode auto --tile-list --host-state --alias-state --species-index --wavefront on --calls 2 --state-dtype auto \
  --host-positions --ledger --output $OUT/scan > $OUT/cap-430.log 2>&1 < /dev/null
note "end 430 exit=$? $(tail -1 $OUT/cap-430.log | cut -c1-1500)"
kill $sampler 2>/dev/null
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg
note done
