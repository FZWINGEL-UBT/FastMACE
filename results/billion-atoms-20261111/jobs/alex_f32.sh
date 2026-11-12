#!/bin/bash -l
# Paper row, float32 host state (conductor, 12 November 2026): alex_go.sh's call (SrTiO3, OMAT-0 medium, tile lists,
# aliased host state, wavefront on, species index, host positions, 2 calls, mode auto) at 320,000,000 atoms (400^3)
# with --state-dtype none (float32 state, no fall to float16). If a size fails (host OOM in the 231.5 GB cgroup), the
# next smaller one while at least 4 h are left: 280.2M (383x383x382), then 239.8M (363x363x364). Manifest: the
# source, the checkpoint's SHA-256, the environment's packages (pip freeze), the GPU and driver.
# Usage: sbatch alex_f32.sh SHA
#SBATCH --job-name=fm-f32
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=08:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-tiledgate/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-tiledgate
SHA=${1:?SHA}
OUT=$B/runs/f32-$SLURM_JOB_ID
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
begin=$(date +%s)
sha256sum $FASTMACE_MODELS/* > $OUT/checkpoint.sha256
python -m pip freeze > $OUT/pip-freeze.txt 2>&1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > $OUT/gpu.csv
echo "$SHA" > $OUT/source_commit
for rep in 400x400x400 383x383x382 363x363x364; do
  left=$(( 8 * 3600 - ($(date +%s) - begin) ))
  if [ $left -lt 14400 ]; then note "skip $rep: $left s left"; break; fi
  note "start $rep"
  python $TMPDIR/run.py tools/bench/tiled_capacity.py --structure SrTiO3 --model omat0-medium --repeat $rep --mode auto --tile-list --host-state --alias-state --species-index --wavefront on --calls 2 --state-dtype none --host-positions --ledger --output $OUT/scan > $OUT/cap-$rep.log 2>&1 < /dev/null
  code=$?
  note "end $rep exit=$code $(tail -1 $OUT/cap-$rep.log | cut -c1-1500)"
  if [ $code -eq 0 ] && tail -1 $OUT/cap-$rep.log | grep -q '"error": null'; then break; fi
done
kill $sampler 2>/dev/null
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg
note done
