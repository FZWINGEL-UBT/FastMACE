#!/bin/bash -l
# Tiled capacity on one A100-SXM4-80GB (10 November 2026: the 397.5M record again with the TILED_SPEED_GPUS switches on and the pinned kept-lists fix; alex_cap2.sh of 9 November): OMAT-0 medium, SrTiO3, wavefront, aliased host state
# with TileConfig.state_dtype "auto" (the kept lists go first, float16 state only if still short). One process per
# size, upward from near the predicted limit; stops at the first failure. Source and caches on the node's $TMPDIR
# (the group's file quota): only the records land in $B/runs. Usage: sbatch alex_cap.sh SHA "REPEAT REPEAT ..."
#SBATCH --job-name=fm-tiledcap
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
# (Alex allots 16 CPUs and 120 GB of host memory per A100 itself: no --mem for GPU jobs)
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-tiledgate/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-tiledgate
SHA=${1:?SHA}
SIZES=${2:?sizes}
OUT=$B/runs/cap-$SLURM_JOB_ID
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
export XDG_CACHE_HOME=$TMPDIR/xdg
n=$(nproc); export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n
cd $SRC || exit 4
cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | sed 's#/step_batch.*##')
note "job $SLURM_JOB_ID on $(hostname); cgroup $cg limit $(cat $cg/memory.max 2>/dev/null); $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1); source $SHA; nproc $n; cgroup memory $(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null); MemAvailable $(awk '/MemAvailable/{print $2}' /proc/meminfo) kB"
export PYTHONUNBUFFERED=1
# Telemetry: the job cgroup's memory and the GPU's every 60 s; Python stacks every 10 min (faulthandler).
( while true; do echo "$(date +%T) cg $(( $(cat $cg/memory.current 2>/dev/null || echo 0) / 1048576 )) MiB peak $(( $(cat $cg/memory.peak 2>/dev/null || echo 0) / 1048576 )) MiB gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) MiB"; sleep 60; done >> $OUT/mem.log ) &
sampler=$!
cat > $TMPDIR/run.py <<'PY'
import faulthandler, runpy, sys
faulthandler.dump_traceback_later(600, repeat=True)
sys.argv = sys.argv[1:]
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(sys.argv[0])))
runpy.run_path(sys.argv[0], run_name="__main__")
PY
for rep in $SIZES; do
  note "start $rep"
  echo "== $rep $(date +%T)" >> $OUT/mem.log
  python $TMPDIR/run.py tools/bench/tiled_capacity.py --structure SrTiO3 --model omat0-medium --repeat ${rep}x${rep}x${rep} --mode auto \
    --tile-list --host-state --alias-state --species-index --wavefront on --calls 2 --state-dtype auto \
    --output $OUT/scan > $OUT/cap-$rep.log 2>&1 < /dev/null
  code=$?
  note "end $rep exit=$code $(tail -1 $OUT/cap-$rep.log | cut -c1-1500)"
  { [ $code = 0 ] && tail -1 $OUT/cap-$rep.log | grep -q '"phase": "done"'; } || break
done
kill $sampler 2>/dev/null
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg
note done
