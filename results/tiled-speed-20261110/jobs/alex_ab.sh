#!/bin/bash -l
# Tiled speed switches on one A100-SXM4-80GB (10 November 2026): the tiled call's speed switches (TILED_SPEED_GPUS:
# the RTX 5060 Ti only) as they are on the A100 (off) against all on, same process, both orders; device state and
# the wavefront (aliased host state); OMAT-0 medium, SrTiO3 58x58x58 (975,560 atoms). Then the whole-system call at
# 18x18x18 for the per-atom ratio. Source and caches on the node's $TMPDIR; records in $B/runs.
# Usage: sbatch alex_ab.sh SHA
#SBATCH --job-name=fm-tiledab
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-tiledcap/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-tiledcap
SHA=${1:?SHA}
OUT=$B/runs/ab-$SLURM_JOB_ID
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
n=$(nproc); export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n PYTHONUNBUFFERED=1
cd $SRC || exit 4
note "job $SLURM_JOB_ID on $(hostname); $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1); source $SHA; nproc $n"
M=fastmace.model.tiled
ON="$M:TILE_PREFETCH_DEVICE=True $M:TILE_VIRIAL=True $M:TILE_GRAPH_CACHE=0.25 $M:TILE_FILTER32=True $M:TILE_EPILOGUE=True $M:TILE_PINNED_SUPERSET=True $M:TILE_GRAPH_CACHE_WAVE=True $M:TILE_HOST_WORKER=True $M:TILE_COPY_STREAM=True"
ab() { name=$1; shift; for order in ab ba; do
  note "start $name $order"
  python tools/bench/tiled_ab.py --repeat 58x58x58 --order $order --blocks 4 --calls 2 --warmup 3 --tag=-$name "$@" \
    --b $ON --output $OUT/ab > $OUT/ab-$name-$order.log 2>&1 < /dev/null
  note "end $name $order exit=$? $(tail -1 $OUT/ab-$name-$order.log | cut -c1-600)"; done; }
ab dev-all
ab wave-all --host-state --alias-state --wavefront on
note "start whole"
python - > $OUT/whole.log 2>&1 <<'PY'
import json, sys, time
sys.path.insert(0, "tools/bench")
import fastmace  # noqa: F401
import torch
import tiled_capacity as tc
import fastmace as fm
atoms = tc.supercell("SrTiO3", (18, 18, 18), 0.0, 0.0)
user = fm.ForceField(fm.load("omat0-medium"), atoms, precision="float32")
times = []
for _ in range(8):
    torch.cuda.synchronize(); t = time.perf_counter(); user.evaluate(atoms, stress=True); torch.cuda.synchronize()
    times.append(time.perf_counter() - t)
best = sorted(times[3:])[len(times[3:]) // 2]
print(json.dumps(dict(atoms=len(atoms), times=times, median_s=best, us_per_atom=best / len(atoms) * 1e6)))
PY
note "end whole exit=$? $(tail -1 $OUT/whole.log | cut -c1-400)"
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg
note done
