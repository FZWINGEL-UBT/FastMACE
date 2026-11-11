#!/bin/bash -l
# Idea 13 step 2 on the A100: per-tensor memory profile of the MD case with EDGE_FORCE off and on (Alex 4984748 found
# the 36x36x36 steady peak reserved 14.42 -> 16.12 GiB with it on, active +0.54 GB; this finds who holds the extra).
# Usage: sbatch alex_edge_force_memory.sh SHA [REPEAT]   (SRC tarball $B/src-SHA.tar; REPEAT default 24x24x24)
#SBATCH --job-name=fm-edgeforce-mem
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=01:30:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-arena/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-arena
E=/home/atuin/v130cb/v130cb18/fm-omat0small
SHA=${1:?SHA}; REPEAT=${2:-24x24x24}
out=$B/runs/edgeforce-mem-$SLURM_JOB_ID; mkdir -p "$out"; log=$out/run.log
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
note "edgeforce-mem job $SLURM_JOB_ID on $(hostname); $(nvidia-smi --query-gpu=name --format=csv,noheader); $SHA $REPEAT"
K=fastmace.kernels.oeq_streaming.wprime_backward:EDGE_FORCE
for side in off on; do
  value=$([ $side = on ] && echo True || echo False)
  timeout 30m python tools/bench/memory_profile.py --case md --structure SrTiO3 --repeat $REPEAT --warmup 5 --steps 5 \
    --constant $K=$value --output $out/$side > $out/$side.log 2>&1 < /dev/null
  note "$side exit=$? $(tail -n 3 $out/$side.log | tr '\n' ' ' | cut -c1-300)"
done
python tools/bench/memory_profile.py --diff $out/off/memprofile-*.json $out/on/memprofile-*.json > $out/diff.log 2>&1
note "diff exit=$?"
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg
note done
