#!/bin/bash -l
# int32 edge_index on the A100 (11 November 2026, results/edge-index-int32-20261111): the 3x float32 force/stress gate
# with NeighbourConfig.edge_index_int32 forced on (skin.EDGE_INDEX_INT32=True; "auto" is off on the A100 until this
# passes), then the same-process MD-step A/B in both build orders (b = int32), OMAT-0 medium SrTiO3 10x10x10, 1 fs,
# skin 0.5, 6 x 256 steps per side. Usage: sbatch alex_edge32.sh SHA   (SRC tarball $B/src-SHA.tar)
#SBATCH --job-name=fm-edge32
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=01:30:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-arena/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-arena
E=/home/atuin/v130cb/v130cb18/fm-omat0small
SHA=${1:?SHA}
out=$B/runs/edge32-$SLURM_JOB_ID; mkdir -p "$out"; log=$out/run.log
note() { echo "$(date +%FT%T%z) $*" | tee -a "$log"; }
src=$TMPDIR/src; mkdir -p $src && tar -xf $B/src-$SHA.tar -C $src
site=$(echo $E/env/lib/python3.*/site-packages)
export PATH=$E/env/bin:/usr/bin:/bin VIRTUAL_ENV=$E/env PYTHONPATH=$src PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib
export FASTMACE_CACHE=$TMPDIR/cache FASTMACE_MODELS=$B/models FASTMACE_STRUCTURES=$E/baseline FASTMACE_SOURCE_COMMIT=$SHA
export TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor XDG_CACHE_HOME=$TMPDIR/xdg
n=$(nproc); export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n PYTHONUNBUFFERED=1
cd $src || exit 4
note "edge32 job $SLURM_JOB_ID on $(hostname); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $SHA"
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=$B/SrTiO3.cif
python tools/force_stress_gate.py --output $out/gate --engine fastmace --model omat0-medium --checkpoint $ck \
  --structures-root $FASTMACE_STRUCTURES --structures "$st@3x3x3" NaCl \
  --production-constant fastmace.neighbors.skin:EDGE_INDEX_INT32=True > $out/gate.log 2>&1 < /dev/null
note "gate exit=$?"
grep -hE "summary" $out/gate.log | tail -n 3 | sed 's/^/  /' | tee -a "$log"
for order in ab ba; do
  pattern=$([ $order = ab ] && echo abab || echo abba)
  python tools/ab_rewrites.py --output $out/ab-$order --checkpoint $ck --model omat0-medium \
    --structures "$st@10x10x10" --warmup 200 --block 256 --blocks 6 --order $order --pattern $pattern \
    --timestep-fs 1 --a-neighbours skin=0.5 edge_index_int32=off --b-neighbours skin=0.5 edge_index_int32=on \
    > $out/ab-$order.log 2>&1 < /dev/null
  note "ab $order exit=$? $(grep -h -E 'b/a' $out/ab-$order.log | tr '\n' ' ' | cut -c1-300)"
done
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg
note done
