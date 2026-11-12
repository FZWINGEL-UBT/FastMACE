#!/bin/bash -l
# compose-outside-graph (12 November 2026, results/compose-outside-graph-20261112) on the Alex A100-SXM4-80GB, the
# A100 production plan, OMAT-0 medium float32:
#   1. gpu_smoke.py: compositions inside a capture per side (a = main's bind, b, b2), bitwise weights and forces;
#   2. the same-process MD-step A/B in both build orders (tools/bench/ab_swap.py: side a binds as main, composing
#      inside the capture), SrTiO3 10x10x10 and NbCl5, 1 fs, skin 0.5, 6 x 256 steps; proof CAPTURE_ADOPTS (b only).
# Caches and the code tree on node-local $TMPDIR (atuin file quota). Usage: sbatch alex_compose.sh SHA
# (tar $B/src-SHA.tar; $B/SrTiO3.cif as in alex_lp.sh)
#SBATCH --job-name=fm-compose
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
out=$B/runs/compose-$SLURM_JOB_ID; mkdir -p "$out"; log=$out/run.log
note() { echo "$(TZ=Europe/Berlin date +%FT%T%z) $*" | tee -a "$log"; }
src=$TMPDIR/src; mkdir -p $src && tar -xf $B/src-$SHA.tar -C $src
site=$(echo $E/env/lib/python3.*/site-packages)
export PATH=$E/env/bin:/usr/bin:/bin VIRTUAL_ENV=$E/env PYTHONPATH=$src:$src/tests PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib
export FASTMACE_CACHE=$TMPDIR/cache FASTMACE_MODELS=$B/models FASTMACE_STRUCTURES=$E/baseline FASTMACE_SOURCE_COMMIT=$SHA
export FASTMACE_RESULTS=$out
export TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor XDG_CACHE_HOME=$TMPDIR/xdg
export MACE_COMPILE_CACHE=$TMPDIR/compile TMPDIR=$TMPDIR
n=$(nproc); export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n PYTHONUNBUFFERED=1
cd $src || exit 4
note "compose job $SLURM_JOB_ID on $(hostname); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $SHA"
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=$B/SrTiO3.cif
K="fastmace.rewrites.mh1.modules:FirstInteractionReassociation.bind=compose_in_graph_bind"
X=fastmace.rewrites.mh1.modules:CAPTURE_ADOPTS
for s in NbCl5 "$st@4x4x4"; do
  tag=$(basename ${s%%@*} .cif)
  python results/compose-outside-graph-20261112/gpu_smoke.py --checkpoint $ck --structure "$s" --steps 300 \
    --output $out/smoke-$tag.json > $out/smoke-$tag.log 2>&1 < /dev/null
  note "smoke $tag exit=$? $(grep -h '\"ok\"' $out/smoke-$tag.json)"
done
for order in ab ba; do
  pattern=$([ $order = ab ] && echo abab || echo abba)
  python tools/bench/ab_swap.py --swap $K --output $out/ab-$order --checkpoint $ck --model omat0-medium \
    --structures "$st@10x10x10" NbCl5 --warmup 200 --block 256 --blocks 6 --order $order --pattern $pattern \
    --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 --b-expect $X > $out/ab-$order.log 2>&1 < /dev/null
  note "ab $order exit=$? (3: proof failed) $(grep -h -E 'b/a|wrong side' $out/ab-$order.log | tr '\n' ' ' | cut -c1-600)"
done
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg $TMPDIR/compile
note done
