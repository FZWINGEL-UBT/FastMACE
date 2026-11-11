#!/bin/bash -l
# Idea 11 on one A100-SXM4-80GB (11 November 2026): (1) the 3x float32 force/stress gate with TILE_LOCAL_FRAME and
# WHOLE_SYSTEM_FLOAT64_EDGES forced on (the size rule leaves the gate's small cells off; tiled R = 48, compiled;
# OMAT-0 medium; SrTiO3 3x3x3, NaCl, NbCl5); (1b) the +1,500 A reproduction (idea11_gpu2.py, SrTiO3 6^3); (2) the not-slower A/B of the tile-local frame (tiled_ab.py, SrTiO3 58^3 = 975,560 atoms, R 8,192,
# compiled, the A100's defaults: compiled_graphs on), both orders; (3) the not-slower A/B of the whole-system float64
# edges (whole_system_ab.py, SrTiO3 40^3 = 320,000 atoms, the evaluation plan's call with stress), both orders.
# Usage: sbatch alex_idea11.sh SHA
#SBATCH --job-name=fm-idea11
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=02:30:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-tiledgate/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-tiledgate
SHA=${1:?SHA}
OUT=$B/runs/idea11-$SLURM_JOB_ID
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
note "start gate"
python tools/force_stress_gate.py --output $OUT/gate --engine fastmace --model omat0-medium \
  --checkpoint $FASTMACE_MODELS/mace-omat-0-medium.model --structures-root $B/structures \
  --structures results/loup-a100-20261103/SrTiO3.cif@3x3x3 NaCl NbCl5 --tiled-receivers 48 --tiled-mode compiled \
  --production-constant "fastmace.model.forces:WHOLE_SYSTEM_FLOAT64_EDGES=True" "fastmace.model.tiled:TILE_LOCAL_FRAME=True" \
  > $OUT/gate.log 2>&1 < /dev/null
note "end gate exit=$? $(tail -2 $OUT/gate.log | tr '\n' ' ' | cut -c1-400)"
note "start accuracy"
python $B/idea11_gpu2.py --output $OUT/accuracy > $OUT/accuracy.log 2>&1 < /dev/null
note "end accuracy exit=$?"
sed -n '/^summary/,$p' $OUT/accuracy.log >> $log
for order in ab ba; do
  note "start frame $order"
  python tools/bench/tiled_ab.py --repeat 58x58x58 --receivers 8192 --mode compiled --order $order --blocks 4 \
    --calls 2 --warmup 3 --tag=-frame58-$order --a "fastmace.model.tiled:TILE_LOCAL_FRAME=False" \
    --b "fastmace.model.tiled:TILE_LOCAL_FRAME=True" --output $OUT/ab > $OUT/ab-frame-$order.log 2>&1 < /dev/null
  note "end frame $order exit=$? $(tail -1 $OUT/ab-frame-$order.log | cut -c1-700)"
done
for order in ab ba; do
  note "start whole $order"
  python tools/bench/whole_system_ab.py --repeat 40x40x40 --order $order --blocks 6 --calls 5 --tag=-ws40-$order \
    --a "fastmace.model.forces:WHOLE_SYSTEM_FLOAT64_EDGES=False" \
    --b "fastmace.model.forces:WHOLE_SYSTEM_FLOAT64_EDGES=True" --output $OUT/ab > $OUT/ab-whole-$order.log 2>&1 < /dev/null
  note "end whole $order exit=$? $(tail -1 $OUT/ab-whole-$order.log | cut -c1-700)"
done
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg
note done
