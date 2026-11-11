#!/bin/bash -l
# Idea 13 step 2 on the A100 (11 November 2026, results/sh-recompute-20261111): EDGE_FORCE (the w' backward writes
# dL/dv [E, 3] instead of dsh [E, 16]) forced on against off, OMAT-0 medium float32:
#   1. tests/gpu/test_edge_force_cuda.py (forces on against off, each end-of-edge reduction);
#   2. the 3x float32 force/stress gate with EDGE_FORCE=True (SrTiO3 3x3x3, NaCl);
#   3. the same-process MD-step A/B in both build orders (b = on), SrTiO3 10x10x10, 1 fs, skin 0.5, 6 x 256 steps;
#   4. capacity points at REPEAT (default 36x36x36), off and on: the steady peak reserved;
#   5. capacity scans from 36^3, off and on: the largest passing size.
# Usage: sbatch alex_edge_force.sh SHA [REPEAT]   (SRC tarball $B/src-SHA.tar)
#SBATCH --job-name=fm-edgeforce
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-arena/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-arena
E=/home/atuin/v130cb/v130cb18/fm-omat0small
SHA=${1:?SHA}; REPEAT=${2:-36x36x36}
out=$B/runs/edgeforce-$SLURM_JOB_ID; mkdir -p "$out"; log=$out/run.log
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
note "edgeforce job $SLURM_JOB_ID on $(hostname); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $SHA"
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=$B/SrTiO3.cif
K=fastmace.kernels.oeq_streaming.wprime_backward:EDGE_FORCE
C=tools/bench/capacity_scan.py
python -m pytest -q -p no:cacheprovider -rs tests/gpu/test_edge_force_cuda.py > $out/gputest.log 2>&1 < /dev/null
note "gpu test exit=$? $(tail -n 1 $out/gputest.log)"
python tools/force_stress_gate.py --output $out/gate --engine fastmace --model omat0-medium --checkpoint $ck \
  --structures-root $FASTMACE_STRUCTURES --structures "$st@3x3x3" NaCl \
  --production-constant $K=True > $out/gate.log 2>&1 < /dev/null
note "gate exit=$?"
grep -hE "summary" $out/gate.log | tail -n 3 | sed 's/^/  /' | tee -a "$log"
for order in ab ba; do
  pattern=$([ $order = ab ] && echo abab || echo abba)
  python tools/ab_rewrites.py --output $out/ab-$order --checkpoint $ck --model omat0-medium \
    --structures "$st@10x10x10" --warmup 200 --block 256 --blocks 6 --order $order --pattern $pattern \
    --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 --a-constant $K=False --b-constant $K=True \
    > $out/ab-$order.log 2>&1 < /dev/null
  note "ab $order exit=$? $(grep -h -E 'b/a' $out/ab-$order.log | tr '\n' ' ' | cut -c1-300)"
done
for side in off on; do
  value=$([ $side = on ] && echo True || echo False)
  timeout 50m python $C point --structure SrTiO3 --model omat0-medium --repeat $REPEAT --steps 100 --warmup 20 \
    --output $out/point-$side --constant $K=$value > $out/point-$side.log 2>&1 < /dev/null
  note "point $side exit=$? $(grep -h 'steady peak' $out/point-$side.log | tail -n 1 | cut -c1-200)"
done
for side in off on; do
  value=$([ $side = on ] && echo True || echo False)
  timeout 55m python $C scan --structure SrTiO3 --model omat0-medium --steps 50 --warmup 20 --start-atoms 233280 \
    --budget-minutes 40 --output $out/scan-$side --constant $K=$value > $out/scan-$side.log 2>&1 < /dev/null
  note "scan $side exit=$? $(tail -n 2 $out/scan-$side.log | tr '\n' ' ' | cut -c1-300)"
done
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg
note done
