#!/bin/bash -l
# Late pullback (12 November 2026, results/wprime-late-pullback-20261112) on the Alex A100-SXM4-80GB, the A100 plan:
#   1. tools/omat0_kernel_lab.py: base vs lp (LATE_PULLBACK's transform of the captured w'; edge-vector comparison);
#   2. the 3x float32 force/stress gate with LATE_PULLBACK=True (SrTiO3 3x3x3, NaCl), proof counter APPLIED;
#   3. the same-process MD-step A/B in both build orders (b = on), SrTiO3 10x10x10 and NbCl5, 1 fs, skin 0.5,
#      6 x 256 steps.
# Caches and the code tree on node-local $TMPDIR (atuin file quota). Usage: sbatch alex_lp.sh SHA (tar $B/src-SHA.tar)
#SBATCH --job-name=fm-latepull
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-arena/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-arena
E=/home/atuin/v130cb/v130cb18/fm-omat0small
SHA=${1:?SHA}
out=$B/runs/latepull-$SLURM_JOB_ID; mkdir -p "$out"; log=$out/run.log
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
note "latepull job $SLURM_JOB_ID on $(hostname); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $SHA"
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=$B/SrTiO3.cif
K=fastmace.kernels.oeq_streaming.wprime_backward:LATE_PULLBACK
X=fastmace.kernels.oeq_streaming.wprime_pullback:APPLIED  # proof: the transform ran (side b only)
python tools/omat0_kernel_lab.py --checkpoint $ck --output $out/lab --structure "$st@10x10x10" --kernels wprime \
  --variants base lp --rounds 21 > $out/lab.log 2>&1 < /dev/null
note "lab exit=$? $(grep -h -E '^wprime +(base|lp) ' $out/lab.log | tr '\n' ' ' | cut -c1-500)"
python tools/force_stress_gate.py --output $out/gate --engine fastmace --model omat0-medium --checkpoint $ck \
  --structures-root $FASTMACE_STRUCTURES --structures "$st@3x3x3" NaCl \
  --production-constant $K=True --production-expect $X > $out/gate.log 2>&1 < /dev/null
note "gate exit=$? expect failures $(grep -c 'expected counters' $out/gate.log)"
grep -hiE "summary|verdict|pass|fail" $out/gate.log | tail -n 6 | sed 's/^/  /' | tee -a "$log"
for order in ab ba; do
  pattern=$([ $order = ab ] && echo abab || echo abba)
  python tools/ab_rewrites.py --output $out/ab-$order --checkpoint $ck --model omat0-medium \
    --structures "$st@10x10x10" NbCl5 --warmup 200 --block 256 --blocks 6 --order $order --pattern $pattern \
    --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 --a-constant $K=False --b-constant $K=True \
    --b-expect $X > $out/ab-$order.log 2>&1 < /dev/null
  note "ab $order exit=$? (3: proof failed) $(grep -c 'EXPECT FAILED' $out/ab-$order.log) expect failures; $(grep -h -E 'b/a' $out/ab-$order.log | tr '\n' ' ' | cut -c1-400)"
done
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg
note done
