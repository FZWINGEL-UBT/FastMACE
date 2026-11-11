#!/bin/bash -l
# Nested radial basis, step 3 on the A100 (11 November 2026, results/nested-radial-20261111): NESTED_RADIAL forced on
# against off, OMAT-0 medium float32:
#   1. tests/gpu/test_round2_kernels_cuda.py (the split walks against the full walks, and the round-2 kernels);
#   2. the 3x float32 force/stress gate with NESTED_RADIAL=True (SrTiO3 3x3x3, NaCl);
#   3. the same-process MD-step A/B in both build orders (b = on), SrTiO3 10x10x10 and NbCl5, 1 fs, skin 0.5,
#      6 x 256 steps.
# Usage: sbatch alex_nested_ab.sh SHA   (SRC tarball $B/src-SHA.tar)
#SBATCH --job-name=fm-nested-ab
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=02:30:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/fm-arena/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
set -uo pipefail
module purge >/dev/null 2>&1
B=/home/atuin/v130cb/v130cb18/fm-arena
E=/home/atuin/v130cb/v130cb18/fm-omat0small
SHA=${1:?SHA}
out=$B/runs/nested-ab-$SLURM_JOB_ID; mkdir -p "$out"; log=$out/run.log
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
note "nested-ab job $SLURM_JOB_ID on $(hostname); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $SHA"
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=$B/SrTiO3.cif
K=fastmace.rewrites.radial.nested:NESTED_RADIAL
X=fastmace.kernels.triton.tabled_outer_sum:SPLIT_WALKS  # proof: the split walks ran (side b only)
python -m pytest -q -p no:cacheprovider -rs tests/gpu/test_round2_kernels_cuda.py > $out/gputest.log 2>&1 < /dev/null
note "gpu test exit=$? $(tail -n 1 $out/gputest.log)"
python tools/force_stress_gate.py --output $out/gate --engine fastmace --model omat0-medium --checkpoint $ck \
  --structures-root $FASTMACE_STRUCTURES --structures "$st@3x3x3" NaCl \
  --production-constant $K=True --production-expect $X > $out/gate.log 2>&1 < /dev/null
note "gate exit=$? expect failures $(grep -c 'expected counters' $out/gate.log); $(grep -c "NESTED_RADIAL on" $out/gate.log) nested-on lines"; grep -h "NESTED_RADIAL" $out/gate.log | sort | uniq -c | head -n 5 | tee -a "$log"
grep -hE "summary" $out/gate.log | tail -n 3 | sed 's/^/  /' | tee -a "$log"
for order in ab ba; do
  pattern=$([ $order = ab ] && echo abab || echo abba)
  python tools/ab_rewrites.py --output $out/ab-$order --checkpoint $ck --model omat0-medium \
    --structures "$st@10x10x10" NbCl5 --warmup 200 --block 256 --blocks 6 --order $order --pattern $pattern \
    --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 --a-constant $K=False --b-constant $K=True --b-expect $X \
    > $out/ab-$order.log 2>&1 < /dev/null
  note "ab $order exit=$? (3: proof failed) $(grep -c 'EXPECT FAILED' $out/ab-$order.log) expect failures; nested-on lines $(grep -c 'NESTED_RADIAL on' $out/ab-$order.log); $(grep -h -E 'b/a' $out/ab-$order.log | tr '\n' ' ' | cut -c1-400)"
done
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg
note done
