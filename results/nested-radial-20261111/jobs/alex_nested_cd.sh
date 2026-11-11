#!/bin/bash -l
# Nested radial c/d split on the A100 (conductor, 11 November 2026, results/nested-radial-20261111): one pack, a = today's
# default against b = NESTED_RADIAL (power split), c = POWER_BACKWARD alone, d = NESTED_RADIAL with the recompute split
# (NESTED_POWER False); SrTiO3 10x10x10 and NbCl5, both build orders; proof: split_walks advances on b and d, stays 0
# on a. Then the gate on d (with the proof), the far-edge fraction and edge counts, the GPU split tests.
# Usage: sbatch alex_nested_cd.sh SHA   (SRC tarball $B/src-SHA.tar)
#SBATCH --job-name=fm-nested-cd
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
out=$B/runs/nested-cd-$SLURM_JOB_ID; mkdir -p "$out"; log=$out/run.log
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
note "nested-cd job $SLURM_JOB_ID on $(hostname); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $SHA"
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=$B/SrTiO3.cif
K=fastmace.rewrites.radial.nested:NESTED_RADIAL
T=fastmace.kernels.triton.tabled_outer_sum
X=$T:split_walks
python tools/far_fraction.py $out/far_fraction.json "$st@10x10x10" NbCl5 > $out/far.log 2>&1 < /dev/null
note "far fraction exit=$?"; sed 's/^/  /' $out/far.log | tee -a "$log"
python -m pytest -q -p no:cacheprovider -rs tests/gpu/test_round2_kernels_cuda.py > $out/gputest.log 2>&1 < /dev/null
note "gpu test exit=$? $(tail -n 1 $out/gputest.log)"
for order in ab ba; do
  pattern=$([ $order = ab ] && echo abab || echo abba)
  python tools/ab_rewrites.py --output $out/ab-$order --checkpoint $ck --model omat0-medium \
    --structures "$st@10x10x10" NbCl5 --warmup 200 --block 256 --blocks 6 --order $order --pattern $pattern \
    --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
    --pack "b|$K=True;c|$T:POWER_BACKWARD=True;d|$K=True $T:NESTED_POWER=False" --b-expect b@$X d@$X \
    > $out/ab-$order.log 2>&1 < /dev/null
  note "ab $order exit=$? (3: proof failed) $(grep -c 'EXPECT FAILED' $out/ab-$order.log) expect failures; $(grep -h -E 'b/a' $out/ab-$order.log | tr '\n' ' ' | cut -c1-700)"
done
python tools/force_stress_gate.py --output $out/gate-d --engine fastmace --model omat0-medium --checkpoint $ck \
  --structures-root $FASTMACE_STRUCTURES --structures "$st@3x3x3" NaCl \
  --production-constant $K=True $T:NESTED_POWER=False --production-expect $X > $out/gate-d.log 2>&1 < /dev/null
note "gate d exit=$? expect failures $(grep -c 'expected counters' $out/gate-d.log)"
grep -hE "summary" $out/gate-d.log | tail -n 3 | cut -c1-600 | sed 's/^/  /' | tee -a "$log"
rm -rf $TMPDIR/cache $TMPDIR/triton $TMPDIR/inductor $TMPDIR/xdg
note done
