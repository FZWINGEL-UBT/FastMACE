#!/usr/bin/env bash
# Nested radial c/d split on the RTX 4000 Ada (the Ada's default backward is the power basis, so the split runs
# in the recompute basis: c = POWER_BACKWARD False, d = NESTED_RADIAL with POWER_BACKWARD and NESTED_POWER False;
# b = NESTED_RADIAL as shipped). Earlier header:
# Nested radial basis on the RTX 4000 Ada (11 November 2026, results/nested-radial-20261111): NESTED_RADIAL forced on
# against off, OMAT-0 medium float32: GPU kernel tests, the 3x force/stress gate, the MD-step A/B in both build orders
# (SrTiO3 10x10x10 and NbCl5). Waits for a quiet workstation (fm-ada5060/jobs/queue.sh's rule) and stops if anyone
# else starts. Usage: bash ada_nested.sh SHA  (src at /mnt/homes/zwingel/nested-radial/src-SHA from the slim tar)
set -u
SHA=${1:?SHA}
B=/mnt/homes/zwingel/nested-radial
FM=$B/src-$SHA
QUIET=2 bash /mnt/homes/zwingel/fm-ada5060/jobs/queue.sh true
export HOST=ada ROOT=$B/runs/ada-nested-cd-$SHA
mkdir -p $ROOT
s=x; source /mnt/homes/zwingel/fm-ada5060/jobs/common.sh
export FASTMACE_CACHE=$ROOT/cache TRITON_CACHE_DIR=$ROOT/cache/triton TORCHINDUCTOR_CACHE_DIR=$ROOT/cache/inductor \
  XDG_CACHE_HOME=$ROOT/cache/xdg MACE_COMPILE_CACHE=$ROOT/cache/compile FASTMACE_SOURCE_COMMIT=$SHA PYTHONUNBUFFERED=1
mkdir -p $FASTMACE_CACHE
cd $FM && export PYTHONPATH=$FM:$FM/tests
K=fastmace.rewrites.radial.nested:NESTED_RADIAL
X=fastmace.kernels.triton.tabled_outer_sum:split_walks  # proof: the split walks ran (side b only)
T=fastmace.kernels.triton.tabled_outer_sum
note "ada-nested-cd start $SHA $(nvidia-smi --query-gpu=name --format=csv,noheader)"
/mnt/homes/zwingel/environments/mace_cueq_latest/bin/python -m pytest -q -p no:cacheprovider -rs tests/gpu/test_round2_kernels_cuda.py > $ROOT/gputest.log 2>&1 < /dev/null
note "gpu test (mace_cueq_latest) exit=$? $(tail -n 1 $ROOT/gputest.log)"
python tools/far_fraction.py $ROOT/far_fraction.json "$ST@10x10x10" NbCl5 > $ROOT/far.log 2>&1 < /dev/null
note "far fraction exit=$?"; sed 's/^/  /' $ROOT/far.log | tee -a "$log"
for order in ab ba; do
  pattern=$([ $order = ab ] && echo abab || echo abba)
  if why=$(others); then note "STOP before $order: $why"; exit 3; fi
  python tools/ab_rewrites.py --output $ROOT/ab-$order --checkpoint $CK --model omat0-medium \
    --structures "$ST@10x10x10" NbCl5 --warmup 200 --block 256 --blocks 6 --order $order --pattern $pattern \
    --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
    --pack "b|$K=True;c|$T:POWER_BACKWARD=False;d|$K=True $T:POWER_BACKWARD=False $T:NESTED_POWER=False" \
    --b-expect b@$X d@$X > $ROOT/ab-$order.log 2>&1 < /dev/null
  note "ab $order exit=$? (3: proof failed) $(grep -c 'EXPECT FAILED' $ROOT/ab-$order.log) expect failures; $(grep -h -E 'b/a' $ROOT/ab-$order.log | tr '\n' ' ' | cut -c1-700)"
done
note DONE
