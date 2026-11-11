#!/usr/bin/env bash
# The SplineForward registry fix (2a62a9e0) on whichever Ada is quiet first (claim on the shared home):
# 1. compute-sanitizer memcheck around the X4 pack (ab order, SrTiO3 4x4x4 + NbCl5), as memcheck2.sh step 1;
# 2. X4 (leave one out from set C) in both orders, full Benchmark_loup cells (the ab order crashed before the fix);
# 3. X2 again in both orders (its ba NbCl5 cells were INVALID: another user's process on leto's GPU).
B=/mnt/homes/zwingel/fm-ada5060
FM=$B/${FIXTREE:-src-fix}
QUIET=3 bash $B/jobs/queue.sh true
mkdir $B/runs/fixverify 2>/dev/null || { echo "claimed elsewhere"; exit 0; }
hostname > $B/runs/fixverify/host
export HOST=ada ROOT=$B/runs/fixverify
source $B/jobs/common.sh
P=fastmace.kernels.triton.tabled_outer_sum:POWER_BACKWARD R=fastmace.kernels.triton.receiver_sum:RECEIVER_SUM
C="$s:PACKED8=True $k:SORT_BY_LENGTH=True $g:SC_GENERATED=True $w:BUTTERFLY=True $s:BLOCK_ORDER=True $w:STAGE_DOUT=True $w:STAGE_SH=True $w:STAGE_EDGES=True $P=True $R=True $e:GRAPH_SUM=True $w:SCALAR_PATHS=True"
export PACK_X2="no-p8|$s:PACKED8=False;no-sbl|$k:SORT_BY_LENGTH=False;no-scg|$g:SC_GENERATED=False;no-bfly|$w:BUTTERFLY=False"
export PACK_X4="no-power|$P=False;no-rsum|$R=False;no-gsum|$e:GRAPH_SUM=False;no-scalar|$w:SCALAR_PATHS=False"
export AC_X2="--a-constant $C --b-constant $C" AC_X4="--a-constant $C --b-constant $C"
CS="$B/compute-sanitizer/compute-sanitizer --tool memcheck --leak-check no --report-api-errors no --print-limit 50 --error-exitcode 99"
cd $FM && export PYTHONPATH=$FM
note "fixverify start fastmace=$(cat .source_commit)"
# shellcheck disable=SC2086
$CS python tools/ab_rewrites.py --output $ROOT/pack-cs --checkpoint "$CK" --model omat0-medium \
  --structures "$ST@4x4x4" NbCl5 --warmup 20 --block 16 --blocks 2 --order ab --pattern abab --timestep-fs 1 \
  --a-neighbours skin=0.5 --b-neighbours skin=0.5 $AC_X4 --pack "$PACK_X4" > $ROOT/pack-cs.log 2>&1 < /dev/null
note "end pack-cs exit=$? $(grep -m5 -hE 'Invalid|ERROR SUMMARY|illegal|by thread' $ROOT/pack-cs.log | tr '\n' ' ' | cut -c1-600)"
bash $B/jobs/ab2.sh $FM X4 X2
