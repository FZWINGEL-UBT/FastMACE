#!/usr/bin/env bash
# The X4 illegal-address hunt in the multi-engine setting (conductor's lead, 10 Nov: a cache shared between engines
# freeing a tensor another engine's captured graph reads), on whichever Ada is quiet first (claim on the shared home):
# 1. compute-sanitizer 2026.2 memcheck around the X4 pack itself (ab order, abab, SrTiO3 4x4x4 and NbCl5, short blocks);
# 2. the same pack without the sanitizer, full Benchmark_loup cell, ab order, CUDA_LAUNCH_BLOCKING=1, twice;
# 3. single-engine memcheck of C and C minus each X4 switch (repro.sh), small cell.
B=/mnt/homes/zwingel/fm-ada5060
wait_for=$1
[ "$wait_for" = none ] || until grep -qE "DONE|STOP" $B/runs/$wait_for/run.log 2>/dev/null; do sleep 60; done
QUIET=2 bash $B/jobs/queue.sh true
mkdir $B/runs/memcheck 2>/dev/null || { echo "claimed elsewhere"; exit 0; }
hostname > $B/runs/memcheck/host
FM=$B/src-7d18028d
source $B/jobs/common.sh 2>/dev/null || true
CS="$B/compute-sanitizer/compute-sanitizer --tool memcheck --leak-check no --report-api-errors no --print-limit 50 --error-exitcode 99"
P=fastmace.kernels.triton.tabled_outer_sum:POWER_BACKWARD R=fastmace.kernels.triton.receiver_sum:RECEIVER_SUM
s=fastmace.kernels.oeq_streaming.spline_forward w=fastmace.kernels.oeq_streaming.wprime_backward
k=fastmace.neighbors.skin g=fastmace.rewrites.sc_generated e=fastmace.model.energy
C="$s:PACKED8=True $k:SORT_BY_LENGTH=True $g:SC_GENERATED=True $w:BUTTERFLY=True $s:BLOCK_ORDER=True $w:STAGE_DOUT=True $w:STAGE_SH=True $w:STAGE_EDGES=True $P=True $R=True $e:GRAPH_SUM=True $w:SCALAR_PATHS=True"
PACK="no-power|$P=False;no-rsum|$R=False;no-gsum|$e:GRAPH_SUM=False;no-scalar|$w:SCALAR_PATHS=False"
export HOST=ada ROOT=$B/runs/memcheck
source $B/jobs/common.sh
cd $FM && export PYTHONPATH=$FM
note "memcheck pack start (sanitizer, SrTiO3 4x4x4 + NbCl5, ab)"
# shellcheck disable=SC2086
$CS python tools/ab_rewrites.py --output $ROOT/pack-cs --checkpoint "$CK" --model omat0-medium \
  --structures "$ST@4x4x4" NbCl5 --warmup 20 --block 16 --blocks 2 --order ab --pattern abab --timestep-fs 1 \
  --a-neighbours skin=0.5 --b-neighbours skin=0.5 --a-constant $C --b-constant $C --pack "$PACK" > $ROOT/pack-cs.log 2>&1 < /dev/null
note "end pack-cs exit=$? $(grep -m5 -hE 'Invalid|ERROR SUMMARY|illegal|by thread|========= .*at ' $ROOT/pack-cs.log | tr '\n' ' ' | cut -c1-600)"
for i in 1 2; do
  mkdir -p $ROOT/pack$i
  # shellcheck disable=SC2086
  CUDA_LAUNCH_BLOCKING=1 python tools/ab_rewrites.py --output $ROOT/pack$i/ab-ab --checkpoint "$CK" --model omat0-medium \
    --structures "$ST@10x10x10" NbCl5 --warmup 200 --block 256 --blocks 6 --order ab --pattern abab --timestep-fs 1 \
    --a-neighbours skin=0.5 --b-neighbours skin=0.5 --a-constant $C --b-constant $C --pack "$PACK" > $ROOT/pack$i/ab-ab.log 2>&1 < /dev/null
  note "end pack$i exit=$? $(grep -m3 -hE 'illegal|Error' $ROOT/pack$i/ab-ab.log | tr '\n' ' ' | cut -c1-300)"
done
WRAP="$CS" CELL=4x4x4 STEPS=20 CLB=0 bash $B/jobs/repro.sh $FM C no-gsum no-scalar no-power no-rsum
