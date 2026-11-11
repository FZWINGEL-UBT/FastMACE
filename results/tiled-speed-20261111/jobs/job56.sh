#!/bin/bash
# tiled job 56 (11 November 2026): the frame with float64 per row only; accuracy and the frame's A/B. (1) idea11_gpu2.py: SrTiO3 6^3 at 0 / +200 / +1,500 A, the
# whole-system call with WHOLE_SYSTEM_FLOAT64_EDGES off and on and the tiled call with TILE_LOCAL_FRAME on (forces and
# stress against the origin run); (2) the not-slower A/B of the whole-system switch (whole_system_ab.py, SrTiO3 20^3 =
# 40,000 atoms, the evaluation plan's call with stress), both orders; (3) the not-slower A/B of the tile-local frame
# (tiled_ab.py, 40^3 = 320,000 atoms, R 8,192, compiled), both orders. 0.9 memory fraction; the conductor's watchdog
# (11 November): kill on the GPU processes' memory past 14.8 GiB, the card's total past 16,000 MiB, a new kernel line
# matching "soft lockup" or "rcu.*detected stalls", or 3x the step's expected time. Stops at the first failure.
set -u
RUN=/home/fzwin/fm-runs/tiled-speed-20261108
SRC=$RUN/src-${COMMIT}
OUT=$RUN/${JOB}
mkdir -p $OUT
export PATH=/home/fzwin/environments/mace_cueq_oeq/bin:/usr/lib/wsl/lib:$PATH PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1
export FASTMACE_MODELS=/home/fzwin/models FASTMACE_SOURCE_COMMIT=$COMMIT
unset PYTORCH_CUDA_ALLOC_CONF
cd $SRC
kmsg() { sudo -n dmesg 2>/dev/null || journalctl -k -b --no-pager -o cat 2>/dev/null; }
lockups() { kmsg | grep -c -E 'soft lockup|rcu.*detected stalls'; }
note() { echo "$(date +%T) $*"; }

# watch PID STEP LIMIT_S
watch() { pid=$1; step=$2; limit=$3; base=$(lockups); peak=0; t0=$(date +%s)
  while kill -0 $pid 2>/dev/null; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    procs=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
            | awk '$1 ~ /^[0-9]+$/ {s += $1} END {print s + 0}')
    [ "$used" -gt "$peak" ] 2>/dev/null && peak=$used && echo "$peak (processes $procs)" > $OUT/peak-$step.txt
    why=""
    [ "$procs" -gt 15155 ] 2>/dev/null && why="GPU processes at ${procs} MiB (> 14.8 GiB)"
    [ "$used" -gt 16000 ] 2>/dev/null && why="card at ${used} MiB (> 16,000)"
    [ "$(lockups)" -gt "$base" ] && why="new soft lockup / RCU stall kernel line"
    [ $(( $(date +%s) - t0 )) -gt $limit ] && why="past ${limit} s (3x expected)"
    if [ -n "$why" ]; then
      note "WATCHDOG $step: $why, killing"; pkill -9 -P $pid; kill -9 $pid; echo "$why" > $OUT/watchdog-$step.txt; return
    fi
    sleep 2
  done; }

step() { name=$1; limit=$2; shift 2
  note "== $name $(cut -d' ' -f1-3 /proc/loadavg) limit ${limit} s"
  "$@" > $OUT/$name.log 2>&1 &
  pid=$!; t0=$(date +%s)
  watch $pid $name $limit
  wait $pid; code=$?
  [ -e $OUT/watchdog-$name.txt ] && code=99
  note "end $name exit=$code took=$(( $(date +%s) - t0 ))s peak=$(cat $OUT/peak-$name.txt 2>/dev/null) MiB"
  tail -1 $OUT/$name.log | cut -c1-900
  if [ $code != 0 ]; then tail -20 $OUT/$name.log | cut -c1-300; note "STOP after $name"; echo done; exit 1; fi; }

note "driver $(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader); lockup lines $(lockups)"
step accuracy 1800 python $RUN/idea11_gpu2.py --output $OUT
sed -n '/^summary/,$p' $OUT/accuracy.log
for order in ab ba; do
  step tiled-$order 1500 python $RUN/wrap_frac.py 0.9 tools/bench/tiled_ab.py --repeat 40x40x40 --receivers 8192 \
    --mode compiled --a "fastmace.model.tiled:TILE_LOCAL_FRAME=False" --b "fastmace.model.tiled:TILE_LOCAL_FRAME=True" \
    --order $order --blocks 4 --calls 2 --warmup 3 --tag=-frame40-$order --output $OUT/ab
done
echo done
