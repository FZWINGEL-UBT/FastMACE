#!/bin/bash
# tiled job 60 (11 November 2026): frame off against on at a fixed R 16,384 (20^3, compiled): is the R 16,384 frame error (job 58) in this commit?
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
step frame16k 900 python $RUN/wrap_frac.py 0.9 tools/bench/tiled_ab.py --repeat 20x20x20 --receivers 16384 --no-budget \
  --mode compiled --order ab --blocks 1 --calls 1 --warmup 2 --tag=-frame16k \
  --a "fastmace.model.tiled:TILE_LOCAL_FRAME=False" --b "fastmace.model.tiled:TILE_LOCAL_FRAME=True" --output $OUT/ab
echo done
