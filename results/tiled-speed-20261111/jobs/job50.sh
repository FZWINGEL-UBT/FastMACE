#!/bin/bash
# tiled job 50 (11 November 2026): idea 11 on the GPU (idea11_gpu.py: SrTiO3 6^3 at 0 / +200 / +1,500 A, perfect and
# rattled; whole-system and tiled with TILE_LOCAL_FRAME off and on), small (1,080 atoms); under the conductor's
# watchdog (new soft lockup / RCU stall kernel lines, 15 GiB used, 1,800 s) and a 0.9 memory fraction.
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
note "== idea11 $(nvidia-smi --query-gpu=driver_version,memory.used --format=csv,noheader)"
python $RUN/idea11_gpu.py --output $OUT > $OUT/idea11.log 2>&1 &
pid=$!; base=$(lockups); peak=0; t0=$(date +%s)
while kill -0 $pid 2>/dev/null; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$used" -gt "$peak" ] 2>/dev/null && peak=$used
  why=""
  [ "$used" -gt 15360 ] 2>/dev/null && why="${used} MiB used"
  [ "$(lockups)" -gt "$base" ] && why="new soft lockup / RCU stall kernel line"
  [ $(( $(date +%s) - t0 )) -gt 1800 ] && why="past 1800 s"
  if [ -n "$why" ]; then note "WATCHDOG: $why, killing"; pkill -9 -P $pid; kill -9 $pid; break; fi
  sleep 2
done
wait $pid; code=$?
note "exit=$code peak=${peak} MiB took=$(( $(date +%s) - t0 ))s"
sed -n '/^summary/,$p' $OUT/idea11.log
tail -5 $OUT/idea11.log | cut -c1-400
echo done
