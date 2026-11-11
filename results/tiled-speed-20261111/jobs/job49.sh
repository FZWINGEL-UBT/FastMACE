#!/bin/bash
# tiled job 49 (11 November 2026; job 48 stopped on the routine WSL dxg lines), after the 07:29 bugcheck (job 46:
# unguarded compiled_graphs at 975,560 atoms) and the NVIDIA driver update, one reservation for: (1) a smoke test
# (CUDA, a 4096^2 matmul, the driver); (2) job 47's GPU test on the guarded source; (3) job 46 again, stepped
# 30^3 -> 40^3 -> 50^3 (-> 58^3 only if 50^3 peaked at <= 13 GiB), compiled_graphs off (a) against on (b, the memory
# guard), both orders, under a 0.9 memory fraction and a watchdog (conductor, 11 November) that kills the step on a
# new kernel line matching "soft lockup" or "rcu.*detected stalls", nvidia-smi's used memory past 15 GiB, or 3x the
# step's expected time (30^3: 600 s; then the last step's time scaled by the atoms). Peak memory (nvidia-smi) logged
# per step. Stops at the first failure.
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

# watch PID STEP LIMIT_S: kill PID on a new lockup/stall kernel line, past 15 GiB used, or after LIMIT_S seconds;
# the step's peak (MiB, nvidia-smi) in $OUT/peak-STEP.txt
watch() { pid=$1; step=$2; limit=$3; base=$(lockups); peak=0; t0=$(date +%s)
  while kill -0 $pid 2>/dev/null; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ "$used" -gt "$peak" ] 2>/dev/null && peak=$used && echo $peak > $OUT/peak-$step.txt
    why=""
    [ "$used" -gt 15360 ] 2>/dev/null && why="${used} MiB used"
    [ "$(lockups)" -gt "$base" ] && why="new soft lockup / RCU stall kernel line"
    [ $(( $(date +%s) - t0 )) -gt $limit ] && why="past ${limit} s (3x expected)"
    if [ -n "$why" ]; then
      note "WATCHDOG $step: $why, killing"; pkill -9 -P $pid; kill -9 $pid; echo "$why" > $OUT/watchdog-$step.txt; return
    fi
    sleep 2
  done; }

note "== smoke $(cut -d' ' -f1-3 /proc/loadavg)"
note "driver $(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader)"
python -c "
import torch
assert torch.cuda.is_available()
a = torch.randn(4096, 4096, device='cuda')
b = (a @ a).sum().item()
torch.cuda.synchronize()
print('matmul ok', torch.cuda.get_device_name(), torch.version.cuda, b == b)
" > $OUT/smoke.log 2>&1
code=$?
note "smoke exit=$code $(tail -1 $OUT/smoke.log); lockup/stall kernel lines so far: $(lockups)"
[ $code = 0 ] || { note "STOP: smoke failed"; echo done; exit 1; }

note "== job47 pytest"
python -m pytest -q -s -p no:cacheprovider --basetemp=$OUT/tmp \
  "tests/gpu/test_tiled_speed_cuda.py::test_compiled_graphs_alternate_with_uncaptured_calls" > $OUT/pytest.log 2>&1 &
pid=$!
watch $pid pytest 1800
wait $pid; code=$?
[ -e $OUT/watchdog-pytest.txt ] && code=99
note "pytest exit=$code peak=$(cat $OUT/peak-pytest.txt 2>/dev/null) MiB $(tail -1 $OUT/pytest.log | cut -c1-300)"
[ $code = 0 ] || { tail -30 $OUT/pytest.log | cut -c1-400; note "STOP: pytest failed"; echo done; exit 1; }
grep -E '"states"' $OUT/pytest.log | cut -c1-600

expected=600; last_s=0; last_n=0
for size in 30 40 50 58; do
  n=$(( size * size * size ))
  if [ $last_n -gt 0 ]; then expected=$(( last_s * n / last_n )); [ $expected -lt 600 ] && expected=600; fi
  if [ $size = 58 ]; then
    p50=$(cat $OUT/peak-50-ab.txt $OUT/peak-50-ba.txt 2>/dev/null | sort -n | tail -1)
    if [ -z "$p50" ] || [ "$p50" -gt 13312 ]; then note "58^3 skipped: 50^3 peaked at ${p50:-?} MiB (> 13 GiB)"; break; fi
  fi
  for order in ab ba; do
    step=$size-$order
    note "== cg $step $(cut -d' ' -f1-3 /proc/loadavg) limit $(( 3 * expected )) s"
    python $RUN/wrap_frac.py 0.9 tools/bench/tiled_ab.py --repeat ${size}x${size}x${size} --receivers 8192 \
      --mode compiled --a "config:compiled_graphs=False" --b "config:compiled_graphs=True" --order $order \
      --blocks 3 --calls 2 --warmup 3 --tag=-cg$size --output $OUT/ab > $OUT/cg-$step.log 2>&1 &
    pid=$!
    t0=$(date +%s)
    watch $pid $step $(( 3 * expected ))
    wait $pid; code=$?
    took=$(( $(date +%s) - t0 ))
    if [ $order = ab ]; then last_s=$took; last_n=$n; fi
    note "end $step exit=$code took=${took}s peak=$(cat $OUT/peak-$step.txt 2>/dev/null) MiB $(tail -1 $OUT/cg-$step.log | cut -c1-500)"
    if [ $code != 0 ] || [ -e $OUT/watchdog-$step.txt ]; then note "STOP after $step"; echo done; exit 1; fi
  done
done
echo done
