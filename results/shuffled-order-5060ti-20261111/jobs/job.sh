#!/bin/bash
# B4 on the local RTX 5060 Ti (11 November 2026): MD step time, OMAT-0 medium, default plan, SrTiO3 24x24x24
# (69,120 atoms), atoms in ASE's repeat order (o) against a random permutation (s), one engine per process,
# order o s s o (both orders). Warm-up 100 steps, 600 timed. Memory fraction 0.9 (shuffled_point.py).
# Watchdog per process: process tree RSS above 14.8 GiB, card above 16,000 MiB, a new soft-lockup / RCU-stall
# kernel line, or wall time above 3x the expected (build 180 s + 700 steps at 0.25 s = 355 s; limit 1,065 s).
# Runs inside one local_gpu reservation. Stops at the first failure.
set -u
RUN=$HOME/fm-b4
SRC=$RUN/src OUT=$RUN/out
PY=$HOME/environments/mace_cueq_oeq/bin/python
SITE=$HOME/environments/mace_cueq_oeq/lib/python3.10/site-packages
CACHE=$HOME/fastmace-cache/b4
mkdir -p "$OUT/tmp" "$CACHE"
export PATH=/usr/lib/wsl/lib:$PATH
export PYTHONPATH=$SRC FM_SRC=$SRC PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$CACHE FASTMACE_MODELS=$HOME/models \
  FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18 TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor \
  TRITON_CACHE_DIR=$CACHE/triton MACE_COMPILE_CACHE=$CACHE/compile XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$OUT/tmp \
  TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_HYBRID_PYTHON=$PY FASTMACE_SOURCE_COMMIT=$(cat "$SRC/.source_commit") \
  MEMORY_FRACTION=0.9 LD_LIBRARY_PATH=$SITE/nvidia/cu13/lib:$SITE/cuequivariance_ops/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
unset PYTORCH_CUDA_ALLOC_CONF
cd "$SRC"
kmsg() { sudo -n dmesg 2>/dev/null || journalctl -k -b --no-pager -o cat 2>/dev/null; }
lockups() { kmsg | grep -c -E 'soft lockup|rcu.*detected stalls'; }
note() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/run.log"; }
card() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
rss_kib() { local p=$1 t=0; for q in $p $(pgrep -P $p) $(for c in $(pgrep -P $p); do pgrep -P $c; done); do
  t=$(( t + $(awk '/VmRSS/{print $2}' /proc/$q/status 2>/dev/null || echo 0) )); done; echo $t; }

step() { name=$1; shuffle=$2
  note "== $name (SHUFFLE=$shuffle, card $(card) MiB)"
  SHUFFLE=$shuffle "$PY" "$RUN/shuffled_point.py" point --structure SrTiO3 --repeat 24x24x24 --model omat0-medium \
    --precision float32 --output "$OUT/$name" --warmup 100 --steps 600 --timestep-fs 1 > "$OUT/$name.log" 2>&1 < /dev/null &
  pid=$!; t0=$(date +%s); lock0=$(lockups); peak=0; rpeak=0; why=""
  while kill -0 "$pid" 2>/dev/null; do
    used=$(card); r=$(rss_kib $pid)
    [ "$used" -gt "$peak" ] 2>/dev/null && peak=$used
    [ "$r" -gt "$rpeak" ] && rpeak=$r
    [ "$used" -gt 16000 ] 2>/dev/null && why="card at ${used} MiB (> 16,000)"
    [ "$r" -gt $(( 148 * 1024 * 1024 / 10 )) ] && why="process RSS $(( r / 1024 )) MiB (> 14.8 GiB)"
    [ "$(lockups)" -gt "$lock0" ] && why="new soft lockup / RCU stall kernel line"
    [ $(( $(date +%s) - t0 )) -gt 1065 ] && why="past 1,065 s (3x expected)"
    if [ -n "$why" ]; then note "WATCHDOG $name: $why, killing"; pkill -9 -P "$pid"; kill -9 "$pid"; break; fi
    sleep 2
  done
  wait "$pid"; code=$?; [ -n "$why" ] && code=99
  note "end $name exit=$code took=$(( $(date +%s) - t0 ))s card peak=$peak MiB rss peak=$(( rpeak / 1024 )) MiB"
  grep -h "^point " "$OUT/$name.log" | tail -1 | tee -a "$OUT/run.log"
  if [ "$code" != 0 ]; then tail -25 "$OUT/$name.log" | cut -c1-300; note "STOP after $name"; note "JOB DONE (failed)"; exit 1; fi; }

note "source $(cat "$SRC/.source_commit"); $(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader); lockup lines $(lockups)"
step o1 0
step s1 1
step s2 1
step o2 0
note "JOB DONE"
