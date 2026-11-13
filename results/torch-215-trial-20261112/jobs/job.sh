#!/bin/bash
# PyTorch 2.15 trial on the local RTX 5060 Ti (12 November 2026). a = production 2.14.1+cu130 (Python 3.10.12,
# ~/environments/mace_cueq_oeq), b = 2.15.0+cu132 (Python 3.11.16, ~/environments/mace_cueq_oeq_torch215, same
# cuEquivariance 0.12.0 / OpenEquivariance 0.7.0 / mace-torch 0.3.16 pins), c = b + cupti-python 13.5 (profiler
# sample only). One engine per process, OMAT-0 medium float32, the planner's default plan (production rewrites).
#   1 compat.py in a and b
#   2 cold compile: SrTiO3 10x10x10 point with an emptied cache, a then b
#   3 MD-step A/B, warm caches: SrTiO3 10x10x10 and NbCl5 3x3x3, processes a b a b b a b a (ABAB then BABA),
#     warm-up 200 steps, 1000 timed, 1 fs, NVT 300 K
#   4 profiles (20 steps after the timed run): a and b with Kineto, c with Cuspy + graph py-stacks
#   5 the 3x float32 gate (SrTiO3 3x3x3, NbCl5): production b against the 2.14.1 reference (a), then production a
# Watchdog per process as in results/shuffled-order-5060ti-20261111 (RSS 14.8 GiB, card 16,000 MiB, a new
# soft-lockup / RCU-stall line, wall time). Memory fraction 0.9. Runs inside one local_gpu reservation.
set -u
RUN=${RUN:-$HOME/fm-t215}
SRC=$RUN/src OUT=$RUN/out JOBS=$RUN/jobs
mkdir -p "$OUT/tmp" "$RUN/cache"
export PATH=/usr/lib/wsl/lib:$PATH
export PYTHONPATH=$SRC FM_SRC=$SRC PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FASTMACE_MODELS=${MODELS:-$HOME/models} \
  FASTMACE_STRUCTURES=${STRUCTURES:-$HOME/fastmace-baseline/1a-T18} TMPDIR=$OUT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 \
  FASTMACE_SOURCE_COMMIT=$(cat "$SRC/.source_commit") MEMORY_FRACTION=0.9
unset PYTORCH_CUDA_ALLOC_CONF LD_LIBRARY_PATH
cd "$SRC"
kmsg() { sudo -n dmesg 2>/dev/null || journalctl -k -b --no-pager -o cat 2>/dev/null; }
lockups() { kmsg | grep -c -E 'soft lockup|rcu.*detected stalls'; }
note() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/run.log"; }
card() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
rss_kib() { local p=$1 t=0; for q in $p $(pgrep -P $p) $(for c in $(pgrep -P $p); do pgrep -P $c; done); do
  t=$(( t + $(awk '/VmRSS/{print $2}' /proc/$q/status 2>/dev/null || echo 0) )); done; echo $t; }
cachevars() { local c=$1; export FASTMACE_CACHE=$c TORCHINDUCTOR_CACHE_DIR=$c/inductor TRITON_CACHE_DIR=$c/triton \
  MACE_COMPILE_CACHE=$c/compile XDG_CACHE_HOME=$c/xdg; mkdir -p "$c"; }
pyof() { echo "$JOBS/py-$1.sh"; }

# watched NAME LIMIT_S COMMAND...: run, watch, log; returns the exit code (99: watchdog)
watched() { local name=$1 limit=$2; shift 2
  note "== $name (card $(card) MiB)"
  "$@" > "$OUT/$name.log" 2>&1 < /dev/null &
  local pid=$! t0=$(date +%s) lock0=$(lockups) peak=0 rpeak=0 why="" used r
  while kill -0 "$pid" 2>/dev/null; do
    used=$(card); r=$(rss_kib $pid)
    [ "$used" -gt "$peak" ] 2>/dev/null && peak=$used
    [ "$r" -gt "$rpeak" ] && rpeak=$r
    [ "$used" -gt "${CARD_LIMIT_MIB:-16000}" ] 2>/dev/null && why="card at ${used} MiB (> ${CARD_LIMIT_MIB:-16000})"
    [ "$r" -gt "${RSS_LIMIT_KIB:-15518924}" ] && why="process RSS $(( r / 1024 )) MiB (> $(( ${RSS_LIMIT_KIB:-15518924} / 1024 )) MiB)"
    [ "$(lockups)" -gt "$lock0" ] && why="new soft lockup / RCU stall kernel line"
    [ $(( $(date +%s) - t0 )) -gt "$limit" ] && why="past ${limit} s"
    if [ -n "$why" ]; then note "WATCHDOG $name: $why, killing"; pkill -9 -P "$pid"; kill -9 "$pid"; break; fi
    sleep 2
  done
  wait "$pid"; local code=$?; [ -n "$why" ] && code=99
  note "end $name exit=$code took=$(( $(date +%s) - t0 ))s card peak=$peak MiB rss peak=$(( rpeak / 1024 )) MiB"
  return $code; }

# point ENV NAME STRUCTURE REPEAT [extra env...]: one capacity_scan point (through profile_point.py, which only
# profiles when PROFILE_OUT's run asks: the A/B points set PROFILE_STEPS=0 and are never profiled)
point() { local env=$1 name=$2 st=$3 rep=$4
  cachevars "$RUN/cache/$env"
  watched "$name" 1800 env PROFILE_OUT="$OUT/profile" PROFILE_TAG="$name" "${@:5}" bash "$(pyof $env)" \
    "$JOBS/profile_point.py" point --structure "$st" --repeat "$rep" --model omat0-medium --precision float32 \
    --output "$OUT/points/$name" --warmup 200 --steps 1000 --timestep-fs 1
  local code=$?
  grep -h "^point \|^profile" "$OUT/$name.log" | tail -2 | cut -c1-400 | tee -a "$OUT/run.log"
  [ $code != 0 ] && { tail -30 "$OUT/$name.log" | cut -c1-300 | tee -a "$OUT/run.log"; }
  return 0; }

note "source $(cat "$SRC/.source_commit"); $(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader); lockup lines $(lockups)"
for e in a b; do "$(pyof $e)" --version 2>&1 | sed "s/^/env $e: /" | tee -a "$OUT/run.log"; done

STAGE=${STAGE:-all}
if [ "$STAGE" = all ] || [ "$STAGE" = compat ]; then
  for e in a b; do
    cachevars "$RUN/cache/compat-$e"
    watched "compat-$e" 1500 env COMPAT_FORCES="$OUT/compat-$e-forces.npy" bash "$(pyof $e)" "$JOBS/compat.py" "$OUT/compat-$e.json"
    grep -h "compat done" "$OUT/compat-$e.log" | tee -a "$OUT/run.log"
  done
fi
if [ "$STAGE" = all ] || [ "$STAGE" = ab ]; then
  for e in a b; do rm -rf "$RUN/cache/$e"; point $e "cold-$e-SrTiO3" SrTiO3 10x10x10 PROFILE_STEPS=0 REFUSE_FALLBACK=1; done
  for st in SrTiO3:10x10x10 NbCl5:3x3x3; do
    s=${st%%:*} r=${st#*:} i=0
    for e in a b a b b a b a; do i=$((i + 1)); point $e "ab-$s-$i$e" $s $r PROFILE_STEPS=0 REFUSE_FALLBACK=1; done
  done
fi
if [ "$STAGE" = all ] || [ "$STAGE" = profile ]; then
  point a prof-a SrTiO3 10x10x10 PROFILE_STEPS=20 REFUSE_FALLBACK=1
  point b prof-b SrTiO3 10x10x10 PROFILE_STEPS=20 REFUSE_FALLBACK=1
  [ "${PROFILE_C:-1}" = 1 ] && point c prof-c SrTiO3 10x10x10 PROFILE_STEPS=20 PYSTACKS=1 REFUSE_FALLBACK=1
fi
if [ "$STAGE" = all ] || [ "$STAGE" = gate ]; then
  for e in b a; do
    cachevars "$RUN/cache/gate-$e"
    watched "gate-$e" 3600 bash "$(pyof a)" tools/force_stress_gate.py --output "$OUT/gate-$e" --engine fastmace \
      --model omat0-medium --checkpoint "$FASTMACE_MODELS/mace-omat-0-medium.model" \
      --structures-root "$FASTMACE_STRUCTURES" --structures "results/loup-a100-20261103/SrTiO3.cif@3x3x3" NbCl5 \
      --reference-command "bash $(pyof a)" --production-command "bash $(pyof $e)"
    grep -h -i "summary\|gate:" "$OUT/gate-$e.log" | tail -4 | cut -c1-300 | tee -a "$OUT/run.log"
  done
fi
note "JOB DONE"
