#!/bin/bash
# matrix.sh with a w' register budget per variant (12 November 2026, results/torch-215-trial-20261112): the SASS census
# found NVRTC 13.2+ schedules w' loads next to their use at the default 80 registers, and --maxrregcount=96/128 moves
# them far ahead (on the Ada, 23 KiB shared per block caps an SM at 4 blocks, so up to 128 registers costs no
# occupancy). Spec "name:env:nvrtc:regs", regs = wprime_backward.MAX_REGISTERS (an int) or "-" (the GPU's default).
# Output OUT (default $RUN/matrix2). One engine per process, OMAT-0 medium float32, the planner's default plan, fallbacks refused.
# Variants (VARIANTS, default all), each "name:env:nvrtc" (env = py-<env>.sh; nvrtc = a pip nvidia-cuda-nvrtc version
# compiled into $NV/nvrtc-<v>, or "-" for the environment's own NVRTC):
#   a   production: torch 2.14.1+cu130, cuBLAS 13.1.1, NVRTC 13.0
#   c4  a with cuBLAS 13.4.1.3 (what the torch 2.15 cu132 wheel ships)
#   c8  a with cuBLAS 13.8.2.12 (the newest on PyPI)
#   n2 / n4 / n5   a with our NVRTC kernels compiled by NVRTC 13.2 / 13.4 / 13.5 (runtime.NVRTC, a child process)
#   t9  a on Python 3.11.16 with Triton 3.9.0 (torch 2.15's; no cp310 wheels) in place of 3.8.0, --no-deps
#   b   torch 2.15.0+cu132 (cuBLAS 13.4.1, NVRTC 13.2, Triton 3.9)
#   b0  b with our kernels compiled by NVRTC 13.0
# Order per structure: all variants forward, then reversed (both build orders for every pair). Then one profile process
# per variant (20 profiled steps after the timed run), with the NVRTC sources dumped (DUMP_SOURCES) for the SASS census.
# Env: RUN (output root), NV (NVRTC wheel dirs), ENVROOT/MODELS/STRUCTURES as job.sh, CARD_LIMIT_MIB, RSS_LIMIT_KIB, REPS.
set -u
RUN=${RUN:-$HOME/fm-t215}
SRC=$RUN/src OUT=${OUT:-$RUN/matrix2} JOBS=$RUN/jobs NV=${NV:?NV}
mkdir -p "$OUT/tmp" "$RUN/cache"
export PATH=/usr/lib/wsl/lib:$PATH
export PYTHONPATH=$SRC FM_SRC=$SRC PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FASTMACE_MODELS=${MODELS:-$HOME/models} \
  FASTMACE_STRUCTURES=${STRUCTURES:-$HOME/fastmace-baseline/1a-T18} TMPDIR=$OUT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 \
  FASTMACE_SOURCE_COMMIT=$(cat "$SRC/.source_commit") MEMORY_FRACTION=0.9 REFUSE_FALLBACK=1
unset PYTORCH_CUDA_ALLOC_CONF LD_LIBRARY_PATH
cd "$SRC"
VARIANTS=${VARIANTS:-"a:a:-:- r96:a:-:96 r128:a:-:128 n5:a:13.5.37:- n5r128:a:13.5.37:128"}
kmsg() { sudo -n dmesg 2>/dev/null || journalctl -k -b --no-pager -o cat 2>/dev/null; }
lockups() { kmsg | grep -c -E 'soft lockup|rcu.*detected stalls'; }
note() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/run.log"; }
card() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
rss_kib() { local p=$1 t=0; for q in $p $(pgrep -P $p) $(for c in $(pgrep -P $p); do pgrep -P $c; done); do
  t=$(( t + $(awk '/VmRSS/{print $2}' /proc/$q/status 2>/dev/null || echo 0) )); done; echo $t; }
watched() { local name=$1 limit=$2; shift 2
  "$@" > "$OUT/$name.log" 2>&1 < /dev/null &
  local pid=$! t0=$(date +%s) lock0=$(lockups) peak=0 why="" used r
  while kill -0 "$pid" 2>/dev/null; do
    used=$(card); r=$(rss_kib $pid)
    [ "$used" -gt "$peak" ] 2>/dev/null && peak=$used
    [ "$used" -gt "${CARD_LIMIT_MIB:-16000}" ] 2>/dev/null && why="card at ${used} MiB"
    [ "$r" -gt "${RSS_LIMIT_KIB:-15518924}" ] && why="process RSS $(( r / 1024 )) MiB"
    [ "$(lockups)" -gt "$lock0" ] && why="new soft lockup / RCU stall kernel line"
    [ $(( $(date +%s) - t0 )) -gt "$limit" ] && why="past ${limit} s"
    if [ -n "$why" ]; then note "WATCHDOG $name: $why, killing"; pkill -9 -P "$pid"; kill -9 "$pid"; break; fi
    sleep 2
  done
  wait "$pid"; local code=$?; [ -n "$why" ] && code=99
  note "end $name exit=$code took=$(( $(date +%s) - t0 ))s card peak=$peak MiB"
  return $code; }

# point VARIANT_SPEC NAME STRUCTURE REPEAT [env...]
point() { local spec=$1 name=$2 st=$3 rep=$4; shift 4
  local v=${spec%%:*} rest=${spec#*:}; local env=${rest%%:*} r2=${rest#*:}; local nvrtc=${r2%%:*} regs=${r2#*:}
  local c=$RUN/cache/$v extra=() xld=""
  export FASTMACE_CACHE=$c TORCHINDUCTOR_CACHE_DIR=$c/inductor TRITON_CACHE_DIR=$c/triton MACE_COMPILE_CACHE=$c/compile \
    XDG_CACHE_HOME=$c/xdg; mkdir -p "$c"
  if [ "$nvrtc" != - ]; then
    local lib=$NV/nvrtc-$nvrtc/nvidia/cu13/lib/libnvrtc.so.13
    extra=("fastmace.kernels.oeq_streaming.runtime:NVRTC='$lib'"); xld=$(dirname "$lib")
  fi
  if [ "$regs" != - ]; then extra+=("fastmace.kernels.oeq_streaming.wprime_backward:MAX_REGISTERS=$regs"); fi
  [ ${#extra[@]} -gt 0 ] && extra=(--constant "${extra[@]}")  # one --constant (nargs="*": a second flag replaces it)
  watched "$name" 1800 env EXTRA_LD_LIBRARY_PATH="$xld" PROFILE_OUT="$OUT/profile" PROFILE_TAG="$name" "$@" \
    bash "$JOBS/py-$env.sh" "$JOBS/profile_point.py" point --structure "$st" --repeat "$rep" --model omat0-medium \
    --precision float32 --output "$OUT/points/$name" --warmup 200 --steps 1000 --timestep-fs 1 "${extra[@]}"
  grep -a -h "^point " "$OUT/$name.log" | tail -1 | cut -c1-120 | tee -a "$OUT/run.log"; }

note "matrix start: source $(cat "$SRC/.source_commit"); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); variants $VARIANTS"
for spec in $VARIANTS; do e=${spec#*:}; e=${e%%:*}
  bash "$JOBS/py-$e.sh" -c "import torch, importlib.metadata as m; print('env', '$spec', torch.__version__, 'cublas', m.version('nvidia-cublas'))" 2>/dev/null | tee -a "$OUT/run.log"
done
# warm every variant's cache once (not timed)
for spec in $VARIANTS; do point "$spec" "warm-${spec%%:*}" SrTiO3 10x10x10 PROFILE_STEPS=0; done
fwd=($VARIANTS); rev=(); for (( i=${#fwd[@]}-1; i>=0; i-- )); do rev+=("${fwd[$i]}"); done
for st in SrTiO3:10x10x10 NbCl5:3x3x3; do s=${st%%:*} r=${st#*:}
  for rep in $(seq 1 ${REPS:-1}); do
    for spec in "${fwd[@]}"; do point "$spec" "ab-$s-f$rep-${spec%%:*}" $s $r PROFILE_STEPS=0; done
    for spec in "${rev[@]}"; do point "$spec" "ab-$s-r$rep-${spec%%:*}" $s $r PROFILE_STEPS=0; done
  done
done
for spec in $VARIANTS; do
  point "$spec" "prof-${spec%%:*}" SrTiO3 10x10x10 PROFILE_STEPS=20 DUMP_SOURCES="$OUT/sources/${spec%%:*}"
done
note "MATRIX DONE"
