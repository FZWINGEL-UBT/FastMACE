# Sourced by the local (RTX 5060 Ti, WSL2) job files of paper experiment 1. Run inside tools/ci/local_gpu.py run.
# Guards (conductor rules, 11 Nov 2026): memory fraction 0.9 (wrap_frac.py), a watchdog on GPU-process memory
# > 14.8 GiB, card > 16,000 MiB, new soft-lockup / RCU-stall kernel lines, and a per-step wall limit (3x expected).
set -u
RUN=${RUN:?set RUN}; SRC=${SRC:?set SRC}; OUT=${OUT:?set OUT}
mkdir -p $OUT $RUN/tmp
E=$HOME/environments/mace_cueq_oeq; SITE=$E/lib/python3.10/site-packages; C=$HOME/fastmace-cache/paper-exp1
export PATH=$E/bin:/usr/lib/wsl/lib:$PATH PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1 FASTMACE_MODELS=$HOME/models FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18 \
  FASTMACE_CACHE=$C TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton MACE_COMPILE_CACHE=$C/compile \
  XDG_CACHE_HOME=$C/xdg TMPDIR=$RUN/tmp TORCHINDUCTOR_COMPILE_THREADS=4 CUDA_CACHE_DISABLE=1 \
  FASTMACE_HYBRID_PYTHON=$E/bin/python FASTMACE_SOURCE_COMMIT=$(cat $SRC/.source_commit) \
  LD_LIBRARY_PATH=/usr/lib/wsl/lib:$SITE/nvidia/cu13/lib:$SITE/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
unset PYTORCH_CUDA_ALLOC_CONF
CK=$HOME/models/mace-omat-0-medium.model
CIF=$SRC/results/loup-a100-20261103/SrTiO3.cif
cd $SRC
kmsg() { sudo -n dmesg 2>/dev/null || journalctl -k -b --no-pager -o cat 2>/dev/null; }
lockups() { kmsg | grep -c -E 'soft lockup|rcu.*detected stalls'; }
note() { echo "$(date '+%F %T %Z') $*" | tee -a $OUT/run.log; }
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
  note "== $name load $(cut -d' ' -f1-3 /proc/loadavg) limit ${limit} s"
  "$@" > $OUT/$name.log 2>&1 &
  pid=$!; t0=$(date +%s)
  watch $pid $name $limit
  wait $pid; code=$?
  [ -e $OUT/watchdog-$name.txt ] && code=99
  note "end $name exit=$code took=$(( $(date +%s) - t0 ))s peak=$(cat $OUT/peak-$name.txt 2>/dev/null) MiB"
  if [ $code != 0 ]; then tail -25 $OUT/$name.log | cut -c1-400 | tee -a $OUT/run.log; note "STOP after $name"; exit 1; fi; }
PY="python $SRC/results/paper-exp1-20261112/jobs/wrap_frac.py 0.9"
