#!/usr/bin/env bash
# 1b-T11 job runner on murphy: run_murphy.sh RUN_NAME MODE SCRIPT [ARGS...]
#   MODE hybrid-launcher  through the baseline export's oeq_no_nvcc.sh (processes with a research-engine side)
#   MODE plain            the mace_cueq_oeq Python directly (FastMACE-only processes)
# Everything (outputs, caches, TMPDIR) under /mnt/homes/zwingel (ResearchModule AGENTS.md). Refuses to start
# while any compute process is on the GPU, from anyone, and records the idle check.
set -euo pipefail
name=$1; mode=$2; shift 2
work=/mnt/homes/zwingel/fastmace-work/1b-T11
src=${SRC:-$work/src}  # SRC: another source tree (the experiment copy)
run=$work/runs/$name
cache=${CACHE:-/mnt/homes/zwingel/fastmace-cache/1b-T11-murphy}  # CACHE: a fresh directory for cold-cache runs
env_dir=/mnt/homes/zwingel/environments/mace_cueq_oeq
baseline=/mnt/homes/zwingel/fastmace-baseline/1a-T18
mkdir -p "$run/tmp" "$cache"
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')
{
  echo "utc $(date -u +%FT%TZ) host $(hostname)"
  for pid in $pids; do echo "compute pid $pid user $(ps -o user= -p "$pid" || echo '?')"; done
  nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader
  echo "source commit $(cat "$src/.source_commit")"
} > "$run/idle_check.txt"
if [ -n "$pids" ]; then echo "GPU busy: $pids" >&2; cat "$run/idle_check.txt" >&2; exit 3; fi
export PYTHONPATH=$src PYTHONDONTWRITEBYTECODE=1
export FASTMACE_CACHE=$cache FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models FASTMACE_STRUCTURES=$baseline
export TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton MACE_COMPILE_CACHE=${MACE_COMPILE_CACHE_DIR:-$cache/compile}
export XDG_CACHE_HOME=$cache/xdg TMPDIR=$run/tmp TMP=$run/tmp TEMP=$run/tmp
site=$env_dir/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
case $mode in
  hybrid-launcher)
    cmd=(bash "$baseline/Experiments/Langevin_Thermostat/studies/hybrid_oeq/scripts/oeq_no_nvcc.sh" "$env_dir"
         "$cache/oeq-launcher" "$env_dir/bin/python" "$@") ;;
  plain) cmd=("$env_dir/bin/python" "$@") ;;
  *) echo "unknown mode $mode" >&2; exit 2 ;;
esac
printf '%q ' "${cmd[@]}" > "$run/command.txt"; echo >> "$run/command.txt"
if [ "${FOREGROUND:-0}" = 1 ]; then  # inside a detached queue (queue_murphy.sh)
  "${cmd[@]}" > "$run/run.log" 2>&1 < /dev/null
else
  setsid nohup "${cmd[@]}" > "$run/run.log" 2>&1 < /dev/null &
  echo "started $! ; log $run/run.log"
fi
