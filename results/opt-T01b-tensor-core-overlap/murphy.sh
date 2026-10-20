#!/usr/bin/env bash
# OPT-T01b (tensor-core overlap, speed only) on murphy. Start detached:
#   setsid nohup bash murphy.sh SRC RUN_ID < /dev/null > /dev/null 2>&1 &
# One process of tools/gemm_overlap_bench.py after the idle and load check (no compute process on the
# GPU, 1-minute load below LOAD_MAX; waits up to 30 minutes). Files only under /mnt/homes/zwingel.
set -uo pipefail
src=$1; id=$2
work=/mnt/homes/zwingel/fastmace-work/opt-T01b
run=$work/runs/$id
cache=/mnt/homes/zwingel/fastmace-cache/opt-T01b-murphy
oeq=/mnt/homes/zwingel/environments/mace_cueq_oeq
baseline=/mnt/homes/zwingel/fastmace-baseline/1a-T18
LOAD_MAX=${LOAD_MAX:-6}
log=$run/run.log
mkdir -p "$run/tmp" "$cache"
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
export PYTHONPATH=$src PYTHONDONTWRITEBYTECODE=1 TRITON_CACHE_DIR=$cache/triton XDG_CACHE_HOME=$cache/xdg
export TMPDIR=$run/tmp TMP=$run/tmp TEMP=$run/tmp
site=$oeq/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib
waited=0
while true; do
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
  load=$(cut -d' ' -f1 /proc/loadavg)
  if [ -z "$pids" ] && awk -v l="$load" -v m="$LOAD_MAX" 'BEGIN{exit !(l < m)}'; then break; fi
  if [ $waited -ge 1800 ]; then note "NOT IDLE after ${waited}s: pids=[$pids] load=$load"; exit 3; fi
  sleep 30; waited=$((waited + 30))
done
note "source commit $(cat "$src/.source_commit") host $(hostname)"
note "idle (waited ${waited}s) load=$(cut -d' ' -f1-3 /proc/loadavg) gpu=$(nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader)"
note "start overlap"
bash "$baseline/Experiments/Langevin_Thermostat/studies/hybrid_oeq/scripts/oeq_no_nvcc.sh" "$oeq" "$cache/oeq-launcher" \
  "$oeq/bin/python" "$src/tools/gemm_overlap_bench.py" --output "$run/overlap" > "$run/overlap.log" 2>&1 < /dev/null
note "end overlap exit=$? load=$(cut -d' ' -f1-3 /proc/loadavg)"
note done
